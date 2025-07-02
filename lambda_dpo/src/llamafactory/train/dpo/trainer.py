# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma
        self.lambda_dpo_chunk_size = finetuning_args.lambda_dpo_chunk_size

        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        if self.finetuning_args.shuffle_block_size:
            from llamafactory.data import BlockShuffleSampler

            return BlockShuffleSampler(
                self.train_dataset,
                self.finetuning_args.shuffle_block_size,
                seed=int(self.args.seed),
            )

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def get_batch_samples(self, *args, **kwargs):
        r"""Replace the method of DPO Trainer with the one of the standard Trainer."""
        return Trainer.get_batch_samples(self, *args, **kwargs)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute ORPO's odds ratio (OR) loss for batched log probabilities of the policy model."""
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute SimPO loss for batched log probabilities of the policy model."""
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute loss for preference learning."""
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        all_logits: torch.Tensor = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps
        else:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Compute log probabilities of the reference model."""
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        r"""Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

        return losses.mean(), metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        r"""Subclass and override to accept extra kwargs."""
        if self.loss_type == "lambda_dpo":
            return self._compute_lambda_dpo_loss(model, inputs, return_outputs)
        if self.loss_type == "dpo" and "pi_target" in inputs:
            return self._compute_ultra_dpo_loss(model, inputs, return_outputs)
        return super().compute_loss(model, inputs, return_outputs)

    def _compute_lambda_dpo_loss(self, model, inputs, return_outputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        pi_target = inputs["pi_target"]

        chunk_size = self.lambda_dpo_chunk_size or input_ids.size(0)

        seq_log_probs_list = []
        ref_seq_log_probs_list = []
        for start in range(0, input_ids.size(0), chunk_size):
            end = start + chunk_size
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]
            chunk_labels = labels[start:end]

            logits = model(input_ids=chunk_ids, attention_mask=chunk_mask).logits
            log_probs = F.log_softmax(logits[:, :-1], dim=-1)
            labels_shifted = chunk_labels[:, 1:].clone()
            mask = labels_shifted != self.label_pad_token_id
            labels_shifted[labels_shifted == self.label_pad_token_id] = 0
            label_log_probs = torch.gather(log_probs, dim=2, index=labels_shifted.unsqueeze(-1)).squeeze(-1)
            seq_lp = (label_log_probs * mask).sum(dim=1) / mask.sum(dim=1)
            seq_log_probs_list.append(seq_lp)

            with torch.no_grad():
                if self.finetuning_args.use_ref_model and self.ref_model is not None:
                    ref_logits = self.ref_model(input_ids=chunk_ids, attention_mask=chunk_mask).logits
                    ref_log_probs = F.log_softmax(ref_logits[:, :-1], dim=-1)
                    ref_label_log_probs = torch.gather(ref_log_probs, dim=2, index=labels_shifted.unsqueeze(-1)).squeeze(-1)
                    ref_lp = (ref_label_log_probs * mask).sum(dim=1) / mask.sum(dim=1)
                else:
                    ref_lp = torch.zeros_like(seq_lp)
            ref_seq_log_probs_list.append(ref_lp)

        seq_log_probs = torch.cat(seq_log_probs_list, dim=0)
        ref_seq_log_probs = torch.cat(ref_seq_log_probs_list, dim=0)

        r_theta = self.beta * (seq_log_probs - ref_seq_log_probs)
        B = input_ids.size(0) // 16
        r_theta = r_theta.view(B, 4, 4)
        pi_target = pi_target.view(B, 4, 4)

        listwise_losses = []
        for i in range(4):
            r = r_theta[:, i, :]
            p = pi_target[:, i, :] + 1e-10
            log_softmaxed = r - torch.logsumexp(r, dim=1, keepdim=True)
            loss = -(p * log_softmaxed).sum(dim=1).mean()
            listwise_losses.append(loss)

        num_pref = len(listwise_losses)
        weights = torch.ones(num_pref, device=input_ids.device)
        weights /= weights.sum()
        final_loss = sum(w * l for w, l in zip(weights, listwise_losses))

        if return_outputs:
            return final_loss, {"loss_components": listwise_losses}
        return final_loss

    def _compute_ultra_dpo_loss(self, model, inputs, return_outputs):
        """Compute DPO loss for the UltraFeedback listwise dataset."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        pi_target = inputs["pi_target"]

        B = input_ids.size(0) // 16
        pi_target = pi_target.view(B, 4, 4)

        # Uniform preference weights across dimensions
        pref_weights = torch.ones(4, device=pi_target.device) / 4
        aggregated = (pi_target * pref_weights.view(1, -1, 1)).sum(dim=1)

        best_resp = aggregated.argmax(dim=1)
        worst_resp = aggregated.argmin(dim=1)

        batch_offset = torch.arange(B, device=input_ids.device) * 16
        best_indices = batch_offset + best_resp
        worst_indices = batch_offset + worst_resp

        gather_idx = torch.cat([best_indices, worst_indices], dim=0)
        selected_ids = input_ids[gather_idx]
        selected_mask = attention_mask[gather_idx]
        selected_labels = labels[gather_idx]

        logits = model(input_ids=selected_ids, attention_mask=selected_mask).logits
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        labels_shifted = selected_labels[:, 1:].clone()
        mask = labels_shifted != self.label_pad_token_id
        labels_shifted[labels_shifted == self.label_pad_token_id] = 0
        label_log_probs = torch.gather(log_probs, dim=2, index=labels_shifted.unsqueeze(-1)).squeeze(-1)
        seq_log_probs = (label_log_probs * mask).sum(dim=1) / mask.sum(dim=1)

        with torch.no_grad():
            if self.finetuning_args.use_ref_model and self.ref_model is not None:
                ref_logits = self.ref_model(input_ids=selected_ids, attention_mask=selected_mask).logits
                ref_log_probs = F.log_softmax(ref_logits[:, :-1], dim=-1)
                ref_label_log_probs = torch.gather(ref_log_probs, dim=2, index=labels_shifted.unsqueeze(-1)).squeeze(-1)
                ref_seq_log_probs = (ref_label_log_probs * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                ref_seq_log_probs = torch.zeros_like(seq_log_probs)

        policy_best, policy_worst = seq_log_probs[:B], seq_log_probs[B:]
        ref_best, ref_worst = ref_seq_log_probs[:B], ref_seq_log_probs[B:]

        losses, *_ = self.compute_preference_loss(policy_best, policy_worst, ref_best, ref_worst)
        final_loss = losses.mean()
        if return_outputs:
            return final_loss, [losses]
        return final_loss

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        r"""Log `logs` on the various objects watching training, including stored metrics."""
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs, *args, **kwargs)
