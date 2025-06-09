from collections import defaultdict
from typing import Optional, Any

from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


class ListwiseDatasetProcessor(DatasetProcessor):
    """Process datasets with multiple responses and multi-dimensional preference ratings."""

    def _encode_listwise_example(
        self,
        prompt: list[dict[str, str]],
        responses: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list[Any],
        videos: list[Any],
        audios: list[Any],
    ) -> dict[str, Any]:
        # Process prompt and get token ids
        prompt_messages = self.template.mm_plugin.process_messages(prompt, images, videos, audios, self.processor)
        prompt_ids, _ = self.template.encode_oneturn(
            self.tokenizer, prompt_messages + [{"role": "assistant", "content": ""}], system, tools
        )
        prompt_ids = prompt_ids[:-1] if prompt_ids and prompt_ids[-1] == self.tokenizer.eos_token_id else prompt_ids
        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )

        # Encode each response and determine maximum response length
        all_input_ids, all_labels, all_attention_masks = [], [], []
        max_response_len = 0
        for response in responses:
            _, response_ids = self.template.encode_oneturn(
                self.tokenizer,
                self.template.mm_plugin.process_messages(prompt + [response], images, videos, audios, self.processor),
                system,
                tools
            )
            if self.template.efficient_eos:
                response_ids += [self.tokenizer.eos_token_id]
            max_response_len = max(max_response_len, len(response_ids))

        source_len, target_len = infer_seqlen(len(prompt_ids), max_response_len, self.data_args.cutoff_len)
        prompt_ids = prompt_ids[:source_len]

        for response in responses:
            _, response_ids = self.template.encode_oneturn(
                self.tokenizer,
                self.template.mm_plugin.process_messages(prompt + [response], images, videos, audios, self.processor),
                system,
                tools
            )
            if self.template.efficient_eos:
                response_ids += [self.tokenizer.eos_token_id]
            response_ids = response_ids[:target_len]

            input_ids = prompt_ids + response_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
            attention_mask = [1] * len(input_ids)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_mask)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_masks": all_attention_masks,
            "num_responses": len(responses),
        }

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)

        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1:
                continue
            responses = examples["helpfulness"][i]
            if not isinstance(responses, list) or len(responses) < 2:
                continue

            encoded = self._encode_listwise_example(
                prompt=examples["_prompt"][i],
                responses=[{"role": "assistant", "content": r["content"]} for r in responses],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )

            model_inputs["input_ids"].append(encoded["input_ids"])
            model_inputs["labels"].append(encoded["labels"])
            model_inputs["attention_masks"].append(encoded["attention_masks"])
            model_inputs["num_responses"].append(encoded["num_responses"])
            model_inputs["preference_distributions"].append(examples["_pi_target"][i])
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs
