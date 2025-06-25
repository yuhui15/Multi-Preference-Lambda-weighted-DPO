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
    ) -> dict[str, list[list[int]]]:
        """Encode a single listwise example and return tokenized sequences."""
        prompt_messages = self.template.mm_plugin.process_messages(prompt, images, videos, audios, self.processor)
        prompt_ids, _ = self.template.encode_oneturn(
            self.tokenizer, prompt_messages + [{"role": "assistant", "content": ""}], system, tools
        )
        if prompt_ids and prompt_ids[-1] == self.tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:-1]
        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )

        all_input_ids, all_labels, all_attention_masks = [], [], []
        max_response_len = 0

        # First pass: find max response length
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
            "input_ids": all_input_ids,             # List[List[int]] x 4
            "labels": all_labels,
            "attention_masks": all_attention_masks,
        }

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Flatten a multi-dimensional preference dataset into DPO-compatible format."""
        model_inputs = defaultdict(list)

        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1:
                continue

            example_input_ids = []
            example_labels = []
            example_attention_masks = []
            example_pi_targets = []

            for dimension in ["helpfulness", "honesty", "instruction_following", "truthfulness"]:
                responses = examples[dimension][i]
                if not isinstance(responses, list) or len(responses) != 4:
                    continue  # skip if malformed

                encoded = self._encode_listwise_example(
                    prompt=examples["_prompt"][i],
                    responses=[{"role": "assistant", "content": r["content"]} for r in responses],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                )

                example_input_ids.extend(encoded["input_ids"])
                example_labels.extend(encoded["labels"])
                example_attention_masks.extend(encoded["attention_masks"])
                example_pi_targets.extend(examples["_pi_target"][i][dimension])

            if len(example_input_ids) != 0:
                model_inputs["input_ids"].append(example_input_ids)
                model_inputs["labels"].append(example_labels)
                model_inputs["attention_mask"].append(example_attention_masks)
                model_inputs["pi_target"].append(example_pi_targets)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """Print a listwise data example to stdout."""
        pi_targets = example.get("pi_target")

        for i, (input_ids, labels) in enumerate(zip(example["input_ids"], example["labels"])):
            valid_labels = list(filter(lambda x: x != IGNORE_INDEX, labels))
            print(f"response {i} input_ids:\n{input_ids}")
            print(
                "response {} inputs:\n{}".format(
                    i, self.tokenizer.decode(input_ids, skip_special_tokens=False)
                )
            )
            print("response {} label_ids:\n{}".format(i, labels))
            print(
                "response {} labels:\n{}".format(
                    i, self.tokenizer.decode(valid_labels, skip_special_tokens=False)
                )
            )
            if pi_targets is not None and i < len(pi_targets):
                print("response {} pi_target:\n{}".format(i, pi_targets[i]))
