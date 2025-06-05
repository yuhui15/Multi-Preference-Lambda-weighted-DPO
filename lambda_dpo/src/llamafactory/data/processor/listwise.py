# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional
import numpy as np

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class ListwiseDatasetProcessor(DatasetProcessor):
    """Process datasets with multiple responses and multi-dimensional preference ratings."""
    
    def _convert_ratings_to_preferences(self, ratings: list[float], temperature: float = 1.0) -> list[float]:
        """Convert ratings to normalized preference distribution using softmax."""
        # Handle edge cases
        if not ratings:
            return []
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        ratings_array = np.array(ratings)
        # Apply temperature scaling
        scaled_ratings = ratings_array / temperature
        # Compute softmax
        exp_ratings = np.exp(scaled_ratings - np.max(scaled_ratings))  # Subtract max for numerical stability
        preferences = exp_ratings / np.sum(exp_ratings)
        return preferences.tolist()
    
    def _encode_listwise_example(
        self,
        prompt: list[dict[str, str]],
        responses: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> dict[str, Any]:
        """Encode a single example with multiple responses."""
        # Process prompt
        prompt_messages = self.template.mm_plugin.process_messages(
            prompt, images, videos, audios, self.processor
        )
        
        # Get prompt encoding (without response)
        prompt_ids, _ = self.template.encode_oneturn(
            self.tokenizer, prompt_messages + [{"role": "assistant", "content": ""}], system, tools
        )
        # Remove the assistant placeholder
        prompt_ids = prompt_ids[:-1] if prompt_ids and prompt_ids[-1] == self.tokenizer.eos_token_id else prompt_ids
        
        # Process token IDs for multimodal inputs
        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        
        # Encode each response
        all_input_ids = []
        all_labels = []
        all_attention_masks = []
        max_response_len = 0
        
        for response in responses:
            response_messages = self.template.mm_plugin.process_messages(
                prompt + [response], images, videos, audios, self.processor
            )
            _, response_ids = self.template.encode_oneturn(
                self.tokenizer, response_messages, system, tools
            )
            
            if self.template.efficient_eos:
                response_ids += [self.tokenizer.eos_token_id]
            
            max_response_len = max(max_response_len, len(response_ids))
        
        # Determine appropriate lengths
        source_len, target_len = infer_seqlen(len(prompt_ids), max_response_len, self.data_args.cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        
        # Build input sequences for each response
        for response in responses:
            response_messages = self.template.mm_plugin.process_messages(
                prompt + [response], images, videos, audios, self.processor
            )
            _, response_ids = self.template.encode_oneturn(
                self.tokenizer, response_messages, system, tools
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
        """Build model inputs from examples with listwise preferences."""
        model_inputs = defaultdict(list)
        
        for i in range(len(examples["_prompt"])):
            # Validate example
            if len(examples["_prompt"][i]) % 2 != 1:
                logger.warning_rank0(
                    f"Dropped example with invalid prompt length: {examples['_prompt'][i]}"
                )
                continue
            
            responses = examples["_response"][i]
            if not isinstance(responses, list) or len(responses) < 2:
                logger.warning_rank0(
                    f"Dropped example with insufficient responses: {len(responses) if isinstance(responses, list) else 'not a list'}"
                )
                continue
            
            # Encode the listwise example
            encoded = self._encode_listwise_example(
                prompt=examples["_prompt"][i],
                responses=responses,
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            
            # Process preference distributions if available
            preference_dists = {}
            if "_preference_data" in examples and examples["_preference_data"][i]:
                pref_data = examples["_preference_data"][i]
                
                # Convert ratings to preference distributions for each dimension
                for dimension in ["helpfulness", "honesty", "instruction_following", "truthfulness"]:
                    if dimension in pref_data and pref_data[dimension]:
                        ratings = pref_data[dimension]
                        if len(ratings) == len(responses):
                            preference_dists[dimension] = self._convert_ratings_to_preferences(ratings)
                        else:
                            logger.warning_rank0(
                                f"Mismatched ratings and responses for {dimension}: "
                                f"{len(ratings)} ratings vs {len(responses)} responses"
                            )
            
            # Add to model inputs
            model_inputs["input_ids"].append(encoded["input_ids"])
            model_inputs["labels"].append(encoded["labels"])
            model_inputs["attention_masks"].append(encoded["attention_masks"])
            model_inputs["num_responses"].append(encoded["num_responses"])
            model_inputs["preference_distributions"].append(preference_dists)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
        
        return model_inputs
    
    def print_data_example(self, dataset: dict[str, list[Any]]) -> None:
        """Print a formatted data example for debugging."""
        if not dataset or len(dataset.get("input_ids", [])) == 0:
            print("No examples to display")
            return
            
        # Get the first example
        example_idx = 0
        print(f"Number of responses: {dataset.get('num_responses', ['N/A'])[example_idx]}")
        
        if "input_ids" in dataset and dataset["input_ids"]:
            example_input_ids = dataset["input_ids"][example_idx]
            example_labels = dataset["labels"][example_idx]
            
            for idx, (input_ids, labels) in enumerate(zip(example_input_ids, example_labels)):
                print(f"\n--- Response {idx + 1} ---")
                print(f"input_ids length: {len(input_ids)}")
                print(f"inputs:\n{self.tokenizer.decode(input_ids, skip_special_tokens=False)}")
                
                valid_labels = list(filter(lambda x: x != IGNORE_INDEX, labels))
                print(f"labels length: {len(valid_labels)}")
                print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
        
        if "preference_distributions" in dataset and dataset["preference_distributions"]:
            example_prefs = dataset["preference_distributions"][example_idx]
            print("\n--- Preference Distributions ---")
            for dimension, prefs in example_prefs.items():
                print(f"{dimension}: {[f'{x:.3f}' for x in prefs]}")
