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

import os
import pytest
import torch
from unittest.mock import Mock
from transformers import AutoTokenizer
from llamafactory.data.processor.listwise import ListwiseDatasetProcessor
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
from llamafactory.extras.constants import IGNORE_INDEX

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

@pytest.fixture
def setup_processor():
    data_args = DataArguments(cutoff_len=512, template="llama3")
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = Mock()
    return ListwiseDatasetProcessor(data_args, template, tokenizer, processor), tokenizer

def test_listwise_outputs_structure(setup_processor):
    processor, tokenizer = setup_processor

    examples = {
        "_prompt": [[{"role": "user", "content": "What is 2+2?"}]],
        "helpfulness": [[
            {"model": "a", "content": "2+2 equals 4.", "score": 3.0},
            {"model": "b", "content": "The answer is 4.", "score": 4.0},
            {"model": "c", "content": "Four.", "score": 2.0}
        ]],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_pi_target": [{
            "helpfulness": [0.25, 0.5, 0.25],
            "honesty": [0.33, 0.33, 0.34]
        }]
    }

    model_inputs = processor.preprocess_dataset(examples)

    assert len(model_inputs["input_ids"]) == 1
    assert model_inputs["num_responses"][0] == 3
    assert len(model_inputs["input_ids"][0]) == 3
    assert all(len(x) == len(y) == len(z)
               for x, y, z in zip(
                   model_inputs["input_ids"][0],
                   model_inputs["labels"][0],
                   model_inputs["attention_masks"][0]))

    prefs = model_inputs["preference_distributions"][0]
    assert abs(sum(prefs["helpfulness"]) - 1.0) < 1e-6
    assert abs(sum(prefs["honesty"]) - 1.0) < 1e-6

def test_invalid_and_insufficient(setup_processor):
    processor, tokenizer = setup_processor

    examples = {
        "_prompt": [[{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]],
        "helpfulness": [[{"model": "a", "content": "Four.", "score": 1.0}]],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_pi_target": [None]
    }
    assert processor.preprocess_dataset(examples)["input_ids"] == []

    examples_insufficient = {
        "_prompt": [[{"role": "user", "content": "What is 2+2?"}]],
        "helpfulness": [[{"model": "a", "content": "4", "score": 1.0}]],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_pi_target": [None]
    }
    assert processor.preprocess_dataset(examples_insufficient)["input_ids"] == []

def test_label_ignore_index(setup_processor):
    processor, tokenizer = setup_processor

    prompt = [{"role": "user", "content": "What is 2+2?"}]
    responses = [
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "assistant", "content": "Four."}
    ]

    encoded = processor._encode_listwise_example(prompt, responses, "", "", [], [], [])
    for labels in encoded["labels"]:
        ignore_prefix = sum(1 for x in labels if x == IGNORE_INDEX)
        assert ignore_prefix > 0
