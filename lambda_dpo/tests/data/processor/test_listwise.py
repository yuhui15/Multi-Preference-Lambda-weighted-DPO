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
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoTokenizer

from llamafactory.data.processor.listwise import ListwiseDatasetProcessor
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")


@pytest.fixture
def setup_processor():
    """Set up the ListwiseDatasetProcessor for testing."""
    data_args = DataArguments(cutoff_len=512, template="llama3")
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = Mock()
    
    listwise_processor = ListwiseDatasetProcessor(
        data_args=data_args,
        template=template,
        tokenizer=tokenizer,
        processor=processor,
    )
    
    return listwise_processor, tokenizer


def test_convert_ratings_to_preferences(setup_processor):
    """Test rating to preference conversion."""
    listwise_processor, _ = setup_processor
    
    # Test basic conversion
    ratings = [1.0, 2.0, 3.0]
    preferences = listwise_processor._convert_ratings_to_preferences(ratings)
    
    # Check that preferences sum to 1
    assert abs(sum(preferences) - 1.0) < 1e-6
    # Check that higher ratings get higher preferences
    assert preferences[2] > preferences[1] > preferences[0]
    
    # Test with temperature scaling
    preferences_high_temp = listwise_processor._convert_ratings_to_preferences(ratings, temperature=2.0)
    preferences_low_temp = listwise_processor._convert_ratings_to_preferences(ratings, temperature=0.5)
    
    # Higher temperature should make distribution more uniform
    # Lower temperature should make it more peaked
    high_temp_entropy = -sum(p * torch.log(torch.tensor(p + 1e-10)) for p in preferences_high_temp)
    low_temp_entropy = -sum(p * torch.log(torch.tensor(p + 1e-10)) for p in preferences_low_temp)
    assert high_temp_entropy > low_temp_entropy
    
    # Test edge cases
    assert listwise_processor._convert_ratings_to_preferences([]) == []
    
    # Test with invalid temperature
    with pytest.raises(ValueError):
        listwise_processor._convert_ratings_to_preferences([1.0, 2.0], temperature=0)
    
    with pytest.raises(ValueError):
        listwise_processor._convert_ratings_to_preferences([1.0, 2.0], temperature=-1.0)


def test_preprocess_dataset_basic(setup_processor):
    """Test basic listwise dataset preprocessing."""
    listwise_processor, tokenizer = setup_processor
    
    # Create sample data with multiple responses
    examples = {
        "_prompt": [
            [{"role": "user", "content": "What is 2+2?"}]
        ],
        "_response": [
            [
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "assistant", "content": "The answer is 4."},
                {"role": "assistant", "content": "Four."}
            ]
        ],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_preference_data": [
            {
                "helpfulness": [3.0, 4.0, 2.0],
                "honesty": [4.0, 4.0, 3.0],
                "instruction_following": [4.0, 3.0, 2.0],
                "truthfulness": [4.0, 4.0, 4.0]
            }
        ]
    }
    
    model_inputs = listwise_processor.preprocess_dataset(examples)
    
    # Check basic structure
    assert "input_ids" in model_inputs
    assert "labels" in model_inputs
    assert "attention_masks" in model_inputs
    assert "num_responses" in model_inputs
    assert "preference_distributions" in model_inputs
    
    # Check that we have one example
    assert len(model_inputs["input_ids"]) == 1
    assert model_inputs["num_responses"][0] == 3
    
    # Check that each response is properly tokenized
    example_input_ids = model_inputs["input_ids"][0]
    example_labels = model_inputs["labels"][0]
    assert len(example_input_ids) == 3  # 3 responses
    assert len(example_labels) == 3
    
    # Check preference distributions
    prefs = model_inputs["preference_distributions"][0]
    assert "helpfulness" in prefs
    assert "honesty" in prefs
    assert "instruction_following" in prefs
    assert "truthfulness" in prefs
    
    # Check that each dimension sums to 1
    for dimension, dist in prefs.items():
        assert abs(sum(dist) - 1.0) < 1e-6


def test_preprocess_dataset_validation(setup_processor):
    """Test dataset preprocessing validation."""
    listwise_processor, tokenizer = setup_processor
    
    # Test with invalid prompt (even number of messages)
    invalid_examples = {
        "_prompt": [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"}
            ]
        ],
        "_response": [[{"role": "assistant", "content": "The answer is 4."}]],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_preference_data": [None]
    }
    
    model_inputs = listwise_processor.preprocess_dataset(invalid_examples)
    # Should skip invalid examples
    assert len(model_inputs["input_ids"]) == 0
    
    # Test with insufficient responses
    insufficient_responses = {
        "_prompt": [[{"role": "user", "content": "What is 2+2?"}]],
        "_response": [[{"role": "assistant", "content": "4"}]],  # Only 1 response
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_preference_data": [None]
    }
    
    model_inputs = listwise_processor.preprocess_dataset(insufficient_responses)
    # Should skip examples with < 2 responses
    assert len(model_inputs["input_ids"]) == 0


def test_preprocess_dataset_mismatched_ratings(setup_processor):
    """Test handling of mismatched ratings and responses."""
    listwise_processor, tokenizer = setup_processor
    
    # Create data with mismatched number of ratings and responses
    examples = {
        "_prompt": [[{"role": "user", "content": "What is 2+2?"}]],
        "_response": [
            [
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "assistant", "content": "The answer is 4."},
                {"role": "assistant", "content": "Four."}
            ]
        ],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_preference_data": [
            {
                "helpfulness": [3.0, 4.0],  # Only 2 ratings for 3 responses
                "honesty": [4.0, 4.0, 3.0, 2.0],  # 4 ratings for 3 responses
            }
        ]
    }
    
    model_inputs = listwise_processor.preprocess_dataset(examples)
    
    # Should still process the example but without preference distributions for mismatched dimensions
    assert len(model_inputs["input_ids"]) == 1
    prefs = model_inputs["preference_distributions"][0]
    
    # Mismatched dimensions should not appear in preference distributions
    assert "helpfulness" not in prefs
    assert "honesty" not in prefs


def test_preprocess_dataset_no_preference_data(setup_processor):
    """Test processing without preference data."""
    listwise_processor, tokenizer = setup_processor
    
    examples = {
        "_prompt": [[{"role": "user", "content": "What is 2+2?"}]],
        "_response": [
            [
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "assistant", "content": "The answer is 4."}
            ]
        ],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_preference_data": [None]
    }
    
    model_inputs = listwise_processor.preprocess_dataset(examples)
    
    # Should still work without preference data
    assert len(model_inputs["input_ids"]) == 1
    assert model_inputs["preference_distributions"][0] == {}


def test_encode_listwise_example(setup_processor):
    """Test encoding of individual listwise examples."""
    listwise_processor, tokenizer = setup_processor
    
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    responses = [
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "assistant", "content": "Four."}
    ]
    
    encoded = listwise_processor._encode_listwise_example(
        prompt=prompt,
        responses=responses,
        system="",
        tools="",
        images=[],
        videos=[],
        audios=[]
    )
    
    # Check structure
    assert "input_ids" in encoded
    assert "labels" in encoded
    assert "attention_masks" in encoded
    assert "num_responses" in encoded
    
    assert encoded["num_responses"] == 3
    assert len(encoded["input_ids"]) == 3
    assert len(encoded["labels"]) == 3
    assert len(encoded["attention_masks"]) == 3
    
    # Check that each sequence has proper structure
    for i in range(3):
        input_ids = encoded["input_ids"][i]
        labels = encoded["labels"][i]
        attention_mask = encoded["attention_masks"][i]
        
        assert len(input_ids) == len(labels) == len(attention_mask)
        assert all(mask == 1 for mask in attention_mask)
        
        # Check that prompt tokens are ignored in labels
        ignore_count = sum(1 for label in labels if label == IGNORE_INDEX)
        assert ignore_count > 0  # Should have some ignored tokens from prompt


def test_print_data_example(setup_processor):
    """Test data example printing functionality."""
    listwise_processor, tokenizer = setup_processor
    
    # Test with valid dataset
    dataset = {
        "input_ids": [
            [
                [1, 2, 3, 4],
                [1, 2, 5, 6]
            ]
        ],
        "labels": [
            [
                [IGNORE_INDEX, IGNORE_INDEX, 3, 4],
                [IGNORE_INDEX, IGNORE_INDEX, 5, 6]
            ]
        ],
        "num_responses": [2],
        "preference_distributions": [
            {
                "helpfulness": [0.3, 0.7],
                "honesty": [0.4, 0.6]
            }
        ]
    }
    
    # Should not raise any exceptions
    listwise_processor.print_data_example(dataset)
    
    # Test with empty dataset
    empty_dataset = {"input_ids": []}
    listwise_processor.print_data_example(empty_dataset)
    
    # Test with None dataset
    listwise_processor.print_data_example({})