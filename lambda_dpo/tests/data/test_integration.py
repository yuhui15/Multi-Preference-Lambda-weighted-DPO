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
import tempfile
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from llamafactory.data.loader import get_dataset
from llamafactory.data.processor import ListwiseDatasetProcessor, PairwiseDatasetProcessor
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments, ModelArguments
from transformers import AutoTokenizer


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")


def create_test_dataset_info():
    """Create test dataset info for integration tests."""
    return {
        "test_pairwise": {
            "file_name": "test_pairwise.json",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        },
        "test_listwise": {
            "file_name": "test_listwise.json", 
            "formatting": "ultrafeedback",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "completions": "completions"
            }
        }
    }


def create_pairwise_data():
    """Create sample pairwise data."""
    return [
        {
            "conversations": [
                {"from": "human", "value": "What is 2+2?"}
            ],
            "chosen": {"from": "gpt", "value": "2+2 equals 4."},
            "rejected": {"from": "gpt", "value": "I don't know."}
        },
        {
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"}
            ],
            "chosen": {"from": "gpt", "value": "The capital of France is Paris."},
            "rejected": {"from": "gpt", "value": "I'm not sure."}
        }
    ]


def create_listwise_data():
    """Create sample listwise data with UltraFeedback format."""
    return [
        {
            "instruction": "Explain machine learning in simple terms.",
            "completions": [
                {
                    "response": "Machine learning is a way for computers to learn patterns from data.",
                    "annotations": {
                        "helpfulness": {"Rating": "4"},
                        "honesty": {"Rating": "5"},
                        "instruction_following": {"Rating": "4"},
                        "truthfulness": {"Rating": "5"}
                    }
                },
                {
                    "response": "ML is when computers get smart.",
                    "annotations": {
                        "helpfulness": {"Rating": "2"},
                        "honesty": {"Rating": "3"},
                        "instruction_following": {"Rating": "2"},
                        "truthfulness": {"Rating": "4"}
                    }
                },
                {
                    "response": "Machine learning is a subset of artificial intelligence that uses algorithms to automatically learn and improve from experience without being explicitly programmed.",
                    "annotations": {
                        "helpfulness": {"Rating": "5"},
                        "honesty": {"Rating": "5"},
                        "instruction_following": {"Rating": "5"},
                        "truthfulness": {"Rating": "5"}
                    }
                }
            ]
        }
    ]


@pytest.fixture
def setup_datasets():
    """Set up test datasets in temporary files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create pairwise dataset file
        import json
        pairwise_file = os.path.join(temp_dir, "test_pairwise.json")
        with open(pairwise_file, "w") as f:
            for item in create_pairwise_data():
                f.write(json.dumps(item) + "\n")
        
        # Create listwise dataset file
        listwise_file = os.path.join(temp_dir, "test_listwise.json")
        with open(listwise_file, "w") as f:
            for item in create_listwise_data():
                f.write(json.dumps(item) + "\n")
        
        yield temp_dir, pairwise_file, listwise_file


def test_automatic_processor_detection_pairwise():
    """Test that pairwise data structure triggers PairwiseDatasetProcessor selection."""
    # Test with mock data that simulates pairwise structure
    pairwise_example = {
        "_prompt": [{"role": "user", "content": "What is AI?"}],
        "_response": [
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "assistant", "content": "AI stands for artificial intelligence."}
        ]
    }
    
    # Test the processor selection logic directly
    from llamafactory.data.loader import _get_dataset_processor
    from llamafactory.hparams import DataArguments
    
    data_args = DataArguments()
    template = Mock()
    tokenizer = Mock()
    processor = Mock()
    
    dataset_processor = _get_dataset_processor(
        data_args=data_args,
        stage="rm",
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        peeked_example=pairwise_example
    )
    
    assert isinstance(dataset_processor, PairwiseDatasetProcessor)


def test_automatic_processor_detection_listwise():
    """Test that listwise data structure triggers ListwiseDatasetProcessor selection."""
    # Test with mock data that simulates listwise structure with preference data
    listwise_example = {
        "_prompt": [{"role": "user", "content": "What is AI?"}],
        "_response": [
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "assistant", "content": "AI stands for artificial intelligence."}
        ],
        "_preference_data": {
            "helpfulness": [4.0, 3.0],
            "honesty": [5.0, 4.0]
        }
    }
    
    # Test the processor selection logic directly
    from llamafactory.data.loader import _get_dataset_processor
    from llamafactory.hparams import DataArguments
    
    data_args = DataArguments()
    template = Mock()
    tokenizer = Mock()
    processor = Mock()
    
    dataset_processor = _get_dataset_processor(
        data_args=data_args,
        stage="rm",
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        peeked_example=listwise_example
    )
    
    assert isinstance(dataset_processor, ListwiseDatasetProcessor)


def test_dataset_conversion_ultrafeedback():
    """Test that UltraFeedback data is properly converted with preference data."""
    # Create a simple UltraFeedback-style dataset
    data = create_listwise_data()
    dataset = Dataset.from_list(data)
    
    from llamafactory.data.converter import get_dataset_converter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.hparams import DataArguments
    
    dataset_attr = DatasetAttr("file", "test")
    dataset_attr.prompt = "instruction"
    dataset_attr.completions = "completions"
    data_args = DataArguments()
    
    converter = get_dataset_converter("ultrafeedback", dataset_attr, data_args)
    
    # Convert the first example
    converted = converter(data[0])
    
    # Check structure
    assert "_prompt" in converted
    assert "_response" in converted
    assert "_preference_data" in converted
    
    # Check that we have 3 responses
    assert len(converted["_response"]) == 3
    
    # Check preference data structure
    pref_data = converted["_preference_data"]
    assert "helpfulness" in pref_data
    assert "honesty" in pref_data
    assert "instruction_following" in pref_data
    assert "truthfulness" in pref_data
    
    # Check that all dimensions have 3 ratings (matching 3 responses)
    for dimension, ratings in pref_data.items():
        assert len(ratings) == 3
        assert all(isinstance(r, float) for r in ratings)


def test_end_to_end_listwise_processing():
    """Test end-to-end processing of listwise data."""
    from llamafactory.data.processor.listwise import ListwiseDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    
    # Set up processor
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
    
    # Create test data in the expected format after conversion
    examples = {
        "_prompt": [
            [{"role": "user", "content": "Explain machine learning in simple terms."}]
        ],
        "_response": [
            [
                {"role": "assistant", "content": "Machine learning is a way for computers to learn patterns from data."},
                {"role": "assistant", "content": "ML is when computers get smart."},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that uses algorithms to automatically learn and improve from experience without being explicitly programmed."}
            ]
        ],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_preference_data": [
            {
                "helpfulness": [4.0, 2.0, 5.0],
                "honesty": [5.0, 3.0, 5.0],
                "instruction_following": [4.0, 2.0, 5.0],
                "truthfulness": [5.0, 4.0, 5.0]
            }
        ]
    }
    
    # Process the data
    model_inputs = listwise_processor.preprocess_dataset(examples)
    
    # Verify the output structure
    assert "input_ids" in model_inputs
    assert "labels" in model_inputs
    assert "attention_masks" in model_inputs
    assert "num_responses" in model_inputs
    assert "preference_distributions" in model_inputs
    
    # Check that we have one example with 3 responses
    assert len(model_inputs["input_ids"]) == 1
    assert model_inputs["num_responses"][0] == 3
    assert len(model_inputs["input_ids"][0]) == 3
    
    # Check preference distributions
    prefs = model_inputs["preference_distributions"][0]
    for dimension in ["helpfulness", "honesty", "instruction_following", "truthfulness"]:
        assert dimension in prefs
        # Each preference distribution should sum to ~1.0
        assert abs(sum(prefs[dimension]) - 1.0) < 1e-6