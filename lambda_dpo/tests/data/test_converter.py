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

import pytest

from llamafactory.data import Role
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


def test_alpaca_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "instruction": "Solve the math problem.",
        "input": "3 + 4",
        "output": "The answer is 7.",
    }
    dataset_converter = get_dataset_converter("alpaca", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


def test_sharegpt_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "Solve the math problem.\n3 + 4"},
            {"from": "gpt", "value": "The answer is 7."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "You are a helpful assistant.",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


def test_ultrafeedback_converter():
    """Test UltrafeedbackDatasetConverter with multi-dimensional ratings."""
    dataset_attr = DatasetAttr("hf_hub", "openbmb/UltraFeedback")
    dataset_attr.prompt = "instruction"
    dataset_attr.completions = "completions"
    
    data_args = DataArguments()
    
    example = {
        "instruction": "Explain quantum computing in simple terms.",
        "completions": [
            {
                "response": "Quantum computing uses quantum mechanics to process information.",
                "annotations": {
                    "helpfulness": {"Rating": "4"},
                    "honesty": {"Rating": "5"},
                    "instruction_following": {"Rating": "4"},
                    "truthfulness": {"Rating": "5"}
                }
            },
            {
                "response": "Quantum computers are very fast computers.",
                "annotations": {
                    "helpfulness": {"Rating": "2"},
                    "honesty": {"Rating": "3"},
                    "instruction_following": {"Rating": "2"},
                    "truthfulness": {"Rating": "3"}
                }
            },
            {
                "response": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform computations that would be intractable for classical computers.",
                "annotations": {
                    "helpfulness": {"Rating": "5"},
                    "honesty": {"Rating": "5"},
                    "instruction_following": {"Rating": "5"},
                    "truthfulness": {"Rating": "5"}
                }
            }
        ]
    }
    
    dataset_converter = get_dataset_converter("ultrafeedback", dataset_attr, data_args)
    result = dataset_converter(example)
    
    expected = {
        "_prompt": [{"role": Role.USER.value, "content": "Explain quantum computing in simple terms."}],
        "_response": [
            {"role": Role.ASSISTANT.value, "content": "Quantum computing uses quantum mechanics to process information."},
            {"role": Role.ASSISTANT.value, "content": "Quantum computers are very fast computers."},
            {"role": Role.ASSISTANT.value, "content": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform computations that would be intractable for classical computers."}
        ],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
        "_preference_data": {
            "helpfulness": [4.0, 2.0, 5.0],
            "honesty": [5.0, 3.0, 5.0],
            "instruction_following": [4.0, 2.0, 5.0],
            "truthfulness": [5.0, 3.0, 5.0]
        }
    }
    
    assert result == expected


def test_ultrafeedback_converter_missing_ratings():
    """Test UltrafeedbackDatasetConverter with missing ratings."""
    dataset_attr = DatasetAttr("hf_hub", "openbmb/UltraFeedback")
    dataset_attr.prompt = "instruction"
    dataset_attr.completions = "completions"
    
    data_args = DataArguments()
    
    example = {
        "instruction": "What is AI?",
        "completions": [
            {
                "response": "AI is artificial intelligence.",
                "annotations": {
                    "helpfulness": {"Rating": "4"},
                    # Missing other dimensions
                }
            },
            {
                "response": "AI stands for artificial intelligence and refers to machines that can think.",
                "annotations": {
                    "helpfulness": {"Rating": "5"},
                    "honesty": {"Rating": "4"},
                    # Missing instruction_following and truthfulness
                }
            }
        ]
    }
    
    dataset_converter = get_dataset_converter("ultrafeedback", dataset_attr, data_args)
    result = dataset_converter(example)
    
    # Should use default rating (1.0) for missing dimensions
    assert result["_preference_data"]["helpfulness"] == [4.0, 5.0]
    assert result["_preference_data"]["honesty"] == [1.0, 4.0]  # Default for first response
    assert result["_preference_data"]["instruction_following"] == [1.0, 1.0]  # Default for both
    assert result["_preference_data"]["truthfulness"] == [1.0, 1.0]  # Default for both


def test_ultrafeedback_converter_invalid_ratings():
    """Test UltrafeedbackDatasetConverter with invalid rating values."""
    dataset_attr = DatasetAttr("hf_hub", "openbmb/UltraFeedback")
    dataset_attr.prompt = "instruction"
    dataset_attr.completions = "completions"
    
    data_args = DataArguments()
    
    example = {
        "instruction": "What is machine learning?",
        "completions": [
            {
                "response": "Machine learning is a subset of AI.",
                "annotations": {
                    "helpfulness": {"Rating": "invalid_rating"},
                    "honesty": {"Rating": None},
                    "instruction_following": {"Rating": "4"},
                    "truthfulness": {"Rating": "not_a_number"}
                }
            }
        ]
    }
    
    dataset_converter = get_dataset_converter("ultrafeedback", dataset_attr, data_args)
    result = dataset_converter(example)
    
    # Should use default rating (1.0) for invalid values
    assert result["_preference_data"]["helpfulness"] == [1.0]
    assert result["_preference_data"]["honesty"] == [1.0]
    assert result["_preference_data"]["instruction_following"] == [4.0]
    assert result["_preference_data"]["truthfulness"] == [1.0]


def test_ultrafeedback_converter_no_completions():
    """Test UltrafeedbackDatasetConverter with no completions."""
    dataset_attr = DatasetAttr("hf_hub", "openbmb/UltraFeedback")
    dataset_attr.prompt = "instruction"
    dataset_attr.completions = "completions"
    
    data_args = DataArguments()
    
    example = {
        "instruction": "What is deep learning?",
        "completions": []
    }
    
    dataset_converter = get_dataset_converter("ultrafeedback", dataset_attr, data_args)
    result = dataset_converter(example)
    
    expected = {
        "_prompt": [{"role": Role.USER.value, "content": "What is deep learning?"}],
        "_response": [],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
        "_preference_data": None,
    }
    
    assert result == expected


def test_ultrafeedback_converter_custom_default_rating():
    """Test UltrafeedbackDatasetConverter with custom default rating."""
    dataset_attr = DatasetAttr("hf_hub", "openbmb/UltraFeedback")
    dataset_attr.prompt = "instruction"
    dataset_attr.completions = "completions"
    
    data_args = DataArguments()
    
    # Create converter with custom default rating
    from llamafactory.data.converter import UltrafeedbackDatasetConverter
    converter = UltrafeedbackDatasetConverter(
        dataset_attr=dataset_attr,
        data_args=data_args,
        default_rating=2.5
    )
    
    example = {
        "instruction": "Explain neural networks.",
        "completions": [
            {
                "response": "Neural networks are computational models.",
                "annotations": {}  # No ratings provided
            }
        ]
    }
    
    result = converter(example)
    
    # Should use custom default rating
    assert result["_preference_data"]["helpfulness"] == [2.5]
    assert result["_preference_data"]["honesty"] == [2.5]
    assert result["_preference_data"]["instruction_following"] == [2.5]
    assert result["_preference_data"]["truthfulness"] == [2.5]
