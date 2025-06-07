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
import json
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoTokenizer

from llamafactory.data.converter import UltrafeedbackDatasetConverter
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


def test_preprocess_dataset_basic(setup_processor):
    """Test basic listwise dataset preprocessing."""
    listwise_processor, tokenizer = setup_processor
    
    # Create sample data with multiple responses
    examples = {
        "_prompt": [
            [{"role": "user", "content": "What is 2+2?"}]
        ],
        "helpfulness": [
            [
                {"model": "a", "content": "2+2 equals 4.", "score": 3.0},
                {"model": "b", "content": "The answer is 4.", "score": 4.0},
                {"model": "c", "content": "Four.", "score": 2.0}
            ]
        ],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_pi_target": [
            {
                "helpfulness": [0.25, 0.5, 0.25],
                "honesty": [0.33, 0.33, 0.34],
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
    assert abs(sum(prefs["helpfulness"]) - 1.0) < 1e-6
    assert abs(sum(prefs["honesty"]) - 1.0) < 1e-6


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
        "helpfulness": [[{"model": "a", "content": "The answer is 4.", "score": 1.0}]],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_pi_target": [None]
    }
    
    model_inputs = listwise_processor.preprocess_dataset(invalid_examples)
    # Should skip invalid examples
    assert len(model_inputs["input_ids"]) == 0
    
    # Test with insufficient responses
    insufficient_responses = {
        "_prompt": [[{"role": "user", "content": "What is 2+2?"}]],
        "helpfulness": [[{"model": "a", "content": "4", "score": 1.0}]],  # Only 1 response
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_pi_target": [None]
    }
    
    model_inputs = listwise_processor.preprocess_dataset(insufficient_responses)
    # Should skip examples with < 2 responses
    assert len(model_inputs["input_ids"]) == 0


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


def test_preprocess_ultrafeedback_example(setup_processor):
    """Test preprocessing of a real UltraFeedback example."""
    listwise_processor, tokenizer = setup_processor
    
    # Raw UltraFeedback example
    ultrafeedback_example = {
        "source": "evol_instruct",
        "instruction": "Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea? Here's some starter code to help you out:\n#include <iostream>\n#include <string>\nusing namespace std;\nint main() {\n    string country;\n    // prompt user for input\n    cout << \"Enter the name of a country: \";\n    cin >> country;\n    // check if country borders the Mediterranean Sea\n    // [C++ code]\n    return 0;\n}",
        "models": ["alpaca-7b", "pythia-12b", "starchat", "vicuna-33b"],
        "completions": [
            {
                "annotations": {"helpfulness": {"Rating": "2"}, "honesty": {"Rating": "1"}, "instruction_following": {"Rating": "1"}, "truthfulness": {"Rating": "1"}},
                "model": "alpaca-7b",
                "response": "int main() {\n    string country;\n    // prompt user for input\n    cout << \"Enter the name of a country: \";\n    cin >> country;\n    // check if country borders the Mediterranean Sea\n    if (endsWith(country, \"Mediterranean\")) {\n        cout << \"Yes, the country \" << country\n             << \" borders the Mediterranean Sea.\";\n    } else {\n        cout << \"No, the country \" << country\n             << \" does not border the Mediterranean Sea.\";\n    }\n    return 0;\n}"
            },
            {
                "annotations": {"helpfulness": {"Rating": "5"}, "honesty": {"Rating": "5"}, "instruction_following": {"Rating": "5"}, "truthfulness": {"Rating": "5"}},
                "model": "starchat",
                "response": "Here's a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea:\n\n#include <iostream>\n#include <string>\n#include <set>\n#include <map>\n#include <algorithm>\n\nusing namespace std;\n\nint main() {\n    // store countries and their bordering seas in a map\n    map<string, set<string>> countries;\n    countries[\"Algeria\"] = {\"Mediterranean Sea\", \"North African Coast\"};\n    countries[\"France\"] = {\"Mediterranean Sea\", \"English Channel\"};\n    countries[\"Italy\"] = {\"Mediterranean Sea\", \"Adriatic Sea\"};\n    countries[\"Malta\"] = {\"Mediterranean Sea\", \"North African Coast\"};\n    countries[\"Spain\"] = {\"Mediterranean Sea\", \"Atlantic Ocean\"};\n    countries[\"Tunisia\"] = {\"Mediterranean Sea\", \"North African Coast\"};\n\n    string country;\n    cout << \"Enter the name of a country: \";\n    cin >> country;\n\n    // check if country is in the map\n    if (countries.find(country)!= countries.end()) {\n        // check if country borders the Mediterranean Sea\n        set<string> borders = countries.at(country);\n        if (borders.find(\"Mediterranean Sea\")!= borders.end()) {\n            cout << country << \" borders the Mediterranean Sea.\" << endl;\n        } else {\n            cout << country << \" does not border the Mediterranean Sea.\" << endl;\n        }\n    } else {\n        cout << country << \" is not a recognized country.\" << endl;\n    }\n\n    return 0;\n}\n\nIn this program, we first define a map called countries that stores each country's name as a key and a set of its surrounding seas as the value. We use a set to store each sea name so that we can easily check if a country borders a certain sea.\n\nWe then present the user with a prompt to enter the name of a country. We use a while loop to ensure that the user enters a valid country name. We check if the country is in the map, and if so, we retrieve its bordering seas from the map.\n\nFinally, we check if the Mediterranean Sea is among the set of borders. If it is, we inform the user that the country borders the Mediterranean Sea. If not, we inform the user that the country does not border the Mediterranean Sea."
            },
            {
                "annotations": {"helpfulness": {"Rating": "4"}, "honesty": {"Rating": "4"}, "instruction_following": {"Rating": "5"}, "truthfulness": {"Rating": "3"}},
                "model": "vicuna-33b",
                "response": "Sure! Here's a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea using a simple `if` statement and a `vector` of countries that border the Mediterranean Sea:\n```cpp\n#include <iostream>\n#include <string>\n#include <vector>\nusing namespace std;\n\nvector<string> mediterraneanCountries = {\"Algeria\", \"Egypt\", \"France\", \"Greece\", \"Israel\", \"Italy\", \"Lebanon\", \"Libya\", \"Morocco\", \"Palestine\", \"Portugal\", \"Spain\", \"Syria\", \"Turkey\", \"Tunisia\"};\n\nint main() {\n    string country;\n\n    cout << \"Enter the name of a country: \";\n    cin >> country;\n\n    if (mediterraneanCountries.find(country) != mediterraneanCountries.end()) {\n        cout << \"The country \" << country << \" borders the Mediterranean Sea.\" << endl;\n    } else {\n        cout << \"The country \" << country << \" does not border the Mediterranean Sea.\" << endl;\n    }\n\n    return 0;\n}\n```\nThis program defines a `vector` of countries that border the Mediterranean Sea, and then uses the `find` method to check if the entered country is in the `vector`. If it is, the program outputs a message indicating that the country borders the Mediterranean Sea; otherwise, it outputs a message indicating that the country does not border the Mediterranean Sea."
            },
            {
                "annotations": {"helpfulness": {"Rating": "1"}, "honesty": {"Rating": "1"}, "instruction_following": {"Rating": "2"}, "truthfulness": {"Rating": "1"}},
                "model": "pythia-12b",
                "response": "Sure, here is the program using the C++11 algorithm \"cds::algorithm::GreaterEqual\":\n#include <iostream>\n#include <string>\n#include <algorithm>\n#include <vector>\n#include <cctype>\n\nusing namespace std;\n\nint main() {\n    string country;\n    cout << \"Enter the name of a country: \";\n    cin >> country;\n    std::vector<string> vec;\n    vec.push_back(country);\n    size_t index = std::find_if(vec.begin(), vec.end(), [](const string& s) {\n        return std::any_of(s.begin(), s.end(), [](const char& c) {\n            return c == '}}';\n    });\n    if (index != vec.end()) {\n        if ((*index)[0] == 'M') {\n            cout << \"Country is bordered by the Mediterranean Sea.\" << endl;\n        } else {\n            cout << \"Country does not border the Mediterranean Sea.\" << endl;\n        }\n    } else {\n        cout << \"Country is not found.\" << endl;\n    }\n    return 0;\n}"
            }
        ],
        "correct_answers": ["None"],
        "incorrect_answers": ["None"]
    }
    
    print("--- Raw UltraFeedback Example ---")
    print(ultrafeedback_example)
    
    # Convert the raw example
    dataset_attr = Mock(
        prompt="instruction",
        completions="completions",
        system="",
        tools="",
        images=None,
        videos=None,
        audios=None,
        load_from="dict"
    )
    converter = UltrafeedbackDatasetConverter(dataset_attr=dataset_attr, data_args=Mock())
    converted_example = converter(ultrafeedback_example)
    
    print("\n--- Converted Example ---")
    converted_output = json.dumps(converted_example, indent=4)
    print(converted_output)
    
    # Process the converted example
    model_inputs = listwise_processor.preprocess_dataset({
        "_prompt": [converted_example["_prompt"]],
        "helpfulness": [converted_example["helpfulness"]],
        "honesty": [converted_example["honesty"]],
        "instruction_following": [converted_example["instruction_following"]],
        "truthfulness": [converted_example["truthfulness"]],
        "_system": [converted_example["_system"]],
        "_tools": [converted_example["_tools"]],
        "_images": [converted_example["_images"]],
        "_videos": [converted_example["_videos"]],
        "_audios": [converted_example["_audios"]],
        "_pi_target": [converted_example["_pi_target"]]
    })
    
    print("\n--- Processed Model Inputs ---")
    # Convert tensors to lists for JSON serialization
    serializable_model_inputs = {
        k: [[t.tolist() if isinstance(t, torch.Tensor) else t for t in inner] for inner in v]
        if isinstance(v[0], list) and isinstance(v[0][0], list) else v
        for k, v in model_inputs.items()
    }
    processed_output = json.dumps(serializable_model_inputs, indent=4)
    print(processed_output)

    with open("test_output.json", "w") as f:
        json.dump({
            "converted_example": converted_example,
            "processed_model_inputs": serializable_model_inputs
        }, f, indent=4)
    
    # Verify the output
    assert len(model_inputs["input_ids"]) == 1
    assert model_inputs["num_responses"][0] == 4
    
    prefs = model_inputs["preference_distributions"][0]
    assert "helpfulness" in prefs
    assert "honesty" in prefs
    assert "instruction_following" in prefs
    assert "truthfulness" in prefs
    
    # Check helpfulness distribution (scores: 2, 5, 4, 1)
    helpfulness_prefs = prefs["helpfulness"]
    assert helpfulness_prefs[1] > helpfulness_prefs[2] > helpfulness_prefs[0] > helpfulness_prefs[3]
    
    # Check honesty distribution (scores: 1, 5, 4, 1)
    honesty_prefs = prefs["honesty"]
    assert honesty_prefs[1] > honesty_prefs[2] > honesty_prefs[0]
    assert abs(honesty_prefs[0] - honesty_prefs[3]) < 1e-6
    
    # Check instruction_following distribution (scores: 1, 5, 5, 2)
    instruction_following_prefs = prefs["instruction_following"]
    assert instruction_following_prefs[1] > instruction_following_prefs[3] > instruction_following_prefs[0]
    assert abs(instruction_following_prefs[1] - instruction_following_prefs[2]) < 1e-6
    
    # Check truthfulness distribution (scores: 1, 5, 3, 1)
    truthfulness_prefs = prefs["truthfulness"]
    assert truthfulness_prefs[1] > truthfulness_prefs[2] > truthfulness_prefs[0]
    assert abs(truthfulness_prefs[0] - truthfulness_prefs[3]) < 1e-6
