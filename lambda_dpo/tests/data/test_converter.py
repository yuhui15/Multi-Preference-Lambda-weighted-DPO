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
    
    # Verify the basic structure
    assert "_prompt" in result
    assert len(result["_prompt"]) == 1
    assert result["_prompt"][0]["content"] == "Explain quantum computing in simple terms."
    
    # Verify preference data structure
    assert "_pi_target" in result
    assert "helpfulness" in result
    assert "honesty" in result
    assert "instruction_following" in result
    assert "truthfulness" in result
    
    # Verify scores match the input ratings
    assert len(result["helpfulness"]) == 3
    assert result["helpfulness"][0]["score"] == 4.0
    assert result["helpfulness"][1]["score"] == 2.0
    assert result["helpfulness"][2]["score"] == 5.0
    
    assert result["honesty"][0]["score"] == 5.0
    assert result["honesty"][1]["score"] == 3.0
    assert result["honesty"][2]["score"] == 5.0


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
    assert len(result["helpfulness"]) == 2
    assert result["helpfulness"][0]["score"] == 4.0
    assert result["helpfulness"][1]["score"] == 5.0
    assert result["honesty"][0]["score"] == 1.0  # Default for first response
    assert result["honesty"][1]["score"] == 4.0
    assert result["instruction_following"][0]["score"] == 1.0  # Default for both
    assert result["instruction_following"][1]["score"] == 1.0
    assert result["truthfulness"][0]["score"] == 1.0  # Default for both
    assert result["truthfulness"][1]["score"] == 1.0


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
    assert result["helpfulness"][0]["score"] == 1.0
    assert result["honesty"][0]["score"] == 1.0
    assert result["instruction_following"][0]["score"] == 4.0
    assert result["truthfulness"][0]["score"] == 1.0


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
    
    # Verify basic structure
    assert result["_prompt"] == [{"role": Role.USER.value, "content": "What is deep learning?"}]
    assert result["_system"] == ""
    assert result["_tools"] == ""
    assert result["_images"] is None
    assert result["_videos"] is None
    assert result["_audios"] is None
    
    # Verify empty completions create empty preference structure
    assert "_pi_target" in result
    assert "helpfulness" in result and result["helpfulness"] == []
    assert "honesty" in result and result["honesty"] == []
    assert "instruction_following" in result and result["instruction_following"] == []
    assert "truthfulness" in result and result["truthfulness"] == []


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
    assert result["helpfulness"][0]["score"] == 2.5
    assert result["honesty"][0]["score"] == 2.5
    assert result["instruction_following"][0]["score"] == 2.5
    assert result["truthfulness"][0]["score"] == 2.5


# NOTE: Online test commented out to avoid external dependencies in CI
# def test_ultrafeedback_converter_real_dataset_sample():
#     """Smoke-test the UltrafeedbackDatasetConverter on a few genuine
#     UltraFeedback examples fetched from the Hub.
# 
#     The test downloads a tiny slice (first 50 rows) to check for "N/A" values
#     that were causing warnings during training. It validates that:
#       1. The converter runs without raising.
#       2. The parsed preference vectors (_pi_target / _preference_data)
#          have the same length as the number of completions.
#       3. Identifies any "N/A" rating values that cause parsing warnings.
#     """
#     try:
#         from datasets import load_dataset  # Optional dependency
#     except ImportError:
#         pytest.skip("`datasets` library is not installed.")
# 
#     # Try fetching a larger sample to catch potential "N/A" values
#     try:
#         ds = load_dataset("openbmb/UltraFeedback", split="train[:50]")
#     except Exception as exc:  # pragma: no cover
#         pytest.skip(f"Unable to load UltraFeedback dataset: {exc}")
# 
#     dataset_attr = DatasetAttr("hf_hub", "openbmb/UltraFeedback")
#     dataset_attr.prompt = "instruction"
#     dataset_attr.completions = "completions"
#     data_args = DataArguments()
# 
#     converter = get_dataset_converter("ultrafeedback", dataset_attr, data_args)
#     dims = ["helpfulness", "honesty", "instruction_following", "truthfulness"]
#     
#     na_values_found = []
#     examples_with_na = 0
# 
#     for idx, raw in enumerate(ds):
#         # Check for "N/A" values in annotations
#         for comp_idx, completion in enumerate(raw['completions']):
#             annotations = completion.get('annotations', {})
#             for dim in dims:
#                 if dim in annotations and 'Rating' in annotations[dim]:
#                     rating = annotations[dim]['Rating']
#                     if rating == "N/A" or rating is None or rating == "":
#                         na_values_found.append({
#                             'example_idx': idx,
#                             'completion_idx': comp_idx,
#                             'dimension': dim,
#                             'rating_value': rating
#                         })
#         
#         # Debug prints commented out for cleaner test output
#         # if any("N/A" in str(completion.get('annotations', {})) for completion in raw['completions']):
#         #     examples_with_na += 1
#         #     if examples_with_na <= 3:  # Print first few examples with N/A
#         #         print(f"\n=== EXAMPLE {idx} WITH N/A VALUES ===")
#         #         print(f"Instruction: {raw['instruction'][:100]}...")
#         #         for comp_idx, completion in enumerate(raw['completions']):
#         #             annotations = completion.get('annotations', {})
#         #             print(f"Completion {comp_idx} annotations: {annotations}")
#         
#         converted = converter(raw)
# 
#         # Basic structural sanity checks
#         assert "_prompt" in converted and isinstance(converted["_prompt"], list)
#         assert len(converted["_prompt"]) == 1
# 
#         # Converter may expose either key depending on version
#         pref_key = "_preference_data" if "_preference_data" in converted else "_pi_target"
#         assert pref_key in converted
# 
#         if raw["completions"]:  # usual case
#             for d in dims:
#                 assert d in converted[pref_key]
#                 assert len(converted[pref_key][d]) == len(raw["completions"])
#         else:  # empty completions should yield empty or None prefs
#             assert not converted[pref_key] or all(len(v) == 0 for v in converted[pref_key].values())
#     
#     # Debug summary commented out for cleaner test output
#     # print(f"\n=== SUMMARY ===")
#     # print(f"Total examples checked: {len(ds)}")
#     # print(f"Examples with N/A values: {examples_with_na}")
#     # print(f"Individual N/A rating occurrences: {len(na_values_found)}")
#     # 
#     # if na_values_found:
#     #     print(f"First few N/A occurrences:")
#     #     for na_info in na_values_found[:10]:
#     #         print(f"  Example {na_info['example_idx']}, Completion {na_info['completion_idx']}, "
#     #               f"{na_info['dimension']}: {na_info['rating_value']}")
#     # else:
#     #     print("No N/A values found in this sample!")


def test_ultrafeedback_converter_all_na_dimension():
    """Test what happens when all completions have N/A for a specific dimension.
    Uses the real data structure from Example 23 we found during testing.
    """
    dataset_attr = DatasetAttr("hf_hub", "openbmb/UltraFeedback")
    dataset_attr.prompt = "instruction" 
    dataset_attr.completions = "completions"
    data_args = DataArguments()
    
    # Real example structure from UltraFeedback Example 23 (creative writing task)
    example = {
        "instruction": "Describe what summer means to you in one sentence.",
        "completions": [
            {
                "response": "As an AI, I don't have personal feelings, but summer can be described as a season of warmth, growth, and vibrant life.",
                "annotations": {
                    "helpfulness": {"Rating": "3"},
                    "honesty": {"Rating": "N/A", "Rationale": "This is a creative writing task, and the honesty and uncertainty expression assessment is not applicable."},
                    "instruction_following": {"Rating": "3"},
                    "truthfulness": {"Rating": "3"}
                }
            },
            {
                "response": "Summer is a symphony of golden sunshine, lazy afternoons, and the sweet scent of blooming flowers that fills the air with warmth and joy.",
                "annotations": {
                    "helpfulness": {"Rating": "5"},
                    "honesty": {"Rating": "N/A", "Rationale": "This is a creative writing task, and the honesty and uncertainty expression assessment is not applicable."},
                    "instruction_following": {"Rating": "4"},
                    "truthfulness": {"Rating": "5"}
                }
            },
            {
                "response": "Summer means freedom, adventure, and endless possibilities under the bright blue sky.",
                "annotations": {
                    "helpfulness": {"Rating": "4"},
                    "honesty": {"Rating": "N/A", "Rationale": "This is a creative writing task, and the honesty and uncertainty expression assessment is not applicable."},
                    "instruction_following": {"Rating": "5"},
                    "truthfulness": {"Rating": "5"}
                }
            },
            {
                "response": "As an AI, I don't experience seasons personally, but summer could be characterized as a time of warmth and outdoor activities.",
                "annotations": {
                    "helpfulness": {"Rating": "3"},
                    "honesty": {"Rating": "N/A", "Rationale": "This is a creative writing task, and the honesty and uncertainty expression assessment is not applicable."},
                    "instruction_following": {"Rating": "3"},
                    "truthfulness": {"Rating": "3"}
                }
            }
        ]
    }
    
    converter = get_dataset_converter("ultrafeedback", dataset_attr, data_args)
    result = converter(example)
    
    # Check that all N/A values become default rating (1.0)
    honesty_scores = [comp["score"] for comp in result["honesty"]]
    # Debug: print(f"All N/A honesty scores: {honesty_scores}")
    assert len(honesty_scores) == 4  # UltraFeedback always has 4 completions
    assert all(score == 1.0 for score in honesty_scores)
    
    # Check that preferences are uniform when all scores are equal
    honesty_prefs = result["_pi_target"]["honesty"]
    # Debug: print(f"Uniform preferences (4 completions): {honesty_prefs}")
    # Should be approximately [0.25, 0.25, 0.25, 0.25] for 4 equal scores
    for pref in honesty_prefs:
        assert abs(pref - 0.25) < 0.001
    
    # Verify helpfulness still has real preferences (scores differ: 3,5,4,3)
    help_scores = [comp["score"] for comp in result["helpfulness"]]
    # Debug: print(f"Varied helpfulness scores: {help_scores}")
    assert help_scores == [3.0, 5.0, 4.0, 3.0]
    help_prefs = result["_pi_target"]["helpfulness"]
    # Debug: print(f"Varied preferences: {help_prefs}")
    # Should not be uniform since scores differ, and completion 1 (score=5) should have highest preference
    assert help_prefs[1] > help_prefs[0]  # Score 5 > Score 3
    assert help_prefs[1] > help_prefs[2]  # Score 5 > Score 4
    assert help_prefs[1] > help_prefs[3]  # Score 5 > Score 3


# NOTE: Online test commented out to avoid external dependencies in CI
# def test_ultrafeedback_converter_real_na_example():
#     """Test the N/A fix with the actual Example 23 from UltraFeedback dataset."""
#     try:
#         from datasets import load_dataset
#     except ImportError:
#         pytest.skip("`datasets` library is not installed.")
# 
#     try:
#         # Load just the specific example that had N/A values (Example 23)
#         ds = load_dataset("openbmb/UltraFeedback", split="train[23:24]")
#     except Exception as exc:
#         pytest.skip(f"Unable to load UltraFeedback dataset: {exc}")
# 
#     dataset_attr = DatasetAttr("hf_hub", "openbmb/UltraFeedback")
#     dataset_attr.prompt = "instruction"
#     dataset_attr.completions = "completions"
#     data_args = DataArguments()
# 
#     converter = get_dataset_converter("ultrafeedback", dataset_attr, data_args)
#     
#     # Get the example that should have N/A values
#     raw_example = ds[0]
#     # Debug: print(f"Testing real example: {raw_example['instruction'][:50]}...")
#     
#     # Check if this example indeed has N/A values
#     has_na = False
#     for completion in raw_example['completions']:
#         annotations = completion.get('annotations', {})
#         for dim in ["helpfulness", "honesty", "instruction_following", "truthfulness"]:
#             if dim in annotations and annotations[dim].get('Rating') == 'N/A':
#                 has_na = True
#                 # Debug: print(f"Found N/A in {dim} dimension")
#                 break
#     
#     if not has_na:
#         pytest.skip("Example 23 doesn't have N/A values (dataset may have changed)")
#     
#     # Convert and verify no warnings are generated
#     result = converter(raw_example)
#     
#     # Verify structure
#     assert "_prompt" in result
#     assert "_pi_target" in result
#     assert len(result["helpfulness"]) == 4  # Should have 4 completions
#     
#     # Check that any dimension with all N/A values has uniform preferences
#     for dimension in ["helpfulness", "honesty", "instruction_following", "truthfulness"]:
#         scores = [comp["score"] for comp in result[dimension]]
#         prefs = result["_pi_target"][dimension]
#         # Debug: print(f"{dimension} scores: {scores}")
#         # Debug: print(f"{dimension} preferences: {prefs}")
#         
#         # If all scores are 1.0 (from N/A), preferences should be uniform
#         if all(score == 1.0 for score in scores):
#             # Debug: print(f"  -> {dimension} had all N/A values, checking uniform preferences")
#             for pref in prefs:
#                 assert abs(pref - 0.25) < 0.001  # Should be ~0.25 for 4 equal completions
