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
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset, IterableDataset

from llamafactory.data.loader import _peek_first_example, _get_dataset_processor
from llamafactory.data.processor import ListwiseDatasetProcessor, PairwiseDatasetProcessor
from llamafactory.hparams import DataArguments
from llamafactory.train.test_utils import load_dataset_module


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_DATA = os.getenv("TINY_DATA", "llamafactory/tiny-supervised-dataset")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "template": "llama3",
    "dataset": TINY_DATA,
    "dataset_dir": "ONLINE",
    "cutoff_len": 8192,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


def test_load_train_only():
    dataset_module = load_dataset_module(**TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is None


def test_load_val_size():
    dataset_module = load_dataset_module(val_size=0.1, **TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is not None


def test_load_eval_data():
    dataset_module = load_dataset_module(eval_dataset=TINY_DATA, **TRAIN_ARGS)
    assert dataset_module.get("train_dataset") is not None
    assert dataset_module.get("eval_dataset") is not None


def test_peek_first_example():
    """Test _peek_first_example function with different dataset types."""
    # Test with regular Dataset
    data = [
        {"prompt": "What is AI?", "response": "AI is artificial intelligence."},
        {"prompt": "What is ML?", "response": "ML is machine learning."}
    ]
    dataset = Dataset.from_list(data)
    
    first_example = _peek_first_example(dataset)
    assert first_example == data[0]
    
    # Test with empty Dataset
    empty_dataset = Dataset.from_list([])
    first_example = _peek_first_example(empty_dataset)
    assert first_example is None
    
    # Test with None dataset
    first_example = _peek_first_example(None)
    assert first_example is None


def test_peek_first_example_iterable():
    """Test _peek_first_example with IterableDataset."""
    def data_generator():
        yield {"prompt": "What is AI?", "response": "AI is artificial intelligence."}
        yield {"prompt": "What is ML?", "response": "ML is machine learning."}
    
    iterable_dataset = IterableDataset.from_generator(data_generator)
    
    first_example = _peek_first_example(iterable_dataset)
    assert first_example == {"prompt": "What is AI?", "response": "AI is artificial intelligence."}


def test_get_dataset_processor_pairwise():
    """Test _get_dataset_processor returns PairwiseDatasetProcessor for pairwise data."""
    data_args = DataArguments()
    template = Mock()
    tokenizer = Mock()
    processor = Mock()
    
    # Mock dataset with pairwise structure
    pairwise_example = {
        "_prompt": [{"role": "user", "content": "What is AI?"}],
        "_response": [
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "assistant", "content": "AI stands for artificial intelligence."}
        ]
    }
    
    dataset_processor = _get_dataset_processor(
        data_args=data_args,
        stage="rm",
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        peeked_example=pairwise_example
    )
    
    assert isinstance(dataset_processor, PairwiseDatasetProcessor)


def test_get_dataset_processor_listwise_with_preference_data():
    """Test _get_dataset_processor returns ListwiseDatasetProcessor for data with preference_data."""
    data_args = DataArguments()
    template = Mock()
    tokenizer = Mock()
    processor = Mock()
    
    # Mock dataset with preference data
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
    
    dataset_processor = _get_dataset_processor(
        data_args=data_args,
        stage="rm",
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        peeked_example=listwise_example
    )
    
    assert isinstance(dataset_processor, ListwiseDatasetProcessor)


def test_get_dataset_processor_listwise_with_multiple_responses():
    """Test _get_dataset_processor returns ListwiseDatasetProcessor for data with >2 responses."""
    data_args = DataArguments()
    template = Mock()
    tokenizer = Mock()
    processor = Mock()
    
    # Mock dataset with multiple responses (>2)
    listwise_example = {
        "_prompt": [{"role": "user", "content": "What is AI?"}],
        "_response": [
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "assistant", "content": "AI stands for artificial intelligence."},
            {"role": "assistant", "content": "AI refers to artificial intelligence."}
        ]
    }
    
    dataset_processor = _get_dataset_processor(
        data_args=data_args,
        stage="rm",
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        peeked_example=listwise_example
    )
    
    assert isinstance(dataset_processor, ListwiseDatasetProcessor)


def test_get_dataset_processor_invalid_responses():
    """Test _get_dataset_processor handles invalid response structures."""
    data_args = DataArguments()
    template = Mock()
    tokenizer = Mock()
    processor = Mock()
    
    # Mock dataset with invalid response structure (not dicts)
    invalid_example = {
        "_prompt": [{"role": "user", "content": "What is AI?"}],
        "_response": ["response1", "response2", "response3"]  # Strings instead of dicts
    }
    
    dataset_processor = _get_dataset_processor(
        data_args=data_args,
        stage="rm",
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        peeked_example=invalid_example
    )
    
    # Should fall back to PairwiseDatasetProcessor
    assert isinstance(dataset_processor, PairwiseDatasetProcessor)


def test_get_dataset_processor_no_response_key():
    """Test _get_dataset_processor handles missing _response key."""
    data_args = DataArguments()
    template = Mock()
    tokenizer = Mock()
    processor = Mock()
    
    # Mock dataset without _response key
    no_response_example = {
        "_prompt": [{"role": "user", "content": "What is AI?"}]
    }
    
    dataset_processor = _get_dataset_processor(
        data_args=data_args,
        stage="rm",
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        peeked_example=no_response_example
    )
    
    # Should fall back to PairwiseDatasetProcessor
    assert isinstance(dataset_processor, PairwiseDatasetProcessor)


def test_get_dataset_processor_empty_example():
    """Test _get_dataset_processor handles empty or None examples."""
    data_args = DataArguments()
    template = Mock()
    tokenizer = Mock()
    processor = Mock()
    
    # Test with None example
    dataset_processor = _get_dataset_processor(
        data_args=data_args,
        stage="rm",
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        peeked_example=None
    )
    
    # Should fall back to PairwiseDatasetProcessor
    assert isinstance(dataset_processor, PairwiseDatasetProcessor)
    
    # Test with empty example
    dataset_processor = _get_dataset_processor(
        data_args=data_args,
        stage="rm",
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        peeked_example={}
    )
    
    # Should fall back to PairwiseDatasetProcessor
    assert isinstance(dataset_processor, PairwiseDatasetProcessor)


def test_get_dataset_processor_non_rm_stage():
    """Test _get_dataset_processor for non-rm stages."""
    data_args = DataArguments()
    template = Mock()
    tokenizer = Mock()
    processor = Mock()
    
    from llamafactory.data.processor import SupervisedDatasetProcessor
    
    # For non-rm stages, should not use listwise/pairwise processors
    dataset_processor = _get_dataset_processor(
        data_args=data_args,
        stage="sft",
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        peeked_example={"_prompt": [{"role": "user", "content": "test"}]}
    )
    
    assert isinstance(dataset_processor, SupervisedDatasetProcessor)
