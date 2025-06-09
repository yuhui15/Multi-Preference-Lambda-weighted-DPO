import os
import json
import torch
import pytest
from unittest.mock import Mock
from transformers import AutoTokenizer

from llamafactory.data.processor.listwise import ListwiseDatasetProcessor
from llamafactory.data.converter import UltrafeedbackDatasetConverter
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")


def test_ultrafeedback_dataset_conversion_and_preprocessing():
    data_args = DataArguments(cutoff_len=512, template="llama3")
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    processor = ListwiseDatasetProcessor(
        data_args=data_args,
        template=template,
        tokenizer=tokenizer,
        processor=Mock()
    )

    # Minimal raw UltraFeedback input
    example = {
        "source": "manual",
        "instruction": "What is the capital of France?",
        "models": ["a", "b", "c", "d"],
        "completions": [
            {"response": "Paris", "annotations": {"helpfulness": {"Rating": "4"}}},
            {"response": "Berlin", "annotations": {"helpfulness": {"Rating": "1"}}},
            {"response": "London", "annotations": {"helpfulness": {"Rating": "2"}}},
            {"response": "Madrid", "annotations": {"helpfulness": {"Rating": "3"}}}
        ]
    }

    # Convert dataset
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
    converter = UltrafeedbackDatasetConverter(dataset_attr=dataset_attr, data_args=data_args)
    converted = converter(example)

    # Run preprocess
    model_inputs = processor.preprocess_dataset({
        "_prompt": [converted["_prompt"]],
        "helpfulness": [converted["helpfulness"]],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
        "_pi_target": [converted["_pi_target"]],
    })

    # Check structure
    assert "input_ids" in model_inputs
    assert len(model_inputs["input_ids"]) == 1
    assert len(model_inputs["input_ids"][0]) == 4  # 4 completions
    assert model_inputs["num_responses"][0] == 4

    # Check preference order: helpfulness = [4, 1, 2, 3]
    # Expected normalized: [0.4, 0.1, 0.2, 0.3]
    helpfulness = model_inputs["preference_distributions"][0]["helpfulness"]
    assert len(helpfulness) == 4
    assert abs(sum(helpfulness) - 1.0) < 1e-6
    assert helpfulness[0] > helpfulness[3] > helpfulness[2] > helpfulness[1]

    # Check token alignment
    for i in range(4):
        ids = model_inputs["input_ids"][0][i]
        labels = model_inputs["labels"][0][i]
        attention = model_inputs["attention_masks"][0][i]
        assert isinstance(ids, list) and isinstance(labels, list)
        assert len(ids) == len(labels) == len(attention)
        assert all([m == 1 for m in attention])
        assert any([l == IGNORE_INDEX for l in labels])
