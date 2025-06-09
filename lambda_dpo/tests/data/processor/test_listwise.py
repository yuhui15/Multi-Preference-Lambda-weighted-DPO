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


def test_ultrafeedback_listwise_flattened_structure():
    data_args = DataArguments(cutoff_len=512, template="llama3")
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    processor = ListwiseDatasetProcessor(
        data_args=data_args,
        template=template,
        tokenizer=tokenizer,
        processor=Mock()
    )

    # UltraFeedback example with 4 responses
    example = {
        "source": "manual",
        "instruction": "What is the capital of France?",
        "models": ["a", "b", "c", "d"],
        "completions": [
            {"response": "Paris", "annotations": {
                "helpfulness": {"Rating": "4"},
                "honesty": {"Rating": "4"},
                "instruction_following": {"Rating": "3"},
                "truthfulness": {"Rating": "4"},
            }},
            {"response": "Berlin", "annotations": {
                "helpfulness": {"Rating": "1"},
                "honesty": {"Rating": "1"},
                "instruction_following": {"Rating": "2"},
                "truthfulness": {"Rating": "2"},
            }},
            {"response": "London", "annotations": {
                "helpfulness": {"Rating": "2"},
                "honesty": {"Rating": "3"},
                "instruction_following": {"Rating": "3"},
                "truthfulness": {"Rating": "1"},
            }},
            {"response": "Madrid", "annotations": {
                "helpfulness": {"Rating": "3"},
                "honesty": {"Rating": "2"},
                "instruction_following": {"Rating": "2"},
                "truthfulness": {"Rating": "3"},
            }},
        ]
    }

    # Convert
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

    # Preprocess
    model_inputs = processor.preprocess_dataset({
        "_prompt": [converted["_prompt"]],
        "helpfulness": [converted["helpfulness"]],
        "honesty": [converted["honesty"]],
        "instruction_following": [converted["instruction_following"]],
        "truthfulness": [converted["truthfulness"]],
        "_system": [converted["_system"]],
        "_tools": [converted["_tools"]],
        "_images": [converted["_images"]],
        "_videos": [converted["_videos"]],
        "_audios": [converted["_audios"]],
        "_pi_target": [converted["_pi_target"]],
    })

    # 🔍 Check output structure
    assert "input_ids" in model_inputs
    assert "labels" in model_inputs
    assert "attention_mask" in model_inputs
    assert "pi_target" in model_inputs

    assert len(model_inputs["input_ids"]) == 16
    assert len(model_inputs["labels"]) == 16
    assert len(model_inputs["attention_mask"]) == 16
    assert len(model_inputs["pi_target"]) == 16

    # 🔍 Check token alignment and masks
    for i in range(16):
        ids = model_inputs["input_ids"][i]
        labels = model_inputs["labels"][i]
        attention = model_inputs["attention_mask"][i]

        assert isinstance(ids, list) and isinstance(labels, list)
        assert len(ids) == len(labels) == len(attention)
        assert all(m == 1 for m in attention)
        assert any(l == IGNORE_INDEX for l in labels)

    # 🔍 Check normalized pi_target for each group of 4
    for start in range(0, 16, 4):
        pi_slice = model_inputs["pi_target"][start:start + 4]
        assert abs(sum(pi_slice) - 1.0) < 1e-5
