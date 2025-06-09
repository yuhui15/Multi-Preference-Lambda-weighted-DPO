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

import torch
from PIL import Image

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.collator import ListwiseDataCollatorWithPadding, MultiModalDataCollatorForSeq2Seq, prepare_4d_attention_mask
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")


def test_base_collator():
    model_args, data_args, *_ = get_infer_args({"model_name_or_path": TINY_LLAMA3, "template": "default"})
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    p = tokenizer_module["tokenizer"].pad_token_id
    q = IGNORE_INDEX
    features = [
        {
            "input_ids": [0, 1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "labels": [q, q, 2, 3, 4, 5],
        },
        {
            "input_ids": [6, 7],
            "attention_mask": [1, 1],
            "labels": [q, 7],
        },
    ]
    batch_input = data_collator(features)
    expected_input = {
        "input_ids": [
            [0, 1, 2, 3, 4, 5, p, p],
            [6, 7, p, p, p, p, p, p],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        "labels": [
            [q, q, 2, 3, 4, 5, q, q],
            [q, 7, q, q, q, q, q, q],
        ],
    }
    for k in batch_input.keys():
        assert batch_input[k].eq(torch.tensor(expected_input[k])).all()


def test_multimodal_collator():
    model_args, data_args, *_ = get_infer_args(
        {"model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct", "template": "qwen2_vl"}
    )
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        pad_to_multiple_of=4,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    p = tokenizer_module["tokenizer"].pad_token_id
    q = IGNORE_INDEX
    s = tokenizer_module["tokenizer"].convert_tokens_to_ids("<|vision_start|>")
    e = tokenizer_module["tokenizer"].convert_tokens_to_ids("<|vision_end|>")
    m = tokenizer_module["tokenizer"].convert_tokens_to_ids("<|image_pad|>")
    fake_image = Image.new("RGB", (64, 64), (255, 255, 255))

    features = [
        {
            "input_ids": [0, 1, 2, 3],
            "attention_mask": [1, 1, 1, 1],
            "labels": [0, 1, 2, 3],
        },
    ]
    batch_input = data_collator(features)
    expected_input = {
        "input_ids": [
            [0, 1, 2, 3, s, m, m, m, m, e, p, p],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "labels": [
            [0, 1, 2, 3, q, q, q, q, q, q, q, q],
        ],
        **tokenizer_module["processor"].image_processor(fake_image),
    }
    for k in batch_input.keys():
        assert batch_input[k].eq(torch.tensor(expected_input[k])).all()


def test_4d_attention_mask():
    o = 0.0
    x = torch.finfo(torch.float16).min
    attention_mask_with_indices = torch.tensor(
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
    )
    attention_mask_computed = prepare_4d_attention_mask(attention_mask_with_indices, torch.float16)
    attention_mask_expected = torch.tensor(
        [
            [
                [
                    [o, x, x, x, x, x],
                    [o, o, x, x, x, x],
                    [x, x, o, x, x, x],
                    [x, x, o, o, x, x],
                    [x, x, o, o, o, x],
                    [x, x, x, x, x, x],
                ]
            ],
            [
                [
                    [o, x, x, x, x, x],
                    [x, o, x, x, x, x],
                    [x, o, o, x, x, x],
                    [x, x, x, o, x, x],
                    [x, x, x, o, o, x],
                    [x, x, x, o, o, o],
                ]
            ],
        ],
        dtype=torch.float16,
    )
    assert list(attention_mask_computed.size()) == [2, 1, 6, 6]
    assert torch.all(attention_mask_computed == attention_mask_expected)


def test_listwise_collator_basic():
    """Test basic functionality of ListwiseDataCollatorWithPadding."""
    model_args, data_args, *_ = get_infer_args({"model_name_or_path": TINY_LLAMA3, "template": "default"})
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    data_collator = ListwiseDataCollatorWithPadding(
        template=template,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    
    p = tokenizer_module["tokenizer"].pad_token_id
    q = IGNORE_INDEX
    
    # Create 4 examples representing responses to the same prompt (like the restructured dataset)
    features = [
        {
            "input_ids": [0, 1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "labels": [q, q, q, q, 4, 5],
            "pi_target": 0.6,
            "images": [],
            "videos": [],
            "audios": [],
        },
        {
            "input_ids": [0, 1, 2, 3, 6],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [q, q, q, q, 6],
            "pi_target": 0.1,
            "images": [],
            "videos": [],
            "audios": [],
        },
        {
            "input_ids": [0, 1, 2, 3, 7],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [q, q, q, q, 7],
            "pi_target": 0.2,
            "images": [],
            "videos": [],
            "audios": [],
        },
        {
            "input_ids": [0, 1, 2, 3, 8, 9],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "labels": [q, q, q, q, 8, 9],
            "pi_target": 0.1,
            "images": [],
            "videos": [],
            "audios": [],
        },
    ]
    
    batch_input = data_collator(features)
    
    # Check basic structure
    assert "input_ids" in batch_input
    assert "attention_mask" in batch_input
    assert "labels" in batch_input
    assert "pi_target" in batch_input
    
    # Check shapes
    assert batch_input["input_ids"].shape[0] == 4  # 4 examples
    assert batch_input["attention_mask"].shape[0] == 4
    assert batch_input["labels"].shape[0] == 4
    assert batch_input["pi_target"].shape[0] == 4
    
    # Check pi_target values are preserved
    expected_pi_targets = torch.tensor([0.6, 0.1, 0.2, 0.1], dtype=torch.float32)
    assert torch.allclose(batch_input["pi_target"], expected_pi_targets)
    
    # Check padding (sequences should be padded to multiple of 8)
    seq_len = batch_input["input_ids"].shape[1]
    assert seq_len % 8 == 0
    assert seq_len == 8  # Expected padded length
    
    # Check padding tokens
    expected_input = torch.tensor([
        [0, 1, 2, 3, 4, 5, p, p],
        [0, 1, 2, 3, 6, p, p, p],
        [0, 1, 2, 3, 7, p, p, p],
        [0, 1, 2, 3, 8, 9, p, p],
    ])
    assert torch.equal(batch_input["input_ids"], expected_input)
    
    expected_attention = torch.tensor([
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ])
    assert torch.equal(batch_input["attention_mask"], expected_attention)
    
    expected_labels = torch.tensor([
        [q, q, q, q, 4, 5, q, q],
        [q, q, q, q, 6, q, q, q],
        [q, q, q, q, 7, q, q, q],
        [q, q, q, q, 8, 9, q, q],
    ])
    assert torch.equal(batch_input["labels"], expected_labels)


def test_listwise_collator_validation():
    """Test that ListwiseDataCollatorWithPadding validates group size."""
    model_args, data_args, *_ = get_infer_args({"model_name_or_path": TINY_LLAMA3, "template": "default"})
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    data_collator = ListwiseDataCollatorWithPadding(
        template=template,
        **tokenizer_module,
    )
    
    # Test with invalid number of examples (not multiple of 4)
    features = [
        {
            "input_ids": [0, 1, 2],
            "attention_mask": [1, 1, 1],
            "labels": [-100, -100, 2],
            "pi_target": 0.5,
        },
        {
            "input_ids": [0, 1, 3],
            "attention_mask": [1, 1, 1],
            "labels": [-100, -100, 3],
            "pi_target": 0.5,
        },
        # Only 2 examples instead of 4
    ]
    
    try:
        data_collator(features)
        assert False, "Should have raised ValueError for non-multiple of 4"
    except ValueError as e:
        assert "groups of 4 examples" in str(e)


def test_listwise_collator_with_real_data():
    """Test ListwiseDataCollatorWithPadding with data format from restructured dataset."""
    model_args, data_args, *_ = get_infer_args({"model_name_or_path": TINY_LLAMA3, "template": "llama3"})
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    data_collator = ListwiseDataCollatorWithPadding(
        template=template,
        pad_to_multiple_of=4,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    
    # Simulated data similar to the restructured dataset file
    features = [
        {
            "input_ids": [128000, 128006, 882, 128007, 271, 3923, 374, 279, 6864, 315, 9822, 30, 128009, 128006, 78191, 128007, 271, 60704, 128009],
            "attention_mask": [1] * 19,
            "labels": [-100] * 17 + [60704, 128009],
            "pi_target": 0.6439142598879724,
        },
        {
            "input_ids": [128000, 128006, 882, 128007, 271, 3923, 374, 279, 6864, 315, 9822, 30, 128009, 128006, 78191, 128007, 271, 95509, 128009],
            "attention_mask": [1] * 19,
            "labels": [-100] * 17 + [95509, 128009],
            "pi_target": 0.032058603280084995,
        },
        {
            "input_ids": [128000, 128006, 882, 128007, 271, 3923, 374, 279, 6864, 315, 9822, 30, 128009, 128006, 78191, 128007, 271, 40672, 128009],
            "attention_mask": [1] * 19,
            "labels": [-100] * 17 + [40672, 128009],
            "pi_target": 0.08714431874203259,
        },
        {
            "input_ids": [128000, 128006, 882, 128007, 271, 3923, 374, 279, 6864, 315, 9822, 30, 128009, 128006, 78191, 128007, 271, 38136, 1907, 128009],
            "attention_mask": [1] * 20,  # Madrid has 2 tokens so 1 extra
            "labels": [-100] * 17 + [38136, 1907, 128009],
            "pi_target": 0.23688281808991016,
        },
    ]
    
    batch_input = data_collator(features)
    
    # Check basic structure
    assert "input_ids" in batch_input
    assert "attention_mask" in batch_input
    assert "labels" in batch_input
    assert "pi_target" in batch_input
    
    # Check that pi_target values sum to 1.0 (within tolerance)
    pi_sum = batch_input["pi_target"].sum().item()
    assert abs(pi_sum - 1.0) < 1e-6, f"Pi targets should sum to 1.0, got {pi_sum}"
    
    # Check that variable length sequences are handled (Madrid has 2 tokens vs others 1)
    # All sequences should be padded to the same length
    seq_lengths = batch_input["attention_mask"].sum(dim=1)
    max_len = seq_lengths.max().item()
    
    # The longest sequence (Madrid) should determine the padded length
    assert seq_lengths[3].item() == 20  # Madrid sequence
    assert seq_lengths[0].item() == 19  # Other sequences
    
    # Check that sequences are properly padded to multiple of 4
    padded_len = batch_input["input_ids"].shape[1]
    assert padded_len % 4 == 0
    assert padded_len >= max_len
