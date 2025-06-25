import random
from torch.utils.data import Dataset

from llamafactory.data import BlockShuffleSampler


class RangeDataset(Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return idx


def test_block_shuffle_sampler_groups():
    dataset = RangeDataset(32)
    sampler = BlockShuffleSampler(dataset, block_size=16, seed=42)
    indices = list(iter(sampler))
    assert len(indices) == 32
    first_block = indices[:16]
    second_block = indices[16:]
    assert first_block in [list(range(16)), list(range(16, 32))]
    assert second_block in [list(range(16)), list(range(16, 32))]
    assert first_block == sorted(first_block)
    assert second_block == sorted(second_block)
