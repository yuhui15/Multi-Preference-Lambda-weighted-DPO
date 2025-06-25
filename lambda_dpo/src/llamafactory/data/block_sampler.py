import random
from typing import Iterator, Iterable, Sized

from torch.utils.data import Sampler


class BlockShuffleSampler(Sampler[int]):
    """Sampler that groups indices into fixed-size blocks and shuffles the blocks."""

    data_source: Sized
    block_size: int
    seed: int

    def __init__(self, data_source: Sized, block_size: int, seed: int = 0) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.data_source = data_source
        self.block_size = block_size
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.data_source)))
        blocks = [
            indices[i : i + self.block_size]
            for i in range(0, len(indices), self.block_size)
        ]
        rng = random.Random(self.seed)
        rng.shuffle(blocks)
        for block in blocks:
            for idx in block:
                yield idx

    def __len__(self) -> int:
        return len(self.data_source)
