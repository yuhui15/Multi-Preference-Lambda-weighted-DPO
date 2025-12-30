"""Lambda scheduler for λ-DPO that samples from a fixed table."""

import random
from typing import Sequence, Tuple

import torch


class TableLambdaScheduler:
    """Sample λ vectors. Currently uses a uniform weight across dimensions."""

    #: Fixed set of λ vectors and their associated sampling probabilities.
    #: For uniform weighting, we only keep a single entry.
    TABLE: Sequence[Tuple[Sequence[float], float]] = (
        ([0.0, 1.0, 0.0, 0.0], 1.0),
    )

    def __init__(self, seed: int = 42) -> None:
        self.rand = random.Random(seed)

        lambdas, probs = zip(*self.TABLE)
        self.lambdas = [torch.tensor(l, dtype=torch.float32) for l in lambdas]
        total = sum(probs)
        self.probs = [p / total for p in probs]

    def sample(self) -> torch.Tensor:
        r = self.rand.random()
        cum = 0.0
        for lmbd, p in zip(self.lambdas, self.probs):
            cum += p
            if r <= cum:
                return lmbd.clone()
        return self.lambdas[-1].clone()
