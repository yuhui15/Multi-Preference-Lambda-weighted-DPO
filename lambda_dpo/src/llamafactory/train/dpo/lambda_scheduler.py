"""Lambda scheduler for λ-DPO that samples from a fixed table."""

import random
from typing import Sequence, Tuple

import torch


class TableLambdaScheduler:
    """Sample λ vectors from a fixed scheduler table."""

    #: Fixed set of λ vectors and their associated sampling probabilities.
    TABLE: Sequence[Tuple[Sequence[float], float]] = (
        ([0.469, 0.092, 0.198, 0.242], 0.102),
        ([0.013, 0.341, 0.623, 0.022], 0.094),
        ([0.193, 0.321, 0.345, 0.141], 0.108),
        ([0.445, 0.128, 0.207, 0.221], 0.103),
        ([0.562, 0.069, 0.238, 0.131], 0.097),
        ([0.105, 0.161, 0.425, 0.310], 0.105),
        ([0.110, 0.011, 0.465, 0.415], 0.098),
        ([0.163, 0.112, 0.616, 0.109], 0.100),
        ([0.018, 0.447, 0.527, 0.008], 0.095),
        ([0.213, 0.513, 0.082, 0.193], 0.098),
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
