"""Lambda scheduler for λ-DPO that samples from a fixed table."""

import random
from typing import Sequence, Tuple

import torch


class TableLambdaScheduler:
    """Sample λ vectors from a predefined table with custom probabilities."""

    #: Fixed set of λ vectors and their associated sampling probabilities. The
    #: probabilities correspond to the softmax values shown in the paper table.
    TABLE: Sequence[Tuple[Sequence[float], float]] = (
        ([0.212, 0.334, 0.245, 0.209], 0.095),
        ([0.126, 0.237, 0.131, 0.507], 0.075),
        ([0.542, 0.079, 0.256, 0.123], 0.117),
        ([0.233, 0.721, 0.020, 0.025], 0.185),
        ([0.004, 0.334, 0.281, 0.381], 0.067),
        ([0.320, 0.069, 0.513, 0.099], 0.082),
        ([0.236, 0.155, 0.358, 0.251], 0.093),
        ([0.141, 0.110, 0.701, 0.048], 0.110),
        ([0.070, 0.198, 0.388, 0.344], 0.081),
        ([0.090, 0.139, 0.306, 0.465], 0.095),
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


