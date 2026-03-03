from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


class BasePolicy:
    """
    Common interface for all policies.
    Must implement:
      select_action(x, quality, costs, meta) -> int (action index)
    """

    def select_action(
        self,
        x: torch.Tensor,         # (D,) or (1,D)
        quality: torch.Tensor,   # (K,)
        costs: torch.Tensor,     # (K,)
        meta: Optional[dict] = None,
    ) -> int:
        raise NotImplementedError


@dataclass
class RandomPolicy(BasePolicy):
    seed: int = 42

    def __post_init__(self):
        self._g = torch.Generator()
        self._g.manual_seed(int(self.seed))

    def select_action(self, x, quality, costs, meta=None) -> int:
        K = int(quality.numel())
        a = torch.randint(low=0, high=K, size=(1,), generator=self._g).item()
        return int(a)


class MaxQualityPolicy(BasePolicy):
    """Always choose argmax quality."""
    def select_action(self, x, quality, costs, meta=None) -> int:
        return int(torch.argmax(quality).item())


class MinCostPolicy(BasePolicy):
    """Always choose argmin cost."""
    def select_action(self, x, quality, costs, meta=None) -> int:
        return int(torch.argmin(costs).item())


class UCBPolicy(BasePolicy):
    """
    Adapter to use your UCB class (PerActionLastLayerUCB) as a policy.
    Expects ucb.select_action(model, x_batch) -> (a_star, ucb_scores, mean_scores)
    """

    def __init__(self, model, ucb):
        self.model = model
        self.ucb = ucb

    def select_action(self, x, quality, costs, meta=None) -> int:
        # ensure (1,D)
        if x.dim() == 1:
            xb = x.unsqueeze(0)
        else:
            xb = x
        a_star, _, _ = self.ucb.select_action(self.model, xb)
        return int(a_star.item())