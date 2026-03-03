# metrics.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, DefaultDict
from collections import defaultdict

import torch


def _safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


@dataclass
class OnlineMeters:
    """
    Split-level metrics for offline replay of an online contextual bandit.

    Logs (per split):
      - split/avg_reward
      - split/cumulative_reward
      - split/avg_gap
      - split/avg_quality_selected
      - split/avg_cost_selected
      - split/action_rate/<model_name>

    Plus per-domain (per split):
      - split_dom/n_rounds/<domain>
      - split_dom/avg_reward/<domain>
      - split_dom/cumulative_reward/<domain>
    """

    model_names: List[str]

    # cumulative across all splits
    cumulative_reward: float = 0.0
    cumulative_gap: float = 0.0
    total_rounds: int = 0

    # cumulative per domain across all splits
    cumulative_reward_by_domain: Dict[str, float] = field(default_factory=dict)

    # internal per-split buffers (overall)
    _split_rewards: List[float] = field(default_factory=list, init=False)
    _split_gaps: List[float] = field(default_factory=list, init=False)
    _split_qualities: List[float] = field(default_factory=list, init=False)
    _split_costs: List[float] = field(default_factory=list, init=False)
    _split_losses: List[float] = field(default_factory=list, init=False)
    _action_counts: List[int] = field(default_factory=list, init=False)

    # internal per-split buffers (by domain)
    _dom_rewards: DefaultDict[str, List[float]] = field(default_factory=lambda: defaultdict(list), init=False)
    _dom_gaps: DefaultDict[str, List[float]] = field(default_factory=lambda: defaultdict(list), init=False)
    _dom_qualities: DefaultDict[str, List[float]] = field(default_factory=lambda: defaultdict(list), init=False)
    _dom_costs: DefaultDict[str, List[float]] = field(default_factory=lambda: defaultdict(list), init=False)
    _dom_counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int), init=False)

    def __post_init__(self):
        if not self.model_names:
            raise ValueError("model_names must be non-empty")
        self._action_counts = [0 for _ in range(len(self.model_names))]

    def reset_split(self):
        """Reset split buffers (does NOT reset cumulative stats)."""
        self._split_rewards.clear()
        self._split_gaps.clear()
        self._split_qualities.clear()
        self._split_costs.clear()
        self._split_losses.clear()
        self._action_counts = [0 for _ in range(len(self.model_names))]

        self._dom_rewards.clear()
        self._dom_gaps.clear()
        self._dom_qualities.clear()
        self._dom_costs.clear()
        self._dom_counts.clear()

    def update(
        self,
        *,
        reward: float,
        gap: float,
        quality: float,
        cost: float,
        action: int,
        domain: str,
        loss: Optional[float] = None,
        ignore_cumulative: bool = False,  # for ablations where reward is not meaningful
    ):
        """
        Update meters for one interaction round.
        """
        # overall buffers
        self._split_rewards.append(float(reward))
        self._split_gaps.append(float(gap))
        self._split_qualities.append(float(quality))
        self._split_costs.append(float(cost))
        if loss is not None:
            self._split_losses.append(float(loss))

        if action < 0 or action >= len(self.model_names):
            raise ValueError(f"action index out of range: {action}")
        self._action_counts[action] += 1

        # per-domain buffers
        d = str(domain)
        self._dom_rewards[d].append(float(reward))
        self._dom_gaps[d].append(float(gap))
        self._dom_qualities[d].append(float(quality))
        self._dom_costs[d].append(float(cost))
        self._dom_counts[d] += 1

        # cumulative overall
        self.total_rounds += 1
        if not ignore_cumulative:
            self.cumulative_reward += float(reward)
            self.cumulative_gap += float(gap)
            

            if d not in self.cumulative_reward_by_domain:
                self.cumulative_reward_by_domain[d] = 0.0
            self.cumulative_reward_by_domain[d] += float(reward)

    def split_summary(self, split_idx: int) -> Dict[str, float]:
        """
        Produce a dict of metrics for wandb logging at the end of a split.
        """
        n = len(self._split_rewards)

        out: Dict[str, float] = {
            "split/idx": float(split_idx),
            "split/n_rounds": float(n),
            "split/avg_reward": _safe_mean(self._split_rewards) if n > 0 else 0.0,
            # "split/cumulative_reward": float(self.cumulative_reward),
            "split/cum_reward": float(self.cumulative_reward),
            "split/avg_gap": _safe_mean(self._split_gaps) if n > 0 else 0.0,
            "split/avg_quality_selected": _safe_mean(self._split_qualities) if n > 0 else 0.0,
            "split/avg_cost_selected": _safe_mean(self._split_costs) if n > 0 else 0.0,
        }

        if len(self._split_losses) > 0:
            out["split/avg_mse_loss"] = _safe_mean(self._split_losses)

        # overall per-model selection rates
        if n > 0:
            for i, name in enumerate(self.model_names):
                out[f"split/action_rate/{name}"] = float(self._action_counts[i] / n)
        else:
            for name in self.model_names:
                out[f"split/action_rate/{name}"] = 0.0

        # per-domain metrics
        # log counts too, so you can judge stability of domain curves
        # for d, cnt in self._dom_counts.items():
        #     out[f"split_dom/n_rounds/{d}"] = float(cnt)
        #     out[f"split_dom/avg_reward/{d}"] = _safe_mean(self._dom_rewards[d])
        #     # cumulative reward by domain (across splits)
        #     out[f"split_dom/cumulative_reward/{d}"] = float(self.cumulative_reward_by_domain.get(d, 0.0))

        return out

    def running_debug(self) -> Dict[str, float]:
        """
        Optional: log during training (every log_every steps).
        Uses recent values in current split buffers.
        """
        out: Dict[str, float] = {}
        if len(self._split_rewards) > 0:
            out["train/reward_running_mean"] = _safe_mean(self._split_rewards[-200:])
        if len(self._split_losses) > 0:
            out["train/mse_loss_running_mean"] = _safe_mean(self._split_losses[-200:])
        return out