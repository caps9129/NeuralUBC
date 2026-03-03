# ucb_linear.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F


class PerActionLinearUCB:
    """
    Per-action LinearUCB on a feature vector h(x, a) (or shared h(x)).

    For each action a:
      A_a = lambda * I + sum h h^T
      b_a = sum r * h
      theta_a = A_a^{-1} b_a

    UCB(x,a) = theta_a^T h + beta * sqrt( h^T A_a^{-1} h )

    This class tries to match your existing interface:
      - select_action(model, x) -> (a_star, ucb_scores, mean_scores)
      - update(a_int, h_vec, reward=None)
      - rebuild_from_buffer(model, dataset, buffer, device_for_model, batch_size, verbose)
      - reset_A(), state_dict(), load_state_dict()

    Notes:
      - If you pass reward to update(...), it will also update b/theta.
      - If reward is None, it only updates covariance (A_inv) (useful for decision-time update),
        but then mean_scores will be near 0 unless you also update with rewards somewhere.
      - Recommended: during decide, compute reward for chosen action and call update(a, h, r).
    """

    def __init__(
        self,
        num_actions: int,
        feat_dim: int,
        cfg,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if cfg is None:
            raise ValueError("cfg must be provided (pass cfg.ucb from config.py).")

        self.num_actions = int(num_actions)
        self.feat_dim = int(feat_dim)
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype

        lam = float(getattr(self.cfg, "lambda_", getattr(self.cfg, "lambda", 1.0)))
        if lam <= 0:
            raise ValueError(f"cfg.ucb.lambda_ must be > 0, got {lam}")

        # per-action A_inv and b
        eye = torch.eye(self.feat_dim, device=self.device, dtype=self.dtype)
        self.A_inv = (1.0 / lam) * eye.unsqueeze(0).repeat(self.num_actions, 1, 1)  # (K,P,P)
        self.b = torch.zeros(self.num_actions, self.feat_dim, device=self.device, dtype=self.dtype)  # (K,P)

        # debug
        self.counts = torch.zeros(self.num_actions, device=self.device, dtype=torch.long)
        self.total_updates = torch.zeros((), device=self.device, dtype=torch.long)
        self._step = 0

    @torch.no_grad()
    def _maybe_normalize(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, dtype=self.dtype)
        if bool(getattr(self.cfg, "l2_normalize_ucb_feat", False)):
            h = F.normalize(h, p=2, dim=-1)
        return h

    @torch.no_grad()
    def _ensure_2d(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            return h.unsqueeze(0)
        if h.dim() == 2:
            return h
        raise ValueError(f"h must be (P,) or (B,P), got {tuple(h.shape)}")

    @torch.no_grad()
    def mean_from_h_all_actions(self, h_all: torch.Tensor) -> torch.Tensor:
        """
        h_all: (B,K,P)
        return mean_scores: (B,K) where mean[b,a] = theta_a^T h[b,a]
        """
        # theta = A_inv @ b
        # A_inv: (K,P,P), b:(K,P) => theta:(K,P)
        # theta = torch.einsum("kpp,kp->kp", self.A_inv, self.b)  # (K,P)
        theta = torch.einsum("kpq,kq->kp", self.A_inv, self.b)
        mean = torch.einsum("bkp,kp->bk", h_all, theta)
        return mean

    @torch.no_grad()
    def bonus_from_h_all_actions(self, h_all: torch.Tensor) -> torch.Tensor:
        """
        h_all: (B,K,P)
        return bonus: (B,K)
        """
        # quad[b,k] = h[b,k]^T A_inv[k] h[b,k]
        tmp = torch.einsum("bkp,kpq->bkq", h_all, self.A_inv)     # (B,K,P)
        quad = torch.einsum("bkq,bkq->bk", tmp, h_all)            # (B,K)
        quad = torch.clamp(quad, min=0.0)

        beta = float(getattr(self.cfg, "beta", 1.0))
        eps = float(getattr(self.cfg, "eps", 1e-8))
        return beta * torch.sqrt(quad + eps)

    @torch.no_grad()
    def select_action(self, model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Match your interface:
        x: (B,D)
        returns:
          a_star: (B,)
          ucb_scores: (B,K)
          mean_scores: (B,K)
        """
        if x.dim() != 2:
            raise ValueError(f"x must be (B,D), got {tuple(x.shape)}")

        model.eval()

        x0_device = x.device
        x = x.to(self.device)

        B = x.size(0)
        K = self.num_actions
        P = self.feat_dim

        # Build features for all actions: h_all (B,K,P)
        # Option A: model.get_last_hidden(x, a_tensor) exists -> action-specific features
        # Option B: shared features model.get_last_hidden(x, None) -> same h for all actions
        h_list = []
        for a in range(K):
            a_tensor = torch.full((B,), a, device=self.device, dtype=torch.long)
            try:
                h = model.get_last_hidden(x, a_tensor)  # (B,P)
            except TypeError:
                # if your get_last_hidden(x, None) style
                h = model.get_last_hidden(x, None)      # (B,P)
            h = self._maybe_normalize(h)
            if h.shape[-1] != P:
                raise ValueError(f"feat dim mismatch: expected {P}, got {h.shape[-1]}")
            h_list.append(h)

        h_all = torch.stack(h_list, dim=1)  # (B,K,P)

        mean_scores = self.mean_from_h_all_actions(h_all)    # (B,K)
        bonus = self.bonus_from_h_all_actions(h_all)         # (B,K)
        ucb_scores = mean_scores + bonus
        a_star = torch.argmax(ucb_scores, dim=1).long()

        # debug (same vibe as yours)
        dbg_every = int(getattr(self.cfg, "debug_every", 0) or 0)
        self._step += 1
        if dbg_every > 0 and (self._step % dbg_every) == 0:
            a_mean = torch.argmax(mean_scores, dim=1)
            change_ratio = (a_star != a_mean).float().mean().item()
            ms_range = (mean_scores.max() - mean_scores.min()).item()
            bm_range = (bonus.max() - bonus.min()).item()
            ms_std = mean_scores.std().item()
            bm_std = bonus.std().item()
            print(
                f"[LinUCB dbg step={self._step}] "
                f"mean_range={ms_range:.3e} mean_std={ms_std:.3e} | "
                f"bonus_range={bm_range:.3e} bonus_std={bm_std:.3e} | "
                f"change_ratio={change_ratio:.3f}"
            )

        return a_star.to(x0_device), ucb_scores.to(x0_device), mean_scores.to(x0_device)

    @torch.no_grad()
    def update(self, a_int: int, h_vec: torch.Tensor, reward: Optional[float] = None):
        """
        Update per-action A_inv (and optionally b) with Sherman-Morrison.

        a_int: chosen action id
        h_vec: (P,) or (P,1) or (1,P)
        reward: scalar reward for the chosen action (recommended). If None, only updates A_inv.
        """
        ai = int(a_int)
        if not (0 <= ai < self.num_actions):
            raise ValueError(f"a_int out of range: {ai} not in [0, {self.num_actions})")

        h = h_vec.detach()
        if h.dim() == 2:
            if h.size(0) == 1:
                h = h.squeeze(0)
            elif h.size(1) == 1:
                h = h.squeeze(1)
            else:
                raise ValueError(f"h_vec has unexpected 2D shape {tuple(h.shape)}")
        elif h.dim() != 1:
            raise ValueError(f"h_vec must be 1D (P,) or squeezeable, got {tuple(h.shape)}")

        if h.numel() != self.feat_dim:
            raise ValueError(f"h_vec dim mismatch: expected {self.feat_dim}, got {h.numel()}")

        h = self._maybe_normalize(h)  # (P,)
        h_col = h.view(-1, 1)         # (P,1)

        Ainv = self.A_inv[ai]         # (P,P)
        Ah = Ainv @ h_col             # (P,1)
        denom = 1.0 + (h_col.t() @ Ah).item()
        if denom <= 0:
            return

        self.A_inv[ai] = Ainv - (Ah @ Ah.t()) / float(denom)

        # optional b update (needed for meaningful mean)
        if reward is not None:
            r = float(reward)
            self.b[ai] = self.b[ai] + r * h

        self.counts[ai] += 1
        self.total_updates += 1

    @torch.no_grad()
    def reset_A(self):
        lam = float(getattr(self.cfg, "lambda_", getattr(self.cfg, "lambda", 1.0)))
        eye = torch.eye(self.feat_dim, device=self.device, dtype=self.dtype)
        self.A_inv = (1.0 / lam) * eye.unsqueeze(0).repeat(self.num_actions, 1, 1)
        self.b.zero_()
        self.counts.zero_()
        self.total_updates = torch.zeros((), device=self.device, dtype=torch.long)
        self._step = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "A_inv": self.A_inv.detach().cpu(),
            "b": self.b.detach().cpu(),
            "counts": self.counts.detach().cpu(),
            "total_updates": int(self.total_updates.item()),
            "step": int(self._step),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.A_inv = state["A_inv"].to(self.device, dtype=self.dtype)
        self.b = state.get("b", torch.zeros(self.num_actions, self.feat_dim)).to(self.device, dtype=self.dtype)
        self.counts = state.get("counts", torch.zeros(self.num_actions)).to(self.device, dtype=torch.long)
        self.total_updates = torch.tensor(int(state.get("total_updates", 0)), device=self.device, dtype=torch.long)
        self._step = int(state.get("step", 0))

    @torch.no_grad()
    def rebuild_from_buffer(
        self,
        model,
        dataset,
        buffer: List[Tuple[int, int]],  # [(data_idx, a_int), ...]
        device_for_model: str,
        batch_size: int = 256,
        verbose: bool = False,
        reward_fn=None,  # optional: callable (q, c, a_int, cfg.utility) -> r scalar
        utility_cfg=None,
    ):
        """
        Rebuild A_inv (and optionally b) from replay buffer.

        If you pass reward_fn + utility_cfg, it will also rebuild b using true reward.
        Otherwise it only rebuilds A_inv.
        """
        if len(buffer) == 0:
            self.reset_A()
            return

        self.reset_A()
        by_action = defaultdict(list)
        for data_idx, a_int in buffer:
            by_action[int(a_int)].append(int(data_idx))

        model.eval()
        total_used = 0

        for a_int, idx_list in by_action.items():
            if not (0 <= a_int < self.num_actions):
                continue

            for s in range(0, len(idx_list), batch_size):
                chunk = idx_list[s : s + batch_size]

                xs, qs, cs = [], [], []
                for di in chunk:
                    x, q, c, _ = dataset[di]
                    xs.append(x); qs.append(q); cs.append(c)

                xb = torch.stack([t.to(device_for_model) for t in xs], dim=0)

                a_tensor = torch.full((xb.size(0),), a_int, device=device_for_model, dtype=torch.long)
                try:
                    h = model.get_last_hidden(xb, a_tensor)
                except TypeError:
                    h = model.get_last_hidden(xb, None)

                h = h.to(self.device, dtype=self.dtype)
                h = self._maybe_normalize(h)

                # compute rewards if asked
                rewards = None
                if reward_fn is not None:
                    if utility_cfg is None:
                        raise ValueError("utility_cfg must be provided if reward_fn is provided.")
                    qb = torch.stack([t.to(device_for_model) for t in qs], dim=0)
                    cb = torch.stack([t.to(device_for_model) for t in cs], dim=0)
                    a_b = torch.full((xb.size(0),), a_int, device=device_for_model, dtype=torch.long)
                    r_b = reward_fn(qb, cb, a_b, utility_cfg)  # (B,)
                    rewards = r_b.detach().to("cpu").tolist()

                for i in range(h.size(0)):
                    r_i = None if rewards is None else float(rewards[i])
                    self.update(a_int, h[i], reward=r_i)
                total_used += h.size(0)

        if verbose:
            print(
                f"[LinUCB rebuild] buffer={len(buffer)} used={total_used} "
                f"counts_sum={int(self.counts.sum().item())}"
            )