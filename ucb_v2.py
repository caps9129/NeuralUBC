from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn.functional as F


class PerActionLastLayerUCB:
    """
    Last-layer UCB with PER-ACTION (independent) covariance.

    UCB(x,a) = mu(x,a) + beta * sqrt( h(x,a)^T A_a^{-1} h(x,a) )

    New model inputs:
      - x_emb: (B, D_emb)
      - x_feat: (B, D_feat)
      - domain_id: (B,)
    """

    def __init__(
        self,
        num_actions: int,
        hidden_dim: int,
        cfg,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if cfg is None:
            raise ValueError("cfg must be provided (pass cfg.ucb from config.py).")

        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype

        self.ucb_feat_dim = self.hidden_dim + 1  # P = H+1
        lam = float(getattr(cfg, "lambda_", getattr(cfg, "lambda", 1.0)))

        # Per-action A_inv: (K, P, P)
        I = torch.eye(self.ucb_feat_dim, device=self.device, dtype=self.dtype)
        self.A_inv = (1.0 / lam) * I.unsqueeze(0).repeat(self.num_actions, 1, 1)

        self.counts = torch.zeros(self.num_actions, device=self.device, dtype=torch.long)
        self.total_updates = torch.zeros((), device=self.device, dtype=torch.long)

        self._step = 0

    @torch.no_grad()
    def _normalize_h(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, dtype=self.dtype)
        return F.normalize(h, p=2, dim=-1)

    @torch.no_grad()
    def bonus_from_h_and_action(self, h: torch.Tensor, a_int: int) -> torch.Tensor:
        """
        h: (B,H) for a fixed action a_int
        returns: (B,)
        """
        if h.dim() != 2:
            raise ValueError(f"h must be (B,H), got {tuple(h.shape)}")
        B, H = h.shape
        if H != self.hidden_dim:
            raise ValueError(f"h hidden dim mismatch: expected {self.hidden_dim}, got {H}")

        ai = int(a_int)
        if not (0 <= ai < self.num_actions):
            raise ValueError(f"a_int out of range: {ai} not in [0, {self.num_actions})")

        h = h.to(self.device, dtype=self.dtype)

        if bool(getattr(self.cfg, "l2_normalize_ucb_feat", False)):
            h = self._normalize_h(h)

        ones = torch.ones((B, 1), dtype=h.dtype, device=h.device)
        g = torch.cat([h, ones], dim=1)  # (B, P)

        Ainv = self.A_inv[ai]  # (P,P)
        if Ainv.shape != (g.shape[1], g.shape[1]):
            raise ValueError(f"A_inv[a] shape mismatch: got {tuple(Ainv.shape)}, expected {(g.shape[1], g.shape[1])}")

        tmp = torch.matmul(g, Ainv)      # (B, P)
        quad = (tmp * g).sum(dim=-1)     # (B,)
        quad = torch.clamp(quad, min=0.0)

        beta = float(getattr(self.cfg, "beta", 1.0))
        eps = float(getattr(self.cfg, "eps", 1e-8))
        return beta * torch.sqrt(quad + eps)

    @torch.no_grad()
    def select_action(self, model, x_emb: torch.Tensor, x_feat: torch.Tensor, domain_id: torch.Tensor):
        """
        Select argmax_a mean(x,a) + bonus(x,a) for each sample in batch.

        Inputs:
          x_emb: (B, D_emb)
          x_feat: (B, D_feat)
          domain_id: (B,)
        """
        if x_emb.dim() != 2:
            raise ValueError(f"x_emb must be (B,D), got {tuple(x_emb.shape)}")
        if x_feat.dim() != 2:
            raise ValueError(f"x_feat must be (B,F), got {tuple(x_feat.shape)}")
        if domain_id.dim() != 1:
            raise ValueError(f"domain_id must be (B,), got {tuple(domain_id.shape)}")
        if x_emb.size(0) != x_feat.size(0) or x_emb.size(0) != domain_id.size(0):
            raise ValueError("Batch size mismatch among x_emb/x_feat/domain_id")

        model.eval()

        x0_device = x_emb.device

        # move to UCB device
        x_emb = x_emb.to(self.device, dtype=self.dtype)
        x_feat = x_feat.to(self.device, dtype=self.dtype)
        domain_id = domain_id.to(self.device)

        B = x_emb.size(0)
        K = self.num_actions

        mean_scores = model.forward_all_actions(x_emb, x_feat, domain_id)  # (B,K)

        bonuses = []
        for a in range(K):
            a_tensor = torch.full((B,), a, device=self.device, dtype=torch.long)
            h = model.get_last_hidden(x_emb, x_feat, domain_id, a_tensor)  # (B,H)
            b = self.bonus_from_h_and_action(h, a)                         # (B,)
            bonuses.append(b)

        bonus_mat = torch.stack(bonuses, dim=1)    # (B,K)
        ucb_scores = mean_scores + bonus_mat
        a_star = torch.argmax(ucb_scores, dim=1).long()

        dbg_every = int(getattr(self.cfg, "debug_every", 0) or 0)
        self._step += 1
        if dbg_every > 0 and (self._step % dbg_every) == 0:
            a_mean = torch.argmax(mean_scores, dim=1)
            change_ratio = (a_star != a_mean).float().mean().item()

            ms_range = (mean_scores.max() - mean_scores.min()).item()
            bm_range = (bonus_mat.max() - bonus_mat.min()).item()
            ms_std = mean_scores.std().item()
            bm_std = bonus_mat.std().item()

            print(
                f"[UCB dbg step={self._step}] "
                f"mean_range={ms_range:.3e} mean_std={ms_std:.3e} | "
                f"bonus_range={bm_range:.3e} bonus_std={bm_std:.3e} | "
                f"change_ratio={change_ratio:.3f}"
            )

            top2 = torch.topk(mean_scores, k=2, dim=1).values
            mean_margin = (top2[:, 0] - top2[:, 1]).mean().item()

            a1 = torch.argmax(mean_scores, dim=1)
            a2 = torch.topk(mean_scores, k=2, dim=1).indices[:, 1]
            bonus_margin = (
                bonus_mat[torch.arange(B, device=self.device), a2]
                - bonus_mat[torch.arange(B, device=self.device), a1]
            ).mean().item()
            print(f"mean_margin={mean_margin:.3e} bonus_margin={bonus_margin:.3e}")

        return (
            a_star.to(x0_device),
            ucb_scores.to(x0_device),
            mean_scores.to(x0_device),
        )

    @torch.no_grad()
    def update(self, a_int: int, h_vec: torch.Tensor):
        update_every = int(getattr(self.cfg, "update_A_every", 1) or 1)
        if update_every > 1 and (self._step % update_every) != 0:
            return

        ai = int(a_int)
        if not (0 <= ai < self.num_actions):
            raise ValueError(f"a_int out of range: {ai} not in [0, {self.num_actions})")

        hi = h_vec.detach()

        if hi.dim() == 2:
            if hi.size(0) == 1:
                hi = hi.squeeze(0)
            elif hi.size(1) == 1:
                hi = hi.squeeze(1)
            else:
                raise ValueError(f"h_vec has unexpected 2D shape {tuple(hi.shape)}")
        elif hi.dim() != 1:
            raise ValueError(f"h_vec must be 1D (H,) (or squeezeable), got {tuple(hi.shape)}")

        if hi.numel() != self.hidden_dim:
            raise ValueError(f"h_vec dim mismatch: expected {self.hidden_dim}, got {hi.numel()}")

        hi = hi.to(self.device, dtype=self.dtype)

        if bool(getattr(self.cfg, "l2_normalize_ucb_feat", False)):
            hi = self._normalize_h(hi)

        one = torch.ones(1, dtype=hi.dtype, device=hi.device)
        gi = torch.cat([hi, one], dim=0)  # (P,)

        g = gi.view(-1, 1)               # (P,1)
        Ainv = self.A_inv[ai]            # (P,P)

        Ag = Ainv @ g                    # (P,1)
        denom = 1.0 + (g.t() @ Ag).item()
        if denom <= 0:
            return

        self.A_inv[ai] = Ainv - (Ag @ Ag.t()) / float(denom)

        self.counts[ai] += 1
        self.total_updates += 1

    def state_dict(self):
        return {
            "A_inv": self.A_inv.detach().cpu(),   # (K,P,P)
            "counts": self.counts.detach().cpu(),
            "total_updates": int(self.total_updates.item()),
            "step": int(self._step),
        }

    def load_state_dict(self, state):
        self.A_inv = state["A_inv"].to(self.device, dtype=self.dtype)
        self.counts = state.get("counts", torch.zeros(self.num_actions)).to(self.device)
        self.total_updates = torch.tensor(int(state.get("total_updates", 0)), device=self.device)
        self._step = int(state.get("step", 0))

    @torch.no_grad()
    def reset_A(self):
        P = self.hidden_dim + 1
        lam = float(getattr(self.cfg, "lambda_", getattr(self.cfg, "lambda", 1.0)))
        I = torch.eye(P, device=self.device, dtype=self.dtype)
        self.A_inv = (1.0 / lam) * I.unsqueeze(0).repeat(self.num_actions, 1, 1)
        self.counts = torch.zeros(self.num_actions, device=self.device, dtype=torch.long)
        self.total_updates = torch.zeros((), device=self.device, dtype=torch.long)

    @torch.no_grad()
    def rebuild_from_buffer(
        self,
        model,
        dataset,
        buffer,
        device_for_model: str,
        batch_size: int = 256,
        verbose: bool = False,
    ):
        """
        buffer: List[(data_idx, a_int)]
        dataset[di] must return:
          x_emb, x_feat, domain_id, quality, costs, meta
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

                xembs, xfeats, dids = [], [], []
                for di in chunk:
                    x_emb, x_feat, domain_id, q, c, m = dataset[di]
                    xembs.append(x_emb)
                    xfeats.append(x_feat)
                    dids.append(domain_id)

                xb = torch.stack(xembs, dim=0).to(device_for_model)          # (B,D_emb)
                xf = torch.stack(xfeats, dim=0).to(device_for_model)         # (B,D_feat)
                did = torch.stack(dids, dim=0).to(device_for_model).long()   # (B,)

                a_tensor = torch.full(
                    (xb.size(0),),
                    int(a_int),
                    device=device_for_model,
                    dtype=torch.long,
                )

                h = model.get_last_hidden(xb, xf, did, a_tensor)  # (B,H)
                h = h.to(self.device, dtype=self.dtype)

                for i in range(h.size(0)):
                    self.update(int(a_int), h[i])
                total_used += h.size(0)

        if verbose:
            print(
                f"[UCB rebuild] buffer={len(buffer)} used={total_used} "
                f"counts_sum={int(self.counts.sum().item())}"
            )