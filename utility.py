# utility.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch




def fit_from_costs_p(
    cfg,
    all_costs: torch.Tensor,
    p: float = 0.95,
    *,
    set_tau: bool = True,
    set_cmax: bool = True,
    cmax_use: Literal["quantile", "max"] = "quantile",
):
    """
    One-knob fitting: provide a single quantile p and set tau/C_max automatically.
    - tau := quantile(all_costs, p) if set_tau
    - C_max := quantile(all_costs, p) if set_cmax and cmax_use="quantile"
            := max(all_costs)        if set_cmax and cmax_use="max"
    """
    if all_costs.dim() != 1:
        all_costs = all_costs.view(-1)
    all_costs = all_costs.detach().to(torch.float32)

    p = float(p)
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0,1].")

    qv = float(torch.quantile(all_costs, p).item())
    mx = float(all_costs.max().item())

    if set_tau:
        cfg.tau = qv
    if set_cmax:
        cfg.C_max = qv if cmax_use == "quantile" else mx

    return cfg

# keep it for more complex analysis later --- IGNORE ---
def fit_utility_params_from_costs(
    cfg,
    all_costs: torch.Tensor,
    *,
    tau_quantile: Optional[float] = 0.90,
    cmax_quantile: Optional[Union[float, str]] = "max",  # "max" or float like 0.99
):
    """
    Fit tau and/or C_max from a 1D tensor of all observed costs.

    all_costs: (N,) tensor
    tau_quantile:
      - if not None and cfg.tau is None, set tau = quantile(all_costs, tau_quantile)
    cmax_quantile:
      - "max": set C_max = max(all_costs)
      - float: set C_max = quantile(all_costs, float)
      - None: do nothing
    """
    if all_costs.dim() != 1:
        all_costs = all_costs.view(-1)

    all_costs = all_costs.detach().to(torch.float32)

    if cfg.tau is None and tau_quantile is not None:
        q = float(tau_quantile)
        if not (0.0 < q <= 1.0):
            raise ValueError("tau_quantile must be in (0,1].")
        cfg.tau = float(torch.quantile(all_costs, q).item())

    if cfg.C_max is None and cmax_quantile is not None:
        if isinstance(cmax_quantile, str):
            if cmax_quantile != "max":
                raise ValueError("cmax_quantile string must be 'max'.")
            cfg.C_max = float(all_costs.max().item())
        else:
            q = float(cmax_quantile)
            if not (0.0 < q <= 1.0):
                raise ValueError("cmax_quantile must be in (0,1].")
            cfg.C_max = float(torch.quantile(all_costs, q).item())

    return cfg


def normalize_cost(costs: torch.Tensor, cfg) -> torch.Tensor:
    mode = cfg.cost_norm

    if mode == "none":
        c_tilde = costs

    elif mode == "linear_clip":
        if cfg.tau is None:
            raise ValueError("cfg.tau is None. Call fit_utility_params_from_costs(...) or set tau manually.")
        c_tilde = costs / float(cfg.tau)
        c_tilde = torch.minimum(c_tilde, torch.ones_like(c_tilde))

    elif mode == "log_norm":
        if cfg.C_max is None:
            raise ValueError("cfg.C_max is None. Call fit_utility_params_from_costs(...) or set C_max manually.")
        denom = torch.log1p(torch.tensor(float(cfg.C_max), device=costs.device, dtype=costs.dtype)) + float(cfg.eps)
        c_tilde = torch.log1p(costs) / denom

    elif mode == "per_sample_minmax":
        if costs.dim() == 1:
            cmin = costs.min()
            cmax = costs.max()
            c_tilde = (costs - cmin) / (cmax - cmin + float(cfg.eps))
        elif costs.dim() == 2:
            cmin = costs.min(dim=1, keepdim=True).values
            cmax = costs.max(dim=1, keepdim=True).values
            c_tilde = (costs - cmin) / (cmax - cmin + float(cfg.eps))
        else:
            raise ValueError(f"costs must be 1D or 2D, got {tuple(costs.shape)}")

    else:
        raise ValueError(f"Unknown cost_norm mode: {mode}")

    if cfg.clamp_cost_01:
        c_tilde = torch.clamp(c_tilde, 0.0, 1.0)

    return c_tilde


def reward_per_action(quality: torch.Tensor, costs: torch.Tensor, cfg) -> torch.Tensor:
    mode = str(getattr(cfg, "reward_mode", "exp"))

    if mode == "exp":
        c_tilde = normalize_cost(costs, cfg)
        r = quality * torch.exp(-float(cfg.lam) * c_tilde)
        if cfg.clamp_reward_01:
            r = torch.clamp(r, 0.0, 1.0)
        return r

    elif mode == "log_reward":
        c_tilde = normalize_cost(costs, cfg)
        lam = float(getattr(cfg, "lam", 1.0))
        eps_q = float(getattr(cfg, "quality_eps", 1e-7))
        r = torch.log(quality + eps_q) - lam * c_tilde
        return r

    elif mode == "softmax_prob":
        r = reward_per_action_softmaxprob(quality, costs, cfg)
        # softmax 本身就在 [0,1]，也不需要 clamp_reward_01
        return r

    raise ValueError(f"Unknown reward_mode: {mode}")

def utility_per_action(quality: torch.Tensor, costs: torch.Tensor, cfg) -> torch.Tensor:
    """
    u_a = quality_a - alpha * c_tilde_a
    returns u with same shape as quality/costs (1D or 2D)
    """
    c_tilde = normalize_cost(costs, cfg)
    alpha = float(getattr(cfg, "alpha", 1.0))
    return quality - alpha * c_tilde

def reward_per_action_softmaxprob(quality: torch.Tensor, costs: torch.Tensor, cfg) -> torch.Tensor:
    """
    r_a = softmax(u/T)_a where u = utility_per_action(...)
    output is in (0,1) and sums to 1 over actions for each sample.
    """
    u = utility_per_action(quality, costs, cfg)

    T = float(getattr(cfg, "softmax_T", 1.0))
    if T <= 0:
        raise ValueError(f"softmax_T must be > 0, got {T}")

    if u.dim() == 1:
        return torch.softmax(u / T, dim=0)

    if u.dim() == 2:
        return torch.softmax(u / T, dim=1)

    raise ValueError(f"quality/costs must be 1D or 2D, got {tuple(u.shape)}")


def reward_of_action(quality: torch.Tensor, costs: torch.Tensor, a: torch.Tensor, cfg) -> torch.Tensor:
    r_all = reward_per_action(quality, costs, cfg)

    if r_all.dim() == 1:
        ai = int(a.item()) if a.numel() == 1 else int(a.view(-1)[0].item())
        return r_all[ai]

    if r_all.dim() == 2:
        if a.dim() != 1:
            raise ValueError(f"a must be (B,), got {tuple(a.shape)}")
        B = r_all.size(0)
        idx = torch.arange(B, device=r_all.device)
        return r_all[idx, a.long()]

    raise ValueError(f"quality/costs must be 1D or 2D, got {tuple(r_all.shape)}")