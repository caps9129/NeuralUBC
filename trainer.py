from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utility import reward_of_action, reward_per_action
from metrics import OnlineMeters


def wandb_log(wandb_run, data: Dict[str, float], step: Optional[int] = None):
    if wandb_run is None:
        return
    if step is None:
        wandb_run.log(data)
    else:
        wandb_run.log(data, step=step)


def save_checkpoint(
    save_dir: str,
    split_idx: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ucb: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"split_{split_idx:03d}.pt")

    payload: Dict[str, Any] = {
        "split_idx": split_idx,
        "model": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if ucb is not None and hasattr(ucb, "state_dict"):
        payload["ucb"] = ucb.state_dict()
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    return path


@torch.no_grad()
def _compute_gap_for_action(
    quality: torch.Tensor,  # (K,) or (B,K)
    costs: torch.Tensor,    # (K,) or (B,K)
    a: torch.Tensor,        # () or (B,)
    utility_cfg,
) -> torch.Tensor:
    """
    gap = max_a r(x,a) - r(x,a_selected)
    (We compute oracle internally only for gap.)
    """
    r_all = reward_per_action(quality, costs, utility_cfg)
    if r_all.dim() == 1:
        oracle = r_all.max()
        chosen = r_all[int(a.item())]
        return oracle - chosen
    if r_all.dim() == 2:
        oracle = r_all.max(dim=1).values
        idx = torch.arange(r_all.size(0), device=r_all.device)
        chosen = r_all[idx, a.long()]
        return oracle - chosen
    raise ValueError(f"reward_per_action must be 1D or 2D, got {tuple(r_all.shape)}")



def train_offline_by_slice_epochs(
    *,
    cfg,
    dataset,
    split_indices: List[np.ndarray],
    model: nn.Module,
    ucb=None,
    policy=None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cuda",
    wandb_run=None,
    burn_in_splits: int = 1,
    do_model_update: bool = True,
    do_ucb_update: bool = True,
    epochs_per_slice: int = 3,
    train_pool: str = "cumulative",  # "slice_only" | "cumulative"
    train_batch_size: int = 256,
    decision_batch_size: int = 256,
    buffer_size: int = 20000,
    rebuild_batch_size: int = 256,
    rebuild_per_slice: int = 3,
    log_every_steps: int = 200,
    show_progress: bool = True,
    save_each_split: bool = False,
    save_dir: str = "./checkpoints",
) -> Any:
    """
    Slice-based offline replay (Epoch version), aligned to new dataset:
      dataset[i] -> (x_emb, x_feat, domain_id, quality[K], costs[K], meta)

    Model API assumption:
      - model(x_emb, x_feat, domain_id, a) -> pred_reward (B,)
      - model.get_last_hidden(x_emb, x_feat, domain_id, a) -> h (B, P)
      - (optional) model.gate_logits(x_emb, x_feat, domain_id) -> (B,) logits
      - (optional) model.gate_proba(x_emb, x_feat, domain_id) -> (B,) in [0,1]

    UCB API assumption:
      - ucb.select_action(model, x_emb, x_feat, domain_id) -> (a_star, ucb_scores, mean_scores)
    """
    assert train_pool in ["slice_only", "cumulative"]
    if do_model_update and optimizer is None:
        raise ValueError("do_model_update=True but optimizer is None.")
    if policy is None and ucb is None:
        raise ValueError("Provide either policy or ucb.")

    model.to(device)

    meters = OnlineMeters(
        model_names=getattr(dataset, "model_names", None) or dataset.df.iloc[0]["model_names"]
    )

    decided: List[Tuple[int, int]] = []
    replay_buffer: List[Tuple[int, int]] = []
    decision_records: List[dict] = []

    global_step = 0
    train_step = 0

    # --------------------------
    # cfg helpers (safe getattr)
    # --------------------------
    def _cfg_train_get(name: str, default):
        train_cfg = getattr(cfg, "train", None)
        if train_cfg is None:
            return default
        return getattr(train_cfg, name, default)

    # --------------------------
    # small helpers
    # --------------------------
    def _as_tensor(v, *, dtype=None) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            t = v
        else:
            t = torch.tensor(v)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t

    def _stack_to_device(ts: List[torch.Tensor], *, dtype=None) -> torch.Tensor:
        out = []
        for t in ts:
            out.append(_as_tensor(t, dtype=dtype).to(device))
        return torch.stack(out, dim=0)

    def _safe_float(x) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.detach().item())
        try:
            return float(x)
        except Exception:
            return 0.0

    def _safe_int(x) -> int:
        if isinstance(x, torch.Tensor):
            return int(x.detach().item())
        try:
            return int(x)
        except Exception:
            return -1

    def _dump_jsonl(records: list, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def wandb_log(_run, d: Dict[str, float], step: int):
        if _run is None:
            return
        try:
            _run.log(d, step=step)
        except Exception:
            _run.log(d)

    # --------------------------
    # decision diagnostics
    # --------------------------
    @torch.no_grad()
    def _append_decision_records(
        *,
        records: list,
        chunk_indices: list,          # list[int], length B
        a_hat: torch.Tensor,          # (B,)
        qb: torch.Tensor,             # (B,K)
        cb: torch.Tensor,             # (B,K)
        metas: list,                  # list[dict], length B
        cfg_utility,
    ):
        """
        Store per-sample decision diagnostics:
        idx, domain, token_len, trunc_ratio, is_truncated
        a_hat, a_q, a_r
        r_hat, r_q, r_r
        gap = r_r - r_q
        regret = r_r - r_hat
        """
        r_all = reward_per_action(qb, cb, cfg_utility)  # (B,K)

        a_r = torch.argmax(r_all, dim=1).long()
        a_q = torch.argmax(qb, dim=1).long()

        B = qb.size(0)
        arange = torch.arange(B, device=qb.device)

        r_hat = r_all[arange, a_hat]
        r_q = r_all[arange, a_q]
        r_r = r_all[arange, a_r]

        gap = r_r - r_q
        regret = r_r - r_hat

        a_hat_cpu = a_hat.detach().cpu().tolist()
        a_q_cpu = a_q.detach().cpu().tolist()
        a_r_cpu = a_r.detach().cpu().tolist()

        r_hat_cpu = r_hat.detach().cpu().tolist()
        r_q_cpu = r_q.detach().cpu().tolist()
        r_r_cpu = r_r.detach().cpu().tolist()

        gap_cpu = gap.detach().cpu().tolist()
        regret_cpu = regret.detach().cpu().tolist()

        for i in range(B):
            m = metas[i] if isinstance(metas[i], dict) else {}
            records.append({
                "idx": int(chunk_indices[i]),
                "domain": str(m.get("domain", "unknown")),
                "token_len": _safe_int(m.get("token_len", -1)),
                "trunc_ratio": _safe_float(m.get("trunc_ratio", 0.0)),
                "is_truncated": bool(m.get("is_truncated", False)),

                "a_hat": int(a_hat_cpu[i]),
                "a_q": int(a_q_cpu[i]),
                "a_r": int(a_r_cpu[i]),

                "r_hat": float(r_hat_cpu[i]),
                "r_q": float(r_q_cpu[i]),
                "r_r": float(r_r_cpu[i]),

                "gap_oracle_vs_maxq": float(gap_cpu[i]),
                "regret_oracle_vs_hat": float(regret_cpu[i]),
            })

    # --------------------------
    # action selection (policy or UCB, with optional gating)
    # --------------------------
    def _select_actions_batch(xb, xfb, db, qb, cb, metas) -> torch.Tensor:
        """
        xb:  (B, D_emb)
        xfb: (B, D_feat)
        db:  (B,)
        qb:  (B, K)
        cb:  (B, K)
        """
        B = xb.size(0)

        # 1) external policy
        if policy is not None:
            a_list = []
            for i in range(B):
                try:
                    ai = int(
                        policy.select_action(
                            x=xb[i],
                            x_feat=xfb[i],
                            domain_id=db[i],
                            quality=qb[i],
                            costs=cb[i],
                            meta=metas[i],
                        )
                    )
                except TypeError:
                    ai = int(
                        policy.select_action(
                            x=xb[i],
                            quality=qb[i],
                            costs=cb[i],
                            meta=metas[i],
                        )
                    )
                a_list.append(ai)
            return torch.tensor(a_list, device=device, dtype=torch.long)

        # 2) UCB branch
        if ucb is None:
            raise ValueError("ucb is None but policy is also None.")

        gate_enable = bool(_cfg_train_get("gate_enable", False))
        gate_tau = float(_cfg_train_get("gate_tau", 0.7))
        has_gate = hasattr(model, "gate_proba")

        if (not gate_enable) or (not has_gate):
            a_star, _, _ = ucb.select_action(model, xb, xfb, db)
            return a_star.long().to(device)

        # gating + UCB (no candidate restriction)
        with torch.no_grad():
            mean_scores = model.forward_all_actions(xb, xfb, db)   # (B,K)
            a_mean = torch.argmax(mean_scores, dim=1).long()

        with torch.no_grad():
            p = model.gate_proba(xb, xfb, db).view(-1)
            mask = (p >= gate_tau)

        a_out = a_mean.clone()

        if mask.any():
            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            xb2 = xb.index_select(0, idx)
            xf2 = xfb.index_select(0, idx)
            db2 = db.index_select(0, idx)

            a_star2, _, _ = ucb.select_action(model, xb2, xf2, db2)
            a_out[idx] = a_star2.long()

        # lightweight gate logging (no spam)
        if wandb_run is not None:
            wandb_log(
                wandb_run,
                {
                    "gate/route_ratio": float(mask.float().mean().item()),
                    "gate/p_mean": float(p.mean().item()),
                    "gate/tau": float(gate_tau),
                },
                step=global_step,
            )

        return a_out.long().to(device)

    # ==========================================
    # 1. Train one epoch on pool
    # ==========================================
    def _train_one_epoch_on_pool(pool_samples: List[Tuple[int, int]]) -> Dict[str, float]:
        nonlocal train_step, global_step

        if (not do_model_update) or (len(pool_samples) == 0):
            return {"train/huber": 0.0, "train/gate_bce": 0.0, "train/loss_total": 0.0}

        model.train()
        perm = np.random.permutation(len(pool_samples))
        total_reg = 0.0
        total_gate = 0.0
        total_all = 0.0
        n_batches = 0

        it = range(0, len(pool_samples), train_batch_size)
        if show_progress:
            it = tqdm(list(it), desc="train_epoch", leave=False)

        delta = float(getattr(cfg, "huber_delta", 1.0))

        # gate training config
        gate_train_enable = bool(_cfg_train_get("gate_train_enable", True))
        gate_loss_weight = float(_cfg_train_get("gate_loss_weight", 0.2))
        gate_gap_thr = float(_cfg_train_get("gate_gap_thr", 0.1))
        gate_pos_weight = float(_cfg_train_get("gate_pos_weight", 1.0))
        has_gate_logits = hasattr(model, "gate_logits")

        bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(gate_pos_weight, device=device, dtype=torch.float32)
        )

        for s in it:
            batch_ids = perm[s : s + train_batch_size]

            xs, xfs, dids, qs, cs = [], [], [], [], []
            a_list = []
            for bi in batch_ids:
                di, ai = pool_samples[int(bi)]
                x_emb, x_feat, domain_id, q, c, _m = dataset[int(di)]
                xs.append(x_emb)
                xfs.append(x_feat)
                dids.append(domain_id)
                qs.append(q)
                cs.append(c)
                a_list.append(int(ai))

            xb = _stack_to_device(xs, dtype=torch.float32)
            xfb = _stack_to_device(xfs, dtype=torch.float32)
            db = _stack_to_device(dids, dtype=torch.long).view(-1)
            qb = _stack_to_device(qs, dtype=torch.float32)
            cb = _stack_to_device(cs, dtype=torch.float32)
            a_int = torch.tensor(a_list, device=device, dtype=torch.long)

            r_t = reward_of_action(qb, cb, a_int, cfg.utility)

            # reward regression
            pred = model(xb, xfb, db, a_int)
            loss_reg = F.smooth_l1_loss(pred, r_t, beta=delta)

            # gate loss (optional)
            loss_gate = torch.zeros((), device=device, dtype=torch.float32)
            if gate_train_enable and has_gate_logits and gate_loss_weight > 0:
                with torch.no_grad():
                    r_all = reward_per_action(qb, cb, cfg.utility)  # (B,K)
                    a_r = torch.argmax(r_all, dim=1).long()
                    a_q = torch.argmax(qb, dim=1).long()
                    arange = torch.arange(qb.size(0), device=device)
                    gap = (r_all[arange, a_r] - r_all[arange, a_q])  # (B,)
                    y_high = (gap > gate_gap_thr).float()            # (B,)

                logits = model.gate_logits(xb, xfb, db).view(-1)     # (B,)
                loss_gate = bce(logits, y_high)

            loss = loss_reg + gate_loss_weight * loss_gate

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_clip = float(_cfg_train_get("grad_clip", 0.0) or 0.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_reg += float(loss_reg.item())
            total_gate += float(loss_gate.item())
            total_all += float(loss.item())
            n_batches += 1
            train_step += 1
            global_step += 1

            if log_every_steps > 0 and (train_step % log_every_steps == 0):
                dbg = {
                    "train/huber_loss_step": float(loss_reg.item()),
                    "train/gate_bce_step": float(loss_gate.item()),
                    "train/loss_total_step": float(loss.item()),
                }
                if hasattr(optimizer, "param_groups"):
                    dbg["train/lr"] = float(optimizer.param_groups[0].get("lr", 0.0))
                wandb_log(wandb_run, dbg, step=global_step)

        denom = max(1, n_batches)
        return {
            "train/huber": total_reg / denom,
            "train/gate_bce": total_gate / denom,
            "train/loss_total": total_all / denom,
        }

    # ==========================================
    # 2. Decide one slice + UCB updates + meters
    # ==========================================
    def _decide_one_slice(slice_idxs: np.ndarray, split_idx: int):
        nonlocal decided, replay_buffer, global_step

        meters.reset_split()
        model.eval()

        idxs = [int(x) for x in slice_idxs.tolist()]
        it = range(0, len(idxs), decision_batch_size)
        if show_progress:
            it = tqdm(list(it), desc=f"decide_slice_{split_idx}", leave=False)

        for s in it:
            chunk = idxs[s : s + decision_batch_size]

            xs, xfs, dids, qs, cs, metas = [], [], [], [], [], []
            for di in chunk:
                x_emb, x_feat, domain_id, q, c, m = dataset[int(di)]
                xs.append(x_emb)
                xfs.append(x_feat)
                dids.append(domain_id)
                qs.append(q)
                cs.append(c)
                metas.append(m)

            xb = _stack_to_device(xs, dtype=torch.float32)
            xfb = _stack_to_device(xfs, dtype=torch.float32)
            db = _stack_to_device(dids, dtype=torch.long).view(-1)
            qb = _stack_to_device(qs, dtype=torch.float32)
            cb = _stack_to_device(cs, dtype=torch.float32)

            a_int = _select_actions_batch(xb, xfb, db, qb, cb, metas)
            B = xb.size(0)

            _append_decision_records(
                records=decision_records,
                chunk_indices=chunk,
                a_hat=a_int,
                qb=qb,
                cb=cb,
                metas=metas,
                cfg_utility=cfg.utility,
            )

            if do_ucb_update and (ucb is not None):
                with torch.no_grad():
                    h_pre = model.get_last_hidden(xb, xfb, db, a_int)
                    for i in range(B):
                        ucb.update(int(a_int[i].item()), h_pre[i])

            for i in range(B):
                decided.append((int(chunk[i]), int(a_int[i].item())))

            if buffer_size > 0:
                replay_buffer.extend([(int(chunk[i]), int(a_int[i].item())) for i in range(B)])
                if len(replay_buffer) > buffer_size:
                    replay_buffer = replay_buffer[-buffer_size:]

            r_batch = reward_of_action(qb, cb, a_int, cfg.utility)
            g_batch = _compute_gap_for_action(qb, cb, a_int, cfg.utility)

            for i in range(B):
                ai_scalar = int(a_int[i].item())
                meters.update(
                    reward=float(r_batch[i].item()),
                    gap=float(g_batch[i].item()),
                    quality=float(qb[i, ai_scalar].item()),
                    cost=float(cb[i, ai_scalar].item()),
                    action=ai_scalar,
                    domain=str(metas[i].get("domain", "unknown")),
                    loss=0.0,
                    ignore_cumulative=(split_idx < burn_in_splits),
                )
                global_step += 1

        return meters.split_summary(split_idx)

    # ==========================================
    # Main slice loop
    # ==========================================
    outer = range(len(split_indices))
    if show_progress:
        outer = tqdm(list(outer), desc="Slices", leave=True)

    for t in outer:
        if train_pool == "slice_only":
            if t == 0:
                train_samples = []
            else:
                prev_idxs = set(int(x) for x in split_indices[t - 1].tolist())
                train_samples = [(di, ai) for (di, ai) in decided if di in prev_idxs]
        else:
            train_samples = decided

        if do_model_update and epochs_per_slice > 0 and len(train_samples) > 0:
            for _ep in range(int(epochs_per_slice)):
                stats = _train_one_epoch_on_pool(train_samples)
                if wandb_run is not None:
                    wandb_log(
                        wandb_run,
                        {
                            "slice_train/huber_ep": float(stats["train/huber"]),
                            "slice_train/gate_bce_ep": float(stats["train/gate_bce"]),
                            "slice_train/loss_total_ep": float(stats["train/loss_total"]),
                        },
                        step=global_step,
                    )

        if (
            do_ucb_update
            and (ucb is not None)
            and (len(replay_buffer) > 0)
            and (rebuild_per_slice > 0)
            and (t > 0)
            and (t % rebuild_per_slice == 0)
        ):
            was_training = model.training
            model.eval()
            with torch.no_grad():
                ucb.reset_A()
                ucb.rebuild_from_buffer(
                    model=model,
                    dataset=dataset,
                    buffer=replay_buffer,
                    device_for_model=device,
                    batch_size=rebuild_batch_size,
                    verbose=True,
                )
            if was_training and do_model_update:
                model.train()

        summary = _decide_one_slice(split_indices[t], t)
        if wandb_run is not None:
            wandb_log(wandb_run, summary, step=global_step)

        if save_each_split:
            save_checkpoint(
                save_dir=save_dir,
                split_idx=t,
                model=model,
                optimizer=optimizer if do_model_update else None,
                scheduler=scheduler if do_model_update else None,
                ucb=ucb if do_ucb_update else None,
                extra={
                    "global_step": int(global_step),
                    "epochs_per_slice": int(epochs_per_slice),
                    "train_pool": str(train_pool),
                },
            )

        if show_progress and hasattr(outer, "set_postfix"):
            outer.set_postfix(
                {
                    "avg_r": f"{summary.get('split/avg_reward', 0.0):.4f}",
                    "pool_n": str(len(decided)),
                }
            )

    dump_path = os.path.join(save_dir, "decision_diagnostics.jsonl")
    _dump_jsonl(decision_records, dump_path)
    print(f"[diagnostics] wrote {len(decision_records)} records to {dump_path}")

    if wandb_run is not None:
        wandb_run.finish()

    return meters