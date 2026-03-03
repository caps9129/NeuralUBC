from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
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


def train_offline_replay(
    *,
    cfg,
    dataset,
    split_indices: List[np.ndarray],
    model: nn.Module,
    ucb=None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    policy=None,
    do_model_update: bool = True,
    do_ucb_update: bool = True,
    device: str = "cuda",
    wandb_run=None,
    log_every: int = 50,
    save_each_split: bool = False,
    save_dir: str = "./checkpoints",
    show_progress: bool = False,
) -> OnlineMeters:
    """
    Offline replay loop with optional round-batching.

    Fixes included:
      - UCB update uses h_pre computed BEFORE optimizer.step (decision-time representation).
      - h_pre is computed under eval() + no_grad() to avoid dropout noise.
      - Model update can be batched using cfg.train.batch_size_round.
      - Action selection is per-sample, but computed as a batch.
      - Optional: periodic recomputation of UCB covariance using a sliding buffer.

    Simple debug included:
      - every cfg.ucb.trainer_debug_every steps:
          prints avg_reward_flipped vs avg_reward_not_flipped
        where "flipped" means (a_ucb != a_mean), and reward is computed on chosen action.

    Required UCB APIs if periodic recomputation is enabled:
      - ucb.reset_A()
      - ucb.rebuild_from_buffer(model, dataset, buffer, device_for_model, batch_size=...)
    """
    model.to(device)

    # set initial mode
    if do_model_update:
        model.train()
    else:
        model.eval()

    meters = OnlineMeters(
        model_names=getattr(dataset, "model_names", None) or dataset.df.iloc[0]["model_names"]
    )

    global_step = 0

    batch_size_round = int(getattr(cfg.train, "batch_size_round", 1))
    if batch_size_round < 1:
        batch_size_round = 1

    # -----------------------------
    # periodic recomputation config
    # -----------------------------
    ucb_cfg = getattr(cfg, "ucb", None)
    buffer_size = int(getattr(ucb_cfg, "buffer_size", 0) or 0)
    rebuild_every = int(getattr(ucb_cfg, "rebuild_every", 0) or 0)
    rebuild_batch_size = int(getattr(ucb_cfg, "rebuild_batch_size", 256) or 256)

    # -----------------------------
    # simple trainer debug config
    # -----------------------------
    # trainer_debug_every = int(getattr(ucb_cfg, "trainer_debug_every", 0) or 0)
    trainer_debug_every = 5

    # store (data_idx, a_int) for last N samples
    replay_buffer: List[Tuple[int, int]] = []

    # helper: maybe rebuild A
    def _maybe_rebuild_A():
        nonlocal replay_buffer
        if not (do_ucb_update and (ucb is not None)):
            return
        if rebuild_every <= 0 or buffer_size <= 0:
            return
        if global_step <= 0:
            return
        if (global_step % rebuild_every) != 0:
            return
        if len(replay_buffer) == 0:
            return

        was_training = model.training
        model.eval()
        with torch.no_grad():
            if not hasattr(ucb, "reset_A") or not hasattr(ucb, "rebuild_from_buffer"):
                raise AttributeError(
                    "UCB periodic recomputation requires ucb.reset_A() and ucb.rebuild_from_buffer()."
                )

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

        if show_progress:
            print(f"[UCB] rebuilt A at global_step={global_step} using buffer_n={len(replay_buffer)}")

    split_iter = enumerate(split_indices)
    if show_progress:
        split_iter = tqdm(split_iter, total=len(split_indices), desc="Splits", leave=True)

    for split_idx, idxs in split_iter:
        meters.reset_split()

        if show_progress:
            step_iter = tqdm(
                range(0, len(idxs), batch_size_round),
                total=(len(idxs) + batch_size_round - 1) // batch_size_round,
                desc=f"Split {split_idx}",
                leave=False,
            )
        else:
            step_iter = range(0, len(idxs), batch_size_round)

        for start in step_iter:
            end = min(start + batch_size_round, len(idxs))
            chunk = idxs[start:end]

            # ---- build a round batch ----
            xs, qs, cs, metas = [], [], [], []
            for data_idx in chunk:
                x, quality, costs, meta = dataset[int(data_idx)]
                xs.append(x)
                qs.append(quality)
                cs.append(costs)
                metas.append(meta)

            xb = torch.stack([t.to(device) for t in xs], dim=0)  # (B,D)
            qb = torch.stack([t.to(device) for t in qs], dim=0)  # (B,K)
            cb = torch.stack([t.to(device) for t in cs], dim=0)  # (B,K)
            B = xb.size(0)

            # ---- 1) choose actions ----
            if policy is not None:
                a_list = []
                for i in range(B):
                    ai = int(
                        policy.select_action(
                            x=xb[i],
                            quality=qb[i],
                            costs=cb[i],
                            meta=metas[i],
                        )
                    )
                    a_list.append(ai)
                a_int = torch.tensor(a_list, device=device, dtype=torch.long)  # (B,)
            else:
                if ucb is None:
                    raise ValueError("policy is None but ucb is None. Provide either policy or ucb.")
                # ucb.select_action sets model.eval() internally
                a_star, _, _ = ucb.select_action(model, xb)
                a_int = a_star.long().to(device)  # (B,)

            # ---- add chosen (data_idx, action) to replay buffer (for periodic recomputation) ----
            if buffer_size > 0:
                replay_buffer.extend([(int(chunk[i]), int(a_int[i].item())) for i in range(B)])
                if len(replay_buffer) > buffer_size:
                    replay_buffer = replay_buffer[-buffer_size:]

            # ---- 1.5) compute decision-time features h_pre for UCB update (BEFORE optimizer.step) ----
            h_pre = None
            if do_ucb_update and (ucb is not None):
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    h_pre = model.get_last_hidden(xb, a_int)
                    if h_pre.dim() == 3 and h_pre.size(1) == 1:
                        h_pre = h_pre.squeeze(1)
                    elif h_pre.dim() == 1:
                        h_pre = h_pre.unsqueeze(0)
                if was_training and do_model_update:
                    model.train()

            # ---- 2) reward + gap + (simple) flipped debug stats ----
            do_dbg = (
                trainer_debug_every > 0
                and (global_step > 0)
                and ((global_step % trainer_debug_every) == 0)
                and (policy is None)
                and (ucb is not None)
            )

            a_mean = None
            if do_dbg:
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    mean_scores_dbg = model.forward_all_actions(xb)  # (B,K)
                    a_mean = torch.argmax(mean_scores_dbg, dim=1).long()
                if was_training and do_model_update:
                    model.train()

            r_list, gap_list, q_sel_list, c_sel_list = [], [], [], []
            flip_sum, flip_n = 0.0, 0
            noflip_sum, noflip_n = 0.0, 0

            for i in range(B):
                ai_scalar = torch.tensor(int(a_int[i].item()), device=device, dtype=torch.long)
                r_i = reward_of_action(qb[i], cb[i], ai_scalar, cfg.utility)
                g_i = _compute_gap_for_action(qb[i], cb[i], ai_scalar, cfg.utility)

                r_list.append(r_i)
                gap_list.append(g_i)
                q_sel_list.append(float(qb[i, int(ai_scalar.item())].item()))
                c_sel_list.append(float(cb[i, int(ai_scalar.item())].item()))

                if do_dbg and (a_mean is not None):
                    if int(a_int[i].item()) != int(a_mean[i].item()):
                        flip_sum += float(r_i.item())
                        flip_n += 1
                    else:
                        noflip_sum += float(r_i.item())
                        noflip_n += 1


            if do_dbg:
                avg_flip = flip_sum / flip_n if flip_n > 0 else 0.0
                avg_noflip = noflip_sum / noflip_n if noflip_n > 0 else 0.0
                

                flip_str = f"{avg_flip:.4f}" if flip_n > 0 else "N/A"
                noflip_str = f"{avg_noflip:.4f}" if noflip_n > 0 else "N/A"

                print(
                    f"[trainer dbg step={global_step}] "
                    f"flip_n={flip_n} avg_reward_flipped={flip_str} | "
                    f"noflip_n={noflip_n} avg_reward_not_flipped={noflip_str}"
                )

            r_t = torch.stack(r_list, dim=0)       # (B,)
            gap_t = torch.stack(gap_list, dim=0)   # (B,)

            # ---- 3) optional model update (batched MSE) ----
            loss_val = 0.0
            if do_model_update:
                if optimizer is None:
                    raise ValueError("do_model_update=True but optimizer is None.")

                model.train()
                optimizer.zero_grad(set_to_none=True)

                pred = model(xb, a_int)
                if pred.dim() == 2 and pred.size(1) == 1:
                    pred = pred.squeeze(1)
                elif pred.dim() == 0:
                    pred = pred.view(1)

                loss = (pred - r_t).pow(2).mean()
                loss.backward()

                if getattr(cfg.train, "grad_clip", 0.0) and cfg.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.train.grad_clip))

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                loss_val = float(loss.item())

            # ---- 4) optional UCB update uses h_pre (decision-time feature) ----
            if do_ucb_update and (ucb is not None):
                if h_pre is None:
                    raise RuntimeError("Internal error: h_pre is None but do_ucb_update=True.")
                with torch.no_grad():
                    for i in range(B):
                        ucb.update(int(a_int[i].item()), h_pre[i])

            # ---- 5) meters update ----
            for i in range(B):
                meters.update(
                    reward=float(r_t[i].item()),
                    gap=float(gap_t[i].item()),
                    quality=q_sel_list[i],
                    cost=c_sel_list[i],
                    action=int(a_int[i].item()),
                    domain=str(metas[i].get("domain", "unknown")),
                    loss=loss_val,
                )
                global_step += 1

                if log_every > 0 and (global_step % log_every == 0):
                    dbg = meters.running_debug()
                    if do_model_update and optimizer is not None and hasattr(optimizer, "param_groups"):
                        dbg["train/lr"] = float(optimizer.param_groups[0].get("lr", 0.0))
                    wandb_log(wandb_run, dbg, step=global_step)

                _maybe_rebuild_A()

            # tqdm postfix (once per round-batch)
            if show_progress and hasattr(step_iter, "set_postfix"):
                dbg = meters.running_debug()
                postfix = {}
                if "train/reward_running_mean" in dbg:
                    postfix["r_mean"] = f"{dbg['train/reward_running_mean']:.4f}"
                if "train/mse_loss_running_mean" in dbg:
                    postfix["mse"] = f"{dbg['train/mse_loss_running_mean']:.4f}"
                if do_model_update and optimizer is not None and hasattr(optimizer, "param_groups"):
                    postfix["lr"] = f"{optimizer.param_groups[0].get('lr', 0.0):.2e}"
                if postfix:
                    step_iter.set_postfix(postfix)

        # end split summary
        summary = meters.split_summary(split_idx)
        wandb_log(wandb_run, summary, step=global_step)

        if save_each_split:
            save_checkpoint(
                save_dir=save_dir,
                split_idx=split_idx,
                model=model,
                optimizer=optimizer if do_model_update else None,
                scheduler=scheduler if do_model_update else None,
                ucb=ucb if do_ucb_update else None,
                extra={
                    "global_step": global_step,
                    "cfg": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else None,
                },
            )

        if show_progress and hasattr(split_iter, "set_postfix"):
            split_iter.set_postfix(
                {
                    "avg_r": f"{summary.get('split/avg_reward', 0.0):.4f}",
                    "avg_gap": f"{summary.get('split/avg_gap', 0.0):.4f}",
                }
            )

    if wandb_run is not None:
        wandb_run.finish()

    return meters