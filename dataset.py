# dataset.py
import os
import hashlib
import pickle
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from huggingface_hub import hf_hub_download


def _unwrap_text(v):
    """RouterBench raw.pkl fields like prompt/model_response sometimes store [str]."""
    if isinstance(v, list) and len(v) > 0:
        return v[0]
    return v


def load_routerbench_df(repo_id: str, pkl_name: str) -> pd.DataFrame:
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=pkl_name)
    with open(path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {pkl_name}, got {type(df)}")
    return df


def filter_df(
    df: pd.DataFrame,
    domain_col: str,
    sample_id_col: str,
    domains_allowlist: Optional[List[str]] = None,
    sample_id_prefix_allowlist: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = df
    if domains_allowlist:
        out = out.loc[out[domain_col].isin(domains_allowlist)]
    if sample_id_prefix_allowlist:
        prefixes = tuple(sample_id_prefix_allowlist)
        out = out.loc[out[sample_id_col].astype(str).str.startswith(prefixes)]
    return out


def get_model_list(df: pd.DataFrame, model_col: str) -> List[str]:
    """Deterministic model order."""
    models = sorted(df[model_col].dropna().unique().tolist())
    return models


def build_wide_table(
    df: pd.DataFrame,
    sample_id_col: str,
    prompt_col: str,
    domain_col: str,
    model_col: str,
    quality_col: str,
    cost_col: str,
    response_col: Optional[str] = None,
    expected_num_models: Optional[int] = None,
    fixed_model_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert long table rows (sample_id, model_name, performance, cost, ...) into one row per sample_id:
      - quality: np.ndarray shape [K]
      - costs: np.ndarray shape [K]
      - model_names: List[str] length K
    """
    if fixed_model_order is None:
        fixed_model_order = get_model_list(df, model_col)
    K = len(fixed_model_order)

    if expected_num_models is not None and K != expected_num_models:
        # Not fatal, but warn via exception for now to keep you safe
        raise ValueError(f"Detected K={K} models, expected {expected_num_models}. models={fixed_model_order}")

    model_to_idx = {m: i for i, m in enumerate(fixed_model_order)}

    rows = []
    grouped = df.groupby(sample_id_col, sort=False)

    for sid, g in grouped:
        # sanity: should have exactly K model rows for each sample
        uniq_models = g[model_col].nunique()
        if uniq_models != K:
            # skip incomplete samples to keep shapes consistent
            continue

        prompt_val = _unwrap_text(g.iloc[0][prompt_col])
        domain_val = g.iloc[0][domain_col]

        quality = np.zeros((K,), dtype=np.float32)
        costs = np.zeros((K,), dtype=np.float32)

        # optional: keep response text per model
        responses = None
        if response_col is not None:
            responses = [None] * K

        ok = True
        for _, r in g.iterrows():
            m = r[model_col]
            if m not in model_to_idx:
                ok = False
                break
            j = model_to_idx[m]
            quality[j] = float(r[quality_col])
            costs[j] = float(r[cost_col])
            if responses is not None:
                responses[j] = _unwrap_text(r[response_col])

        if not ok:
            continue

        row = {
            "sample_id": sid,
            "domain": domain_val,
            "prompt": prompt_val,
            "model_names": fixed_model_order,
            "quality": quality,
            "costs": costs,
        }
        if responses is not None:
            row["responses"] = responses

        rows.append(row)

    wide = pd.DataFrame(rows)
    return wide


def _hash_config_for_cache(d: Dict) -> str:
    s = repr(sorted(d.items()))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def encode_prompts_sentence_transformers(
    prompts: List[str],
    encoder_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    """
    Returns embeddings: np.ndarray [N, D]
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder_name, device=device)
    # sentence-transformers handles truncation via max_seq_length
    model.max_seq_length = max_length
    embs = model.encode(
        prompts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embs = embs.astype(np.float32)
    return embs


def build_or_load_embeddings(
    wide_df: pd.DataFrame,
    cache_dir: str,
    encoder_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    extra_cache_key: Optional[Dict] = None,
) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)

    key_dict = {
        "encoder": encoder_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "n": len(wide_df),
    }
    if extra_cache_key:
        key_dict.update(extra_cache_key)

    cache_key = _hash_config_for_cache(key_dict)
    cache_path = os.path.join(cache_dir, f"emb_{cache_key}.npy")

    if os.path.exists(cache_path):
        embs = np.load(cache_path)
        return embs

    prompts = wide_df["prompt"].astype(str).tolist()
    embs = encode_prompts_sentence_transformers(
        prompts=prompts,
        encoder_name=encoder_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    np.save(cache_path, embs)
    return embs


def build_or_load_lengths(
    wide_df: pd.DataFrame,
    cache_dir: str,
    encoder_name: str,
    batch_size: int,
    max_length: int,
    extra_cache_key: Optional[Dict] = None,
) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)

    key_dict = {
        "kind": "lengths",
        "encoder": encoder_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "n": len(wide_df),
    }
    if extra_cache_key:
        key_dict.update(extra_cache_key)

    cache_key = _hash_config_for_cache(key_dict)
    cache_path = os.path.join(cache_dir, f"len_{cache_key}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    prompts = wide_df["prompt"].astype(str).tolist()
    lens = compute_prompt_token_lens(
        prompts=prompts,
        encoder_name=encoder_name,
        batch_size=batch_size,
        max_length=max_length,
    )
    np.save(cache_path, lens)
    return lens


def split_consecutive(indices: np.ndarray, num_splits: int, domains: np.ndarray, seed: int = 42) -> List[np.ndarray]:
    """
    Stratified split by domain (kept function name for compatibility).

    indices: np.ndarray [N], usually np.arange(N)
    domains: np.ndarray [N], domain label per sample (string or int)
    returns: List[np.ndarray] length=num_splits, each is indices for that split
    """
    from sklearn.model_selection import StratifiedKFold

    if len(indices) != len(domains):
        raise ValueError(f"indices and domains must have same length, got {len(indices)} vs {len(domains)}")

    domains = domains.astype(str)

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=int(seed))

    splits: List[np.ndarray] = []
    for _, test_idx in skf.split(indices, domains):
        splits.append(indices[test_idx])

    return splits


class RouterBenchWideDataset(Dataset):
    """
    Each item:
      x: FloatTensor [D]
      quality: FloatTensor [K]
      costs: FloatTensor [K]
      meta: dict (sample_id, domain, prompt, model_names)
    """

    def __init__(
        self,
        wide_df: pd.DataFrame,
        embeddings: np.ndarray,
        keep_meta: bool = True,
        return_features: bool = False,
        max_length: int = 512,
    ):
        assert len(wide_df) == embeddings.shape[0]
        self.df = wide_df.reset_index(drop=True)
        self.emb = embeddings
        self.keep_meta = keep_meta
        self.return_features = bool(return_features)
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = torch.from_numpy(self.emb[idx]).float()
        quality = torch.from_numpy(row["quality"]).float()
        costs = torch.from_numpy(row["costs"]).float()

        if self.keep_meta:
            meta = {
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "prompt": row["prompt"],
                "model_names": row["model_names"],
            }
        else:
            meta = {}

        if not self.return_features:
            return x, quality, costs, meta

        # --- structural features ---
        token_len = int(row.get("token_len", 0))
        domain_id = int(row.get("domain_id", -1))

        log_len = float(np.log1p(max(token_len, 0)))
        trunc_ratio = float(min(1.0, self.max_length / max(1, token_len)))
        is_truncated = float(token_len > self.max_length)

        x_feat = torch.tensor([log_len, trunc_ratio, is_truncated], dtype=torch.float32)
        domain_id_t = torch.tensor(domain_id, dtype=torch.long)

        # also put them into meta for debugging
        meta["token_len"] = token_len
        meta["domain_id"] = domain_id
        meta["trunc_ratio"] = trunc_ratio
        meta["is_truncated"] = bool(is_truncated)

        return x, x_feat, domain_id_t, quality, costs, meta

def compute_prompt_token_lens(
    prompts: List[str],
    encoder_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """
    Return token lengths without truncation (true length).
    Uses HF tokenizer so it works for many encoder names.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    lens = []

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tok(
            chunk,
            add_special_tokens=True,
            truncation=False,   # IMPORTANT: true length
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # enc["input_ids"] is List[List[int]]
        lens.extend([len(ids) for ids in enc["input_ids"]])

    return np.asarray(lens, dtype=np.int32)


def build_routerbench_datasets(
    cfg,
    device: str = "cuda",
    cache_dir: str = "./cache_routerbench",
    keep_responses: bool = False,
    return_features: bool = False,
) -> Tuple[RouterBenchWideDataset, List[np.ndarray], List[str]]:
    """
    Returns:
      dataset: RouterBenchWideDataset over ALL filtered samples (one row per sample_id)
      split_indices: list of np.ndarray, consecutive splits indices for replay
      model_names: fixed model order (length K)
    """
    df = load_routerbench_df(cfg.data.repo_id, cfg.data.pkl_name)

    df = filter_df(
        df,
        domain_col=cfg.data.domain_col,
        sample_id_col=cfg.data.sample_id_col,
        domains_allowlist=cfg.data.domains_allowlist,
        sample_id_prefix_allowlist=cfg.data.sample_id_prefix_allowlist,
    )

    model_names = get_model_list(df, cfg.data.model_col)

    wide = build_wide_table(
        df=df,
        sample_id_col=cfg.data.sample_id_col,
        prompt_col=cfg.data.prompt_col,
        domain_col=cfg.data.domain_col,
        model_col=cfg.data.model_col,
        quality_col=cfg.data.quality_col,
        cost_col=cfg.data.cost_col,
        response_col=(cfg.data.response_col if keep_responses else None),
        expected_num_models=cfg.data.expected_num_models,
        fixed_model_order=model_names,
    )

    domain_list = sorted(wide["domain"].astype(str).unique().tolist())
    domain2id = {d: i for i, d in enumerate(domain_list)}
    wide["domain_id"] = wide["domain"].astype(str).map(domain2id).astype(np.int32)

    # Build embeddings with caching
    extra_cache_key = {
        "domains": ",".join(cfg.data.domains_allowlist) if cfg.data.domains_allowlist else "ALL",
        "pkl": cfg.data.pkl_name,
    }

    token_lens = build_or_load_lengths(
        wide_df=wide,
        cache_dir=cache_dir,
        encoder_name=cfg.data.text_encoder_name,
        batch_size=cfg.data.batch_size_encode,
        max_length=cfg.data.max_length,
        extra_cache_key=extra_cache_key,
    )
    wide["token_len"] = token_lens
    wide["char_len"] = wide["prompt"].astype(str).map(len).astype(np.int32)

    embs = build_or_load_embeddings(
        wide_df=wide,
        cache_dir=cache_dir,
        encoder_name=cfg.data.text_encoder_name,
        batch_size=cfg.data.batch_size_encode,
        max_length=cfg.data.max_length,
        device=device,
        extra_cache_key=extra_cache_key,
    )

    dataset = RouterBenchWideDataset(wide_df=wide, embeddings=embs, keep_meta=True, return_features=return_features, max_length=cfg.data.max_length)

    indices = np.arange(len(dataset))

    domains = dataset.df["domain"].astype(str).values
    split_indices = split_consecutive(
        indices,
        cfg.data.num_splits,
        domains=domains,
        seed=cfg.data.seed,
    )

    return dataset, split_indices, model_names