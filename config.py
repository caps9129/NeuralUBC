# config.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Literal

@dataclass
class DataConfig:
    # RouterBench source
    repo_id: str = "withmartian/routerbench"
    pkl_name: str = "routerbench_raw.pkl"     # or routerbench_0shot.pkl / routerbench_5shot.pkl

    # replay setting
    num_splits: int = 20
    shuffle: bool = False
    seed: int = 42

    # Column mapping
    sample_id_col: str = "sample_id"
    prompt_col: str = "prompt"
    domain_col: str = "eval_name"             # domain/task name
    model_col: str = "model_name"
    response_col: str = "model_response"
    quality_col: str = "performance"
    cost_col: str = "cost"

    expected_num_models: int = 11

    # Domain / subset selection
    # domains_allowlist: List[str] = field(default_factory=lambda: ["hellaswag", "grade-school-math", "mbpp", "consensus_summary", "mmlu-high-school-biology", "mmlu-college-biology"])
    domains_allowlist: List[str] = None
    sample_id_prefix_allowlist: List[str] = field(default_factory=list) # match df[sample_id_col].startswith(prefix)

    # Text encoder embedding
    text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    x_dim: int = 384                       # all-MiniLM-L6-v2 embedding dim
    max_length: int = 256
    batch_size_encode: int = 128

@dataclass
class ModelPoolConfig:
    model_names: List[str] = field(default_factory=lambda: ["small", "medium", "large"])
    # cost could be dollars per 1k tokens, latency, etc. You pick one consistent unit
    costs: List[float] = field(default_factory=lambda: [1.0, 3.0, 10.0])

    def num_actions(self) -> int:
        return len(self.model_names)


@dataclass
class NetConfig:
    # f_theta(x, a) network
    x_dim: int = 768
    action_embed_dim: int = 32
    hidden_sizes: Tuple[int, ...] = (256, 256)
    dropout: float = 0.0
    use_layernorm: bool = False
    out_activation: str = "none"          # "none" or "sigmoid"

@dataclass
class UCBConfig:
    beta: float = 1.0
    # Which uncertainty approximation you want
    ucb_mode: str = "last_layer"          # "last_layer" (fast) or "full_grad" (later)
    lambda_: float = 1.0                   # regularization for ridge regression (equiv to A matrix init scale)
    update_A_every: int = 1               # update uncertainty stats every N steps
    eps: float = 1e-8                    # numerical stability for sqrt
    debug_every: int = 10               # print debug info every N steps (0 means no debug prints)
    buffer_size: int = 40000                # for batched replay, how many past samples to keep in buffer for UCB updates
    rebuild_every: int = 1000               # for batched replay, how often to rebuild A_inv from buffer (0 means never rebuild, just do incremental updates)
    rebuild_batch_size: int = 256           # for UCB rebuild, how many samples to process at once

@dataclass
class UtilityConfig:
    # reward: r = q * exp(-lam * c_tilde)
    lam: float = 5.0

    # cost normalization choice
    CostNormMode = Literal["linear_clip", "log_norm", "per_sample_minmax", "none"]
    cost_norm: CostNormMode = "linear_clip"
    reward_mode: str = "log_reward"
    # softmax_T: float = 0.2

    # use quantiles to set tau and C_max automatically from observed costs, or set manually if you prefer
    p: float = 0.99

    # linear clipping: c_tilde = min(c / tau, 1)
    tau: Optional[float] = None  # if None, can be auto-fit

    # log norm: c_tilde = log(1+c) / log(1+C_max)
    C_max: Optional[float] = None  # if None, can be auto-fit

    eps: float = 1e-8

    clamp_cost_01: bool = True
    clamp_reward_01: bool = True

@dataclass
class TrainConfig:
    device: str = "cuda"
    lr: float = 3e-4
    weight_decay: float = 0.01

    # Online update style
    batch_size_round: int = 128             # 1 means true online, >1 means batched replay
    grad_steps_per_round: int = 1         # how many optimizer steps per round
    max_rounds: Optional[int] = None      # truncate for debug
    eval_every_rounds: int = 200
    epochs_per_slice: int = 20

    # Logging
    log_every_rounds: int = 50

@dataclass
class WandbConfig:
    enable: bool = True
    project: str = "AI543_proposal"
    name: str = "neuralucb_routerbench"
    tags: List[str] = field(default_factory=lambda: ["offline_replay", "partial_feedback"])
    notes: str = ""
    group: Optional[str] = None

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    pool: ModelPoolConfig = field(default_factory=ModelPoolConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    net: NetConfig = field(default_factory=NetConfig)
    ucb: UCBConfig = field(default_factory=UCBConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def post_init_fixup(self):
        # keep dims consistent if user only sets data.x_dim
        if self.net.x_dim != self.data.x_dim:
            self.net.x_dim = self.data.x_dim
        return self