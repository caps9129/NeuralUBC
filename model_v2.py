from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class UtilityNet(nn.Module):
    """
    Predict utility f_theta(x, a) with:
      - shared encoder phi(x) (does NOT depend on action)
      - per-action head producing (B, K)
    """

    def __init__(
        self,
        x_dim: int,
        num_actions: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
        use_layernorm: bool = False,
        out_activation: str = "none",  # "none" or "sigmoid"
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.num_actions = int(num_actions)
        self.out_activation = out_activation

        # Shared encoder: phi(x)
        in_dim = self.x_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        # Per-action head packed into one Linear: (H -> K)
        self.head = nn.Linear(in_dim, self.num_actions)

        # for last-layer UCB approximation later (optional)
        self._last_hidden_dim = in_dim

    def _apply_out_activation(self, y: torch.Tensor) -> torch.Tensor:
        if self.out_activation == "sigmoid":
            return torch.sigmoid(y)
        elif self.out_activation == "none":
            return y
        else:
            raise ValueError(f"Unknown out_activation={self.out_activation}")

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        a: (B,) long
        returns: (B,) predicted utility for chosen action a
        """
        if x.dim() != 2:
            raise ValueError(f"x must be (B,D), got {tuple(x.shape)}")
        if a.dim() != 1:
            raise ValueError(f"a must be (B,), got {tuple(a.shape)}")

        a = a.long()
        h = self.mlp(x)              # (B, H)
        y_all = self.head(h)         # (B, K)
        y_all = self._apply_out_activation(y_all)

        # pick y for each sample's chosen action
        y = y_all.gather(1, a.view(-1, 1)).squeeze(1)  # (B,)
        return y

    @torch.no_grad()
    def forward_all_actions(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        returns: (B, K) predicted utilities for all actions
        """
        if x.dim() != 2:
            raise ValueError(f"x must be (B,D), got {tuple(x.shape)}")

        h = self.mlp(x)              # (B, H)
        y = self.head(h)             # (B, K)
        y = self._apply_out_activation(y)
        return y

    def get_last_hidden(self, x: torch.Tensor, a: torch.Tensor = None) -> torch.Tensor:
        """
        Return last hidden representation before head (for last-layer UCB).
        Kept signature compatible; action 'a' is ignored in scheme-2.
        x: (B,D)
        returns: (B,H)
        """
        if x.dim() != 2:
            raise ValueError(f"x must be (B,D), got {tuple(x.shape)}")
        return self.mlp(x)


def build_model_from_cfg(cfg) -> UtilityNet:
    model = UtilityNet(
        x_dim=cfg.net.x_dim,
        num_actions=getattr(cfg.net, "num_actions", 11),
        hidden_sizes=tuple(cfg.net.hidden_sizes),
        dropout=cfg.net.dropout,
        use_layernorm=cfg.net.use_layernorm,
        out_activation=cfg.net.out_activation,
    )
    return model