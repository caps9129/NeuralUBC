from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UtilityNet(nn.Module):
    """
    Predict utility f_theta(x, a).

    New input API (aligned to new dataset):
      x_emb:    (B, D_emb)    semantic embedding
      x_feat:   (B, D_feat)   small structured features (len, trunc_ratio, etc.)
      domain_id:(B,) long     domain categorical id
      a:        (B,) long     action index in [0, K-1]

    Output:
      y: (B,) predicted reward/utility (regression target)

    Additional (gate):
      predict p_highgap = P(gap_oracle_vs_maxq > gap_threshold | x)
      gate logits: (B,)
    """

    def __init__(
        self,
        x_emb_dim: int,
        x_feat_dim: int,
        num_domains: int,
        num_actions: int,
        action_embed_dim: int = 32,
        domain_embed_dim: int = 16,
        # two-tower compression sizes
        emb_hidden: Tuple[int, ...] = (256,),
        feat_hidden: Tuple[int, ...] = (32,),
        # final MLP
        hidden_sizes: Tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
        use_layernorm: bool = False,
        out_activation: str = "none",   # "none" or "sigmoid"
        # gate head
        gate_hidden: Tuple[int, ...] = (64,),
        gate_detach_context: bool = True,
    ):
        super().__init__()

        self.x_emb_dim = int(x_emb_dim)
        self.x_feat_dim = int(x_feat_dim)
        self.num_domains = int(num_domains)
        self.num_actions = int(num_actions)
        self.action_embed_dim = int(action_embed_dim)
        self.domain_embed_dim = int(domain_embed_dim)
        self.out_activation = str(out_activation)

        # gate config
        self.gate_detach_context = bool(gate_detach_context)

        # Embeddings
        self.action_emb = nn.Embedding(self.num_actions, self.action_embed_dim)
        self.domain_emb = nn.Embedding(self.num_domains, self.domain_embed_dim)

        # ---------
        # Tower 1: semantic embedding -> compressed
        # ---------
        self.emb_mlp, emb_out_dim = self._make_mlp(
            in_dim=self.x_emb_dim,
            hidden_sizes=tuple(emb_hidden),
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

        # ---------
        # Tower 2: structured features (+ domain_emb) -> compressed
        # ---------
        feat_in_dim = self.x_feat_dim + self.domain_embed_dim
        self.feat_mlp, feat_out_dim = self._make_mlp(
            in_dim=feat_in_dim,
            hidden_sizes=tuple(feat_hidden),
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

        # ---------
        # Final MLP: [h_emb, h_feat, action_emb] -> y
        # ---------
        final_in_dim = emb_out_dim + feat_out_dim + self.action_embed_dim
        self.mlp, last_dim = self._make_mlp(
            in_dim=final_in_dim,
            hidden_sizes=tuple(hidden_sizes),
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.head = nn.Linear(last_dim, 1)
        self._last_hidden_dim = last_dim  # for last-layer UCB

        # ---------
        # Gate head: [h_emb, h_feat] -> logit (binary classification)
        # ---------
        gate_in_dim = emb_out_dim + feat_out_dim
        self.gate_mlp, gate_out_dim = self._make_mlp(
            in_dim=gate_in_dim,
            hidden_sizes=tuple(gate_hidden),
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.gate_head = nn.Linear(gate_out_dim, 1)

        # store dims (useful for debugging)
        self._emb_out_dim = emb_out_dim
        self._feat_out_dim = feat_out_dim
        self._gate_in_dim = gate_in_dim

    def _make_mlp(
        self,
        in_dim: int,
        hidden_sizes: Tuple[int, ...],
        dropout: float,
        use_layernorm: bool,
    ):
        """
        Returns (nn.Sequential, out_dim).
        If hidden_sizes is empty, returns Identity and out_dim=in_dim.
        """
        in_dim = int(in_dim)
        if hidden_sizes is None or len(hidden_sizes) == 0:
            return nn.Identity(), in_dim

        layers = []
        cur = in_dim
        for h in hidden_sizes:
            h = int(h)
            layers.append(nn.Linear(cur, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            cur = h
        return nn.Sequential(*layers), cur

    def _check_inputs(self, x_emb, x_feat, domain_id, a: Optional[torch.Tensor] = None):
        if x_emb.dim() != 2:
            raise ValueError(f"x_emb must be (B,D_emb), got {tuple(x_emb.shape)}")
        if x_feat.dim() != 2:
            raise ValueError(f"x_feat must be (B,D_feat), got {tuple(x_feat.shape)}")
        if domain_id.dim() != 1:
            raise ValueError(f"domain_id must be (B,), got {tuple(domain_id.shape)}")
        if x_emb.size(0) != x_feat.size(0) or x_emb.size(0) != domain_id.size(0):
            raise ValueError(
                f"Batch mismatch: x_emb {x_emb.size(0)}, x_feat {x_feat.size(0)}, domain_id {domain_id.size(0)}"
            )
        if x_emb.size(1) != self.x_emb_dim:
            raise ValueError(f"x_emb dim mismatch: got {x_emb.size(1)}, expected {self.x_emb_dim}")
        if x_feat.size(1) != self.x_feat_dim:
            raise ValueError(f"x_feat dim mismatch: got {x_feat.size(1)}, expected {self.x_feat_dim}")
        if a is not None and a.dim() != 1:
            raise ValueError(f"a must be (B,), got {tuple(a.shape)}")
        if a is not None and a.size(0) != x_emb.size(0):
            raise ValueError(f"a batch mismatch: got {a.size(0)}, expected {x_emb.size(0)}")

    def _encode_context(self, x_emb: torch.Tensor, x_feat: torch.Tensor, domain_id: torch.Tensor):
        """
        Encode x_emb and (x_feat + domain_emb) into compact representations.
        Returns:
          h_emb:  (B, H1)
          h_feat: (B, H2)
        """
        domain_id = domain_id.long()
        d_emb = self.domain_emb(domain_id)                 # (B, Dd)
        feat_in = torch.cat([x_feat, d_emb], dim=-1)       # (B, D_feat + Dd)

        h_emb = self.emb_mlp(x_emb)                        # (B, H1) or (B, D_emb)
        h_feat = self.feat_mlp(feat_in)                    # (B, H2) or (B, D_feat+Dd)
        return h_emb, h_feat

    # =========================
    # Reward head
    # =========================
    def forward(
        self,
        x_emb: torch.Tensor,
        x_feat: torch.Tensor,
        domain_id: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        """
        x_emb: (B, D_emb)
        x_feat: (B, D_feat)
        domain_id: (B,)
        a: (B,)
        return: (B,)
        """
        self._check_inputs(x_emb, x_feat, domain_id, a)

        a = a.long()
        a_emb = self.action_emb(a)                         # (B, Da)

        h_emb, h_feat = self._encode_context(x_emb, x_feat, domain_id)
        h_in = torch.cat([h_emb, h_feat, a_emb], dim=-1)   # (B, H1+H2+Da)

        h = self.mlp(h_in)                                 # (B, H_last)
        y = self.head(h).squeeze(-1)                       # (B,)

        if self.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.out_activation == "none":
            pass
        else:
            raise ValueError(f"Unknown out_activation={self.out_activation}")

        return y

    @torch.no_grad()
    def forward_all_actions(self, x_emb: torch.Tensor, x_feat: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """
        Compute f(x, a) for all actions.
        x_emb: (B, D_emb)
        x_feat: (B, D_feat)
        domain_id: (B,)
        returns: (B, K)
        """
        self._check_inputs(x_emb, x_feat, domain_id, a=None)

        B = x_emb.size(0)
        K = self.num_actions

        # encode context once
        h_emb, h_feat = self._encode_context(x_emb, x_feat, domain_id)  # (B,H1),(B,H2)

        # action embeddings (K, Da)
        a_ids = torch.arange(K, device=x_emb.device)
        a_emb = self.action_emb(a_ids)                                   # (K, Da)

        # expand to (B,K,*)
        h_emb_exp = h_emb.unsqueeze(1).expand(B, K, h_emb.size(1))        # (B, K, H1)
        h_feat_exp = h_feat.unsqueeze(1).expand(B, K, h_feat.size(1))     # (B, K, H2)
        a_exp = a_emb.unsqueeze(0).expand(B, K, self.action_embed_dim)    # (B, K, Da)

        h_in = torch.cat([h_emb_exp, h_feat_exp, a_exp], dim=-1)          # (B, K, H1+H2+Da)

        h_in2 = h_in.reshape(B * K, -1)
        h = self.mlp(h_in2)                                               # (B*K, H_last)
        y = self.head(h).reshape(B, K)                                    # (B, K)

        if self.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        return y

    def get_last_hidden(self, x_emb: torch.Tensor, x_feat: torch.Tensor, domain_id: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Return last hidden representation before head (for last-layer UCB).
        x_emb: (B, D_emb), x_feat:(B,D_feat), domain_id:(B,), a:(B,)
        returns: (B, H_last)
        """
        self._check_inputs(x_emb, x_feat, domain_id, a)

        a = a.long()
        a_emb = self.action_emb(a)
        h_emb, h_feat = self._encode_context(x_emb, x_feat, domain_id)
        h_in = torch.cat([h_emb, h_feat, a_emb], dim=-1)
        h = self.mlp(h_in)
        return h

    # =========================
    # Gate head (binary classification)
    # =========================
    def _encode_gate_context(self, x_emb: torch.Tensor, x_feat: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """
        Build gate input z = [h_emb, h_feat].
        If gate_detach_context=True, stop gradient into emb_mlp/feat_mlp/domain_emb.
        """
        h_emb, h_feat = self._encode_context(x_emb, x_feat, domain_id)
        z = torch.cat([h_emb, h_feat], dim=-1)  # (B, emb_out + feat_out)
        if bool(getattr(self, "gate_detach_context", False)):
            z = z.detach()
        return z

    def gate_logits(self, x_emb: torch.Tensor, x_feat: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """
        Returns logits (B,) for high-gap classification.
        """
        self._check_inputs(x_emb, x_feat, domain_id, a=None)
        z = self._encode_gate_context(x_emb, x_feat, domain_id)           # (B, gate_in)
        h = self.gate_mlp(z)                                              # (B, gate_hidden)
        logit = self.gate_head(h).squeeze(-1)                             # (B,)
        return logit

    @torch.no_grad()
    def gate_proba(self, x_emb: torch.Tensor, x_feat: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """
        Returns probabilities (B,) in [0,1].
        """
        return torch.sigmoid(self.gate_logits(x_emb, x_feat, domain_id))

    def gate_parameters(self):
        """
        Parameters for training gate head only.
        """
        return list(self.gate_mlp.parameters()) + list(self.gate_head.parameters())


def build_model_from_cfg(cfg) -> UtilityNet:
    """
    Expects:
      cfg.data.x_dim == D_emb
      cfg.data.x_feat_dim exists
      cfg.data.num_domains exists
      cfg.pool.num_actions() or cfg.data.num_actions or fallback

      cfg.net.* for dims/hiddens
    """
    x_emb_dim = int(getattr(cfg.data, "x_dim"))
    x_feat_dim = int(getattr(cfg.data, "x_feat_dim", 0))
    num_domains = int(getattr(cfg.data, "num_domains", 1))

    # prefer config if available
    if hasattr(cfg, "pool") and hasattr(cfg.pool, "num_actions"):
        try:
            num_actions = int(cfg.pool.num_actions())
        except TypeError:
            num_actions = int(getattr(cfg.pool, "num_actions"))
    else:
        num_actions = int(getattr(cfg.data, "num_actions", 11))

    domain_embed_dim = int(getattr(cfg.net, "domain_embed_dim", 16))
    emb_hidden = tuple(getattr(cfg.net, "emb_hidden", (256,)))
    feat_hidden = tuple(getattr(cfg.net, "feat_hidden", (32,)))

    # gate config (optional)
    gate_hidden = tuple(getattr(cfg.net, "gate_hidden", (64,)))
    gate_detach_context = bool(getattr(cfg.net, "gate_detach_context", True))

    model = UtilityNet(
        x_emb_dim=x_emb_dim,
        x_feat_dim=x_feat_dim,
        num_domains=num_domains,
        num_actions=num_actions,
        action_embed_dim=int(getattr(cfg.net, "action_embed_dim", 32)),
        domain_embed_dim=domain_embed_dim,
        emb_hidden=emb_hidden,
        feat_hidden=feat_hidden,
        hidden_sizes=tuple(getattr(cfg.net, "hidden_sizes", (256, 256))),
        dropout=float(getattr(cfg.net, "dropout", 0.0)),
        use_layernorm=bool(getattr(cfg.net, "use_layernorm", False)),
        out_activation=str(getattr(cfg.net, "out_activation", "none")),
        gate_hidden=gate_hidden,
        gate_detach_context=gate_detach_context,
    )
    return model