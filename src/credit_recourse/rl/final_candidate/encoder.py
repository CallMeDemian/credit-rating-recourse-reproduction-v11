from __future__ import annotations
from typing import Any
import torch
from torch import nn

from credit_recourse.rl.contracts.avs256_acd_v2 import SCHEMA_VERSION


FINAL_ENCODER_D_MODEL = 256
FINAL_ENCODER_N_HEADS = 8
FINAL_ENCODER_N_LAYERS = 4
FINAL_ENCODER_FF_MULTIPLIER = 4


class ActionConditionalForwardHead(nn.Module):
    """Action-conditional next-state dynamics head with state × action interaction."""

    def __init__(self, d_model: int, n_actions: int, n_acd_targets: int, action_emb_dim: int = 64):
        super().__init__()
        self.d_model = int(d_model)
        self.n_actions = int(n_actions)
        self.n_acd_targets = int(n_acd_targets)
        self.action_emb_dim = int(action_emb_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(self.n_actions, self.action_emb_dim),
            nn.GELU(),
            nn.Linear(self.action_emb_dim, self.d_model),
        )
        self.additive = nn.Sequential(
            nn.Linear(self.d_model + self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.n_acd_targets),
        )
        self.interaction = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.n_acd_targets),
        )
        self.gate = nn.Parameter(torch.zeros(self.n_acd_targets))

    def forward(self, phi: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor:
        if action_norm.ndim != 2 or action_norm.shape[1] != self.n_actions:
            raise ValueError(f"Expected action_norm shape (batch,{self.n_actions}), got {tuple(action_norm.shape)}")
        a_emb = self.action_proj(action_norm)
        add_out = self.additive(torch.cat([phi, a_emb], dim=1))
        inter_out = self.interaction(phi * a_emb)
        g = torch.sigmoid(self.gate).unsqueeze(0)
        return g * inter_out + (1.0 - g) * add_out


class FinalBlockAwareEncoder(nn.Module):
    """Final AVS256 block-aware encoder.

    The normal forward path is state-only. Actions enter only through
    forward_pretrain() for the Stage3 ACD objective and through separate IQL
    critics downstream.
    """

    def __init__(
        self,
        n_features: int,
        block_ids: list[int] | tuple[int, ...],
        direction_ids: list[int] | tuple[int, ...] | None = None,
        d_model: int = FINAL_ENCODER_D_MODEL,
        n_heads: int = FINAL_ENCODER_N_HEADS,
        n_layers: int = FINAL_ENCODER_N_LAYERS,
        dropout: float = 0.1,
        n_actions: int = 10,
        n_categorical_tokens: int = 0,
        n_categorical_fields: int = 0,
        n_acd_targets: int | None = None,
        action_emb_dim: int = 64,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ff_multiplier = FINAL_ENCODER_FF_MULTIPLIER
        self.n_actions = int(n_actions)
        self.n_acd_targets = int(n_acd_targets if n_acd_targets is not None else n_features)
        self.action_emb_dim = int(action_emb_dim)
        if len(block_ids) != self.n_features:
            raise ValueError(f"block_ids length {len(block_ids)} != n_features {self.n_features}")
        if direction_ids is None:
            direction_ids = [0] * self.n_features
        if len(direction_ids) != self.n_features:
            raise ValueError(f"direction_ids length {len(direction_ids)} != n_features {self.n_features}")
        self.register_buffer("block_ids", torch.tensor(block_ids, dtype=torch.long), persistent=True)
        self.register_buffer("direction_ids", torch.tensor(direction_ids, dtype=torch.long), persistent=True)
        self.value_weight = nn.Parameter(torch.empty(self.n_features, self.d_model))
        self.value_bias = nn.Parameter(torch.zeros(self.n_features, self.d_model))
        nn.init.normal_(self.value_weight, std=0.02)
        self.feature_emb = nn.Embedding(self.n_features, self.d_model)
        self.block_emb = nn.Embedding(max(block_ids) + 1 if block_ids else 1, self.d_model)
        self.direction_emb = nn.Embedding(max(direction_ids) + 1 if direction_ids else 1, self.d_model)
        self.missing_emb = nn.Embedding(3, self.d_model)
        self.n_categorical_tokens = int(n_categorical_tokens)
        self.n_categorical_fields = int(n_categorical_fields)
        self.categorical_emb = nn.Embedding(max(1, self.n_categorical_tokens), self.d_model) if self.n_categorical_tokens else None
        self.categorical_field_emb = nn.Embedding(max(1, self.n_categorical_fields), self.d_model) if self.n_categorical_fields else None
        self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * self.ff_multiplier,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self.recon = nn.Linear(self.d_model, 1)
        self.forward_head = ActionConditionalForwardHead(
            self.d_model, self.n_actions, self.n_acd_targets, action_emb_dim=self.action_emb_dim
        )
        self.projector = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, 64))

    def _continuous_tokens(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None, mcm_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Expected x shape (batch,{self.n_features}), got {tuple(x.shape)}")
        feat_ids = torch.arange(self.n_features, device=x.device)
        tok = x.unsqueeze(-1) * self.value_weight.unsqueeze(0) + self.value_bias.unsqueeze(0)
        tok = tok + self.feature_emb(feat_ids).unsqueeze(0)
        tok = tok + self.block_emb(self.block_ids.to(x.device)).unsqueeze(0)
        tok = tok + self.direction_emb(self.direction_ids.to(x.device)).unsqueeze(0)
        status = torch.zeros_like(x, dtype=torch.long)
        if missing_mask is not None:
            status = torch.where(missing_mask.bool(), torch.ones_like(status), status)
        if mcm_mask is not None:
            status = torch.where(mcm_mask.bool(), torch.full_like(status, 2), status)
        return tok + self.missing_emb(status)

    def _categorical_tokens(self, cat: torch.Tensor | None) -> torch.Tensor | None:
        if cat is None or self.categorical_emb is None or self.categorical_field_emb is None:
            return None
        if cat.ndim != 2 or cat.shape[1] != self.n_categorical_fields:
            raise ValueError(f"Expected categorical tensor shape (batch,{self.n_categorical_fields}), got {tuple(cat.shape)}")
        field_ids = torch.arange(self.n_categorical_fields, device=cat.device)
        cat_ids = cat.long().clamp_min(0).clamp_max(max(0, self.n_categorical_tokens - 1))
        return self.categorical_emb(cat_ids) + self.categorical_field_emb(field_ids).unsqueeze(0)

    def tokens(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None, mcm_mask: torch.Tensor | None = None, cat: torch.Tensor | None = None) -> torch.Tensor:
        cont = self._continuous_tokens(x, missing_mask, mcm_mask)
        cats = self._categorical_tokens(cat)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, cont, cats], dim=1) if cats is not None else torch.cat([cls, cont], dim=1)

    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None, cat: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(self.tokens(x, missing_mask=missing_mask, mcm_mask=None, cat=cat))
        return h[:, 0, :]

    def forward_tokens(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None, cat: torch.Tensor | None = None) -> torch.Tensor:
        """Token-level forward path for critic-only cross-attention heads.

        This performs the same tokenization and Transformer encoder pass as
        ``forward()``, but returns every token instead of only the CLS slice.
        It does not add, remove, or rename any parameter/buffer, so existing
        Stage3 encoder checkpoints remain strict-load compatible.
        """
        return self.encoder(self.tokens(x, missing_mask=missing_mask, mcm_mask=None, cat=cat))

    def forward_pretrain(self, x: torch.Tensor, action_norm: torch.Tensor, missing_mask: torch.Tensor | None = None, mcm_mask: torch.Tensor | None = None, cat: torch.Tensor | None = None):
        h = self.encoder(self.tokens(x, missing_mask=missing_mask, mcm_mask=mcm_mask, cat=cat))
        cls = h[:, 0, :]
        feat = h[:, 1 : 1 + self.n_features, :]
        recon = self.recon(feat).squeeze(-1)
        next_pred = self.forward_head(cls, action_norm)
        proj = self.projector(cls)
        return cls, recon, next_pred, proj


def build_encoder_from_payload(payload: dict[str, Any]) -> FinalBlockAwareEncoder:
    cfg = payload.get("model_config", {})
    schema = payload.get("schema", {})
    features = payload.get("features") or schema.get("continuous_columns") or []
    block_ids = cfg.get("block_ids") or schema.get("feature_block_ids") or [0] * len(features)
    direction_ids = cfg.get("direction_ids") or schema.get("feature_direction_ids") or [0] * len(features)
    action_columns = schema.get("action_columns") or cfg.get("action_columns") or []
    cat_vocab = schema.get("categorical_vocab") or {}
    n_cat_tokens = int(cfg.get("n_categorical_tokens", schema.get("n_categorical_tokens", 0) or 0))
    n_cat_fields = int(cfg.get("n_categorical_fields", schema.get("n_categorical_fields", len(cat_vocab) if isinstance(cat_vocab, dict) else 0) or 0))
    n_acd_targets = int(cfg.get("n_acd_targets", schema.get("n_acd_targets", len(schema.get("acd_target_columns", [])) or len(features)) or len(features)))
    action_emb_dim = int(cfg.get("action_emb_dim", schema.get("acd_action_emb_dim", 64) or 64))
    return FinalBlockAwareEncoder(
        len(features),
        block_ids,
        direction_ids,
        d_model=int(cfg.get("d_model", FINAL_ENCODER_D_MODEL)),
        n_heads=int(cfg.get("n_heads", FINAL_ENCODER_N_HEADS)),
        n_layers=int(cfg.get("n_layers", FINAL_ENCODER_N_LAYERS)),
        dropout=float(cfg.get("dropout", 0.1)),
        n_actions=int(cfg.get("n_actions", len(action_columns) or 10)),
        n_categorical_tokens=n_cat_tokens,
        n_categorical_fields=n_cat_fields,
        n_acd_targets=n_acd_targets,
        action_emb_dim=action_emb_dim,
    )




def _validate_final_encoder_architecture(payload: dict[str, Any]) -> None:
    """Fail fast when a Stage3 payload regresses from the final AVS256 encoder."""
    cfg = payload.get("model_config", {}) or {}
    expected = {
        "d_model": FINAL_ENCODER_D_MODEL,
        "n_heads": FINAL_ENCODER_N_HEADS,
        "n_layers": FINAL_ENCODER_N_LAYERS,
        "ff_multiplier": FINAL_ENCODER_FF_MULTIPLIER,
    }
    observed = {
        "d_model": int(cfg.get("d_model", -1)),
        "n_heads": int(cfg.get("n_heads", -1)),
        "n_layers": int(cfg.get("n_layers", -1)),
        "ff_multiplier": int(cfg.get("ff_multiplier", FINAL_ENCODER_FF_MULTIPLIER)),
    }
    bad = {k: {"expected": v, "actual": observed.get(k)} for k, v in expected.items() if observed.get(k) != v}
    if bad:
        raise RuntimeError({
            "message": "Stage3 encoder architecture mismatch; final AVS256 requires 256/8/4/4",
            "mismatch": bad,
            "model_config": cfg,
        })

def load_stage3_encoder_payload(payload: dict[str, Any], strict: bool = True) -> FinalBlockAwareEncoder:
    schema = payload.get("schema", {}) or {}
    _validate_final_encoder_architecture(payload)
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Stage3 encoder schema_version must be {SCHEMA_VERSION}; got {schema.get('schema_version')}")
    if int(schema.get("n_continuous_features", -1)) != 129:
        raise RuntimeError(f"Stage3 encoder must have 129 continuous features; got {schema.get('n_continuous_features')}")
    enc = build_encoder_from_payload(payload)
    state = payload.get("encoder_state_dict")
    if state is None:
        raise KeyError("Stage3 payload missing encoder_state_dict")
    result = enc.load_state_dict(state, strict=strict)
    missing = list(result.missing_keys) if hasattr(result, "missing_keys") else []
    unexpected = list(result.unexpected_keys) if hasattr(result, "unexpected_keys") else []
    if strict and (missing or unexpected):
        raise RuntimeError({"missing_keys": missing, "unexpected_keys": unexpected, "message": "Stage3 encoder state did not load exactly"})
    return enc
