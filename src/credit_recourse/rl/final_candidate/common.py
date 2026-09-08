from __future__ import annotations

import torch
from torch import nn

from .encoder import FinalBlockAwareEncoder


class CandidatePolicy(nn.Module):
    """Stage 4 BC head. Keep this linear head stable for BC -> IQL actor transfer."""

    def __init__(self, encoder: FinalBlockAwareEncoder, n_actions: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(int(encoder.d_model), int(n_actions))

    def forward(self, x, missing_mask=None, cat=None):
        return self.head(self.encoder(x, missing_mask=missing_mask, cat=cat))


class CrossAttentionBlock(nn.Module):
    """Action-query cross-attention block for critic-only Q(s, a) scoring."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(int(d_model), int(n_heads), dropout=float(dropout), batch_first=True)
        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) * 2, int(d_model)),
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # query: (B, n_actions, d), kv: (B, T, d)
        attended, _ = self.attn(self.norm1(query), kv, kv, need_weights=False)
        x = query + attended
        x = x + self.ffn(self.norm2(x))
        return x


class DiscreteIQL(nn.Module):
    """Discrete IQL model with backward-compatible critic-head dispatch.

    ``critic_head_arch='linear'`` preserves the legacy Stage5/Stage6 state_dict
    contract exactly: q1/q2 are MLP heads mapping CLS state embedding to all
    action columns. ``critic_head_arch='cross_attention'`` keeps the actor and V
    heads unchanged, but lets action embeddings attend to token-level encoder
    states for twin Q critics.
    """

    def __init__(
        self,
        encoder: FinalBlockAwareEncoder,
        n_actions: int,
        d_model: int | None = None,
        critic_head_arch: str = "linear",
        actor_head_arch: str = "linear",
        candidate_action_vectors: torch.Tensor | None = None,
        action_vec_dim: int = 10,
        n_attn_blocks: int = 2,
        n_attn_heads: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        dm = int(d_model or encoder.d_model)
        self.n_actions = int(n_actions)
        self.critic_head_arch = str(critic_head_arch)
        self.actor_head_arch = str(actor_head_arch)

        # State-value head is unchanged. The default linear actor preserves the
        # legacy Stage4 BC -> Stage5 IQL state_dict contract. The optional
        # action_conditioned actor mirrors the critic's action-query geometry
        # and is default-off because it cannot reuse the old BC linear-head weights.
        self.v = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, 1))
        if self.actor_head_arch == "linear":
            self.pi = nn.Linear(dm, self.n_actions)
        elif self.actor_head_arch == "action_conditioned":
            action_vec_dim = int(action_vec_dim)
            if candidate_action_vectors is None:
                candidate_action_vectors = torch.zeros(self.n_actions, action_vec_dim, dtype=torch.float32)
            if tuple(candidate_action_vectors.shape) != (self.n_actions, action_vec_dim):
                raise ValueError(
                    f"candidate_action_vectors must be ({self.n_actions},{action_vec_dim}); "
                    f"got {tuple(candidate_action_vectors.shape)}"
                )
            self.register_buffer("candidate_action_vectors", candidate_action_vectors.float(), persistent=True)
            self.actor_action_proj = nn.Sequential(nn.Linear(action_vec_dim, dm), nn.LayerNorm(dm))
            self.actor_blocks = nn.ModuleList([CrossAttentionBlock(dm, n_attn_heads, attn_dropout) for _ in range(int(n_attn_blocks))])
            self.actor_residual = nn.Parameter(torch.zeros(self.n_actions, dm))
            self.actor_out = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, dm // 2), nn.GELU(), nn.Linear(dm // 2, 1))
        else:
            raise ValueError(f"Unknown actor_head_arch={actor_head_arch!r}")

        if self.critic_head_arch == "linear":
            self.q1 = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, self.n_actions))
            self.q2 = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, self.n_actions))
        elif self.critic_head_arch in ("cross_attention", "cross_attention_film"):
            action_vec_dim = int(action_vec_dim)
            if candidate_action_vectors is None:
                candidate_action_vectors = torch.zeros(self.n_actions, action_vec_dim, dtype=torch.float32)
            if tuple(candidate_action_vectors.shape) != (self.n_actions, action_vec_dim):
                raise ValueError(
                    f"candidate_action_vectors must be ({self.n_actions},{action_vec_dim}); "
                    f"got {tuple(candidate_action_vectors.shape)}"
                )
            if not hasattr(self, "candidate_action_vectors"):
                self.register_buffer("candidate_action_vectors", candidate_action_vectors.float(), persistent=True)
            self.action_proj = nn.Sequential(nn.Linear(action_vec_dim, dm), nn.LayerNorm(dm))
            self.action_residual_1 = nn.Parameter(torch.zeros(self.n_actions, dm))
            self.action_residual_2 = nn.Parameter(torch.zeros(self.n_actions, dm))
            self.q1_blocks = nn.ModuleList(
                [CrossAttentionBlock(dm, n_attn_heads, attn_dropout) for _ in range(int(n_attn_blocks))]
            )
            self.q2_blocks = nn.ModuleList(
                [CrossAttentionBlock(dm, n_attn_heads, attn_dropout) for _ in range(int(n_attn_blocks))]
            )
            self.q1_out = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, dm // 2), nn.GELU(), nn.Linear(dm // 2, 1))
            self.q2_out = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, dm // 2), nn.GELU(), nn.Linear(dm // 2, 1))
            if self.critic_head_arch == "cross_attention_film":
                # Per-firm FiLM modulation of the firm-independent action query by the
                # CLS state embedding. Additive cross-attention alone cannot model the
                # multiplicative action x firm-state interaction that governs which
                # action is best for a given firm (e.g. deleveraging is worth more to a
                # high-leverage firm); FiLM injects that interaction, targeting per-firm
                # Q-alpha fidelity. The final layer is zero-initialised so scale=1,
                # shift=0 at init => this variant starts byte-equivalent in output to
                # 'cross_attention' and can only depart from it through learning.
                self.q_film = nn.Sequential(
                    nn.LayerNorm(dm), nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, 2 * dm)
                )
                nn.init.zeros_(self.q_film[-1].weight)
                nn.init.zeros_(self.q_film[-1].bias)
        else:
            raise ValueError(f"Unknown critic_head_arch={critic_head_arch!r}")

    def encode(self, x, missing_mask=None, cat=None):
        return self.encoder(x, missing_mask=missing_mask, cat=cat)

    def _encode_tokens(self, x, missing_mask=None, cat=None):
        if not hasattr(self.encoder, "forward_tokens"):
            raise AttributeError("FinalBlockAwareEncoder.forward_tokens is required for cross_attention critic")
        return self.encoder.forward_tokens(x, missing_mask=missing_mask, cat=cat)

    def _q_attention(
        self,
        tokens: torch.Tensor,
        blocks: nn.ModuleList,
        out_head: nn.Module,
        action_residual: nn.Parameter,
        film_params: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = int(tokens.size(0))
        action_query = self.action_proj(self.candidate_action_vectors) + action_residual
        x = action_query.unsqueeze(0).expand(batch_size, -1, -1)
        if film_params is not None:
            # Firm-conditioned FiLM: x <- x * (1 + scale) + shift, broadcast over actions.
            scale, shift = film_params
            x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        for block in blocks:
            x = block(x, tokens)
        return out_head(x).squeeze(-1)

    def q_values(self, x, missing_mask=None, cat=None):
        if self.critic_head_arch == "linear":
            z = self.encode(x, missing_mask=missing_mask, cat=cat)
            return self.q1(z), self.q2(z)
        tokens = self._encode_tokens(x, missing_mask=missing_mask, cat=cat)
        film_params = None
        if self.critic_head_arch == "cross_attention_film":
            cls_embedding = tokens[:, 0, :]
            scale, shift = self.q_film(cls_embedding).chunk(2, dim=-1)
            film_params = (scale, shift)
        q1 = self._q_attention(tokens, self.q1_blocks, self.q1_out, self.action_residual_1, film_params)
        q2 = self._q_attention(tokens, self.q2_blocks, self.q2_out, self.action_residual_2, film_params)
        return q1, q2

    def q_min(self, x, missing_mask=None, cat=None):
        q1, q2 = self.q_values(x, missing_mask=missing_mask, cat=cat)
        return torch.minimum(q1, q2)

    def v_value(self, x, missing_mask=None, cat=None):
        return self.v(self.encode(x, missing_mask=missing_mask, cat=cat)).squeeze(-1)

    def logits(self, x, missing_mask=None, cat=None):
        if self.actor_head_arch == "linear":
            return self.pi(self.encode(x, missing_mask=missing_mask, cat=cat))
        tokens = self._encode_tokens(x, missing_mask=missing_mask, cat=cat)
        batch_size = int(tokens.size(0))
        action_query = self.actor_action_proj(self.candidate_action_vectors) + self.actor_residual
        z = action_query.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.actor_blocks:
            z = block(z, tokens)
        return self.actor_out(z).squeeze(-1)
