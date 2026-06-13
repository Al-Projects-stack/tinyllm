"""
model.py — Pure-PyTorch GPT-style decoder-only transformer.

Architecture:
  • RMSNorm (no bias, no mean subtraction)
  • Learned token + positional embeddings
  • Causal multi-head self-attention via F.scaled_dot_product_attention
    (uses FlashAttention kernel when available in PyTorch 2.x)
  • SwiGLU-style gated MLP
  • Optional KV-cache for fast autoregressive inference
  • Weight-tied token embedding ↔ LM head

Parameter count with defaults (vocab=32k, d=256, L=3, H=4): ~11.4 M
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean centering)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Uses torch.nn.functional.scaled_dot_product_attention which dispatches to
    FlashAttention-2 when available (PyTorch >= 2.0 on CUDA).

    KV-cache: pass use_cache=True and past_kv=(k_cache, v_cache) for
    incremental decoding.  During training leave both as None.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0, (
            f"hidden_dim {config.hidden_dim} must be divisible by "
            f"num_heads {config.num_heads}"
        )
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.hidden_dim = config.hidden_dim
        self.dropout_p = config.dropout

        # Fused Q/K/V projection (no bias — matches GPT-J / LLaMA style)
        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # Scaled-down output projection (GPT-2 style residual init).
        # Tag first so _init_weights (called via apply() in GPT.__init__) skips re-init.
        self.out_proj._is_residual_proj = True
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2 * config.num_layers))

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)                                          # (B, T, 3C)
        q, k, v = qkv.split(self.hidden_dim, dim=-1)                   # each (B, T, C)

        # Reshape to (B, H, T, D)
        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)   # (B, H, T, D)

        # Append to KV cache if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = (k, v) if use_cache else None

        # F.scaled_dot_product_attention handles causal masking and FlashAttention.
        # is_causal=True is only valid when there is no KV-cache prefix (T_k == T_q).
        # When past_kv is present and T_q > 1 (multi-token prefill into a warm cache)
        # we must build an explicit mask: each new query position i can attend to all
        # past_len cached keys plus the first (i+1) new keys.
        dropout_p = self.dropout_p if self.training else 0.0
        T_q = q.shape[2]
        T_k = k.shape[2]  # past_len + T_q when cache is present
        if past_kv is not None and T_q > 1:
            # Build a (T_q, T_k) causal mask: query i attends to keys 0 .. past_len+i
            past_len = T_k - T_q
            mask = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).tril(
                diagonal=past_len
            )
            attn_mask = torch.zeros(T_q, T_k, dtype=q.dtype, device=q.device).masked_fill(
                ~mask, float("-inf")
            )
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            # No cache, or single new token (T_q == 1): standard causal path.
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        #                                                                (B, H, T, D)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)           # (B, T, C)
        out = self.out_proj(out)

        return out, present_kv


class SwiGLU(nn.Module):
    """
    Gated linear unit with SiLU gate (SwiGLU variant).
    Two-projection design: gate × up, then down.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        inner_dim = config.mlp_multiplier * config.hidden_dim
        self.gate_proj = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.up_proj   = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, config.hidden_dim, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

        # GPT-2 style residual scaling for down projection.
        # Tag first so _init_weights (called via apply() in GPT.__init__) skips re-init.
        self.down_proj._is_residual_proj = True
        nn.init.normal_(self.down_proj.weight, std=0.02 / math.sqrt(2 * config.num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: norm → attn → residual → norm → mlp → residual."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.attn  = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_dim)
        self.mlp   = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present_kv = self.attn(self.norm1(x), use_cache=use_cache, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, present_kv


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    Decoder-only GPT with:
      • Learned token + absolute positional embeddings
      • Stack of TransformerBlocks
      • Final RMSNorm
      • LM head weight-tied to token embedding

    Forward returns (logits, loss, present_kvs).
    loss is None when targets is None.
    present_kvs is a list of (k, v) tensors per layer; each entry is None when
    use_cache=False.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb   = nn.Embedding(config.context_length, config.hidden_dim)
        self.drop      = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm   = RMSNorm(config.hidden_dim)

        # LM head — weight tied to token embedding (no bias)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        # Initialise all weights
        self.apply(self._init_weights)

        print(f"GPT model initialised: {self.num_parameters():,} parameters "
              f"({self.num_parameters()/1e6:.1f} M)")

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # out_proj and down_proj have already been re-initialised above
            if not getattr(module, "_is_residual_proj", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,                           # (B, T)
        targets: Optional[torch.Tensor] = None,            # (B, T)
        use_cache: bool = False,
        past_kvs: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor],
               list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:

        B, T = input_ids.shape
        device = input_ids.device

        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)

        # Position offset accounts for KV-cached past tokens
        past_len = past_kvs[0][0].shape[2] if past_kvs[0] is not None else 0
        total_len = past_len + T
        if total_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {total_len} exceeds context_length "
                f"{self.config.context_length}"
            )

        positions = torch.arange(past_len, past_len + T, device=device)  # (T,)

        tok = self.token_emb(input_ids)   # (B, T, C)
        pos = self.pos_emb(positions)     # (T, C) — broadcasts over B
        x = self.drop(tok + pos)

        present_kvs: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []
        for block, past_kv in zip(self.blocks, past_kvs):
            x, present_kv = block(x, use_cache=use_cache, past_kv=past_kv)
            present_kvs.append(present_kv)

        x = self.norm(x)
        logits = self.lm_head(x)          # (B, T, vocab_size)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            # Flatten to (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, present_kvs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, map_location: str = "cpu") -> "GPT":
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        config = ckpt["model_config"]
        model = cls(config)
        # Strip any torch.compile prefix from state dict keys
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state)
        return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import get_smoke_config
    cfg = get_smoke_config()
    model = GPT(cfg.model)

    B, T = 4, cfg.model.context_length
    ids = torch.randint(0, cfg.model.vocab_size, (B, T))
    targets = torch.randint(0, cfg.model.vocab_size, (B, T))

    logits, loss, _ = model(ids, targets)
    print(f"logits shape: {logits.shape}")
    print(f"loss: {loss.item():.4f}")
    print("model.py OK")
