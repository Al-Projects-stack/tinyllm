from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def top_k_top_p_filter(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        threshold = values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove_mask] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 0, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(0, input_ids, score)
    return logits


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str):
    """Load a HuggingFace BPE tokenizer from a directory containing tokenizer.json."""
    from tokenizers import Tokenizer
    tok_file = Path(tokenizer_path) / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_file}. "
            "Run `python train_tokenizer.py` first."
        )
    return Tokenizer.from_file(str(tok_file))


def get_eos_token_id(tokenizer) -> Optional[int]:
    vocab = tokenizer.get_vocab()
    for candidate in ["[EOS]", "</s>", "<|endoftext|>", "<eos>"]:
        if candidate in vocab:
            return vocab[candidate]
    return None


def get_bos_token_id(tokenizer) -> Optional[int]:
    vocab = tokenizer.get_vocab()
    for candidate in ["[BOS]", "<s>", "<|startoftext|>", "<bos>"]:
        if candidate in vocab:
            return vocab[candidate]
    return None
