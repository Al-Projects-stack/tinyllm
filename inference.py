"""
inference.py — Autoregressive text generation with a trained tiny GPT.

Supports:
  • Temperature sampling
  • Top-k filtering
  • Top-p (nucleus) filtering
  • Repetition penalty
  • Greedy decoding (temperature=0)
  • Context-window sliding (sequences exceeding context_length are trimmed)
  • KV-cache for fast generation
  • Streaming output (character-by-character to stdout)
  • Interactive CLI prompt loop

Usage:
    # Interactive mode
    python inference.py --checkpoint checkpoints/best.pt

    # Single prompt
    python inference.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"

    # Greedy decoding
    python inference.py --checkpoint checkpoints/best.pt --prompt "Hello" --temperature 0

    # Control generation
    python inference.py --checkpoint checkpoints/best.pt \\
        --prompt "In a small village" \\
        --max-new-tokens 300 \\
        --temperature 0.9 \\
        --top-k 40 \\
        --top-p 0.95
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from config import Config, GenerationConfig
from model import GPT


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def top_k_top_p_filter(
    logits: torch.Tensor,      # (vocab_size,)
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Apply top-k and/or top-p (nucleus) filtering to logits.
    Returns filtered logits (masked with -inf for excluded tokens).
    """
    if top_k > 0:
        # Keep only top-k values; mask the rest
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        threshold = values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens whose cumulative probability exceeds top_p
        # Shift by 1 to include the token that pushes probability over p
        remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove_mask] = float("-inf")

        # Scatter back to original indexing
        logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,      # (vocab_size,)
    input_ids: torch.Tensor,   # (T,) — context so far
    penalty: float,
) -> torch.Tensor:
    """Reduce logit of previously generated tokens (repetition penalty > 1.0)."""
    if penalty == 1.0:
        return logits
    # Gather logits at generated positions and divide / multiply
    score = torch.gather(logits, 0, input_ids)
    # Penalise: positive logits are reduced, negative logits are increased
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(0, input_ids, score)
    return logits


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str):
    """Load a HuggingFace BPE tokenizer from tokenizer.json."""
    from tokenizers import Tokenizer  # type: ignore
    tok_file = Path(tokenizer_path) / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_file}. "
            "Run `python train_tokenizer.py` first."
        )
    return Tokenizer.from_file(str(tok_file))


def get_eos_token_id(tokenizer) -> Optional[int]:
    """Return the EOS token id if present, else None."""
    vocab = tokenizer.get_vocab()
    for candidate in ["[EOS]", "</s>", "<|endoftext|>", "<eos>"]:
        if candidate in vocab:
            return vocab[candidate]
    return None


def get_bos_token_id(tokenizer) -> Optional[int]:
    """Return the BOS token id if present, else None."""
    vocab = tokenizer.get_vocab()
    for candidate in ["[BOS]", "<s>", "<|startoftext|>", "<bos>"]:
        if candidate in vocab:
            return vocab[candidate]
    return None


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: GPT,
    input_ids: torch.Tensor,       # (1, T) — already on device
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    stream: bool = True,
    decode_fn=None,
) -> torch.Tensor:
    """
    Autoregressive generation.

    Args:
        model:              GPT model in eval mode.
        input_ids:          Prompt token IDs, shape (1, T).
        max_new_tokens:     Maximum number of new tokens to generate.
        temperature:        Sampling temperature. 0 = greedy.
        top_k:              Keep only top-k tokens.
        top_p:              Nucleus sampling probability threshold.
        repetition_penalty: Penalty for already-generated tokens.
        eos_token_id:       Stop generation when this token is produced.
        stream:             Print tokens to stdout as they are generated.
        decode_fn:          Function mapping list[int] → str for streaming.

    Returns:
        Tensor of shape (1, T + n_generated) containing prompt + new tokens.
    """
    model.eval()
    context_length = model.config.context_length
    generated = input_ids  # (1, T)
    past_kvs = [None] * len(model.blocks)  # one entry per transformer layer

    # For streaming: track what we've already printed
    n_prompt = input_ids.shape[1]

    for _ in range(max_new_tokens):
        # Slide the context window if exceeding max length
        cur_len = generated.shape[1]
        if cur_len >= context_length:
            # Take the last (context_length - 1) tokens as new input
            # and reset KV cache (simple but robust approach)
            generated = generated[:, -(context_length - 1):]
            past_kvs = [None] * len(model.blocks)
            use_new_cache = False
        else:
            use_new_cache = True

        # Forward pass — feed only the last token when KV cache is warm
        if all(kv is not None for kv in past_kvs):
            x = generated[:, -1:]  # (1, 1)
        else:
            x = generated          # (1, T)

        logits, _, past_kvs = model(x, use_cache=use_new_cache, past_kvs=past_kvs)
        # logits: (1, T_new, vocab_size) — take last position
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # Repetition penalty
        if repetition_penalty != 1.0:
            all_ids = generated[0]
            next_logits = apply_repetition_penalty(next_logits, all_ids, repetition_penalty)

        # Sampling
        if temperature == 0.0:
            # Greedy
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (1,)
        else:
            next_logits = next_logits / max(temperature, 1e-8)
            next_logits = top_k_top_p_filter(next_logits, top_k=top_k, top_p=top_p)
            # Guard against all-inf (e.g. empty top-k/top-p result)
            if torch.all(next_logits == float("-inf")):
                next_logits = torch.zeros_like(next_logits)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)          # (1,)

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)  # (1, T+1)

        # Streaming: decode the newly added token and print it
        if stream and decode_fn is not None:
            token_id = next_token.item()
            token_str = decode_fn([token_id])
            # Strip BOS/EOS markers from display
            for marker in ["[BOS]", "[EOS]", "[PAD]", "[SEP]", "[MASK]", "[UNK]"]:
                token_str = token_str.replace(marker, "")
            print(token_str, end="", flush=True)

        # EOS check
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    if stream:
        print()  # newline after streaming

    return generated


# ---------------------------------------------------------------------------
# Model + tokenizer loader
# ---------------------------------------------------------------------------

def load_artifacts(
    checkpoint_path: str,
    tokenizer_path: str,
    device: torch.device,
) -> tuple[GPT, object]:
    """Load model and tokenizer from disk."""
    print(f"[inference] Loading model from {checkpoint_path} ...")
    model = GPT.from_checkpoint(checkpoint_path, map_location=str(device))
    model.to(device)
    model.eval()
    print(f"[inference] Model loaded ({model.num_parameters():,} params)")

    print(f"[inference] Loading tokenizer from {tokenizer_path} ...")
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"[inference] Tokenizer loaded ({tokenizer.get_vocab_size()} tokens)")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Interactive CLI loop
# ---------------------------------------------------------------------------

def run_interactive(
    model: GPT,
    tokenizer,
    device: torch.device,
    gen_config: GenerationConfig,
) -> None:
    """Run a REPL where the user types prompts and the model completes them."""
    eos_id = get_eos_token_id(tokenizer)
    bos_id = get_bos_token_id(tokenizer)

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=False)

    print("\n" + "=" * 60)
    print("  Tiny LLM -- Interactive Generation")
    print("  Type a prompt and press Enter.  Ctrl+C or 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[inference] Bye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("[inference] Bye!")
            break
        if not prompt:
            print("[inference] (empty prompt -- skipping)\n")
            continue

        # Encode prompt
        enc = tokenizer.encode(prompt)
        ids = enc.ids

        # Prepend BOS if available
        if bos_id is not None and (not ids or ids[0] != bos_id):
            ids = [bos_id] + ids

        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

        print(f"\n[Model output]\n{prompt}", end="", flush=True)
        try:
            generate(
                model=model,
                input_ids=input_tensor,
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_k=gen_config.top_k,
                top_p=gen_config.top_p,
                repetition_penalty=gen_config.repetition_penalty,
                eos_token_id=eos_id,
                stream=True,
                decode_fn=decode,
            )
        except Exception as exc:
            print(f"\n[inference] Generation error: {exc}")

        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny GPT inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file).")
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizer",
                        help="Directory containing tokenizer.json.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt text.  If omitted, enters interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming; print full output at once.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', 'cuda', 'mps'. Auto-detected if omitted.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[inference] Using device: {device}")

    model, tokenizer = load_artifacts(args.checkpoint, args.tokenizer_dir, device)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    if args.prompt is not None:
        # Single prompt mode
        eos_id = get_eos_token_id(tokenizer)
        bos_id = get_bos_token_id(tokenizer)

        enc = tokenizer.encode(args.prompt)
        ids = enc.ids
        if bos_id is not None and (not ids or ids[0] != bos_id):
            ids = [bos_id] + ids

        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        stream = not args.no_stream

        if stream:
            print(args.prompt, end="", flush=True)

        output = generate(
            model=model,
            input_ids=input_tensor,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_k=gen_cfg.top_k,
            top_p=gen_cfg.top_p,
            repetition_penalty=gen_cfg.repetition_penalty,
            eos_token_id=eos_id,
            stream=stream,
            decode_fn=lambda x: tokenizer.decode(x, skip_special_tokens=False),
        )

        if not stream:
            new_ids = output[0, len(ids):].tolist()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            print(f"{args.prompt}{text}")
    else:
        run_interactive(model, tokenizer, device, gen_cfg)
