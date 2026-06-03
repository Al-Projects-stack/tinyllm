from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Allow running from the ui/ folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from flask import Flask, jsonify, request

from model import GPT

app = Flask(__name__)


# ------------------------- Tokenizer -------------------------

def load_tokenizer(tokenizer_dir: str = "data/tokenizer"):
    from tokenizers import Tokenizer  # type: ignore

    tok_file = Path(tokenizer_dir) / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_file}. Run `python train_tokenizer.py` first."
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


# ------------------------- Sampling helpers -------------------------

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        threshold = values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        remove_mask = cumulative_probs - probs > top_p
        sorted_logits[remove_mask] = float("-inf")

        logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

    return logits


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 0, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(0, input_ids, score)
    return logits


@torch.no_grad()
def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    device: torch.device,
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    eos_id: Optional[int],
) -> str:
    model.eval()

    enc = tokenizer.encode(prompt)
    ids = enc.ids

    bos_id = get_bos_token_id(tokenizer)
    if bos_id is not None and (not ids or ids[0] != bos_id):
        ids = [bos_id] + ids

    generated = torch.tensor([ids], dtype=torch.long, device=device)
    past_kvs = [None] * len(model.blocks)

    for _ in range(max_new_tokens):
        cur_len = generated.shape[1]

        if cur_len >= context_length:
            generated = generated[:, -(context_length - 1) :]
            past_kvs = [None] * len(model.blocks)
            use_new_cache = False
        else:
            use_new_cache = True

        # Use cache when warm; otherwise feed full context
        if all(kv is not None for kv in past_kvs):
            x = generated[:, -1:]
        else:
            x = generated

        logits, _, past_kvs = model(x, use_cache=use_new_cache, past_kvs=past_kvs)
        next_logits = logits[0, -1, :]

        if repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(next_logits, generated[0], repetition_penalty)

        if temperature == 0.0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / max(temperature, 1e-8)
            next_logits = top_k_top_p_filter(next_logits, top_k=top_k, top_p=top_p)
            if torch.all(next_logits == float("-inf")):
                next_logits = torch.zeros_like(next_logits)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    new_ids = generated[0].tolist()[len(ids) :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ------------------------- Checkpoint/model cache -------------------------


@dataclass(frozen=True)
class ModelBundle:
    model: GPT
    tokenizer: object
    device: torch.device
    context_length: int
    eos_id: Optional[int]


_BUNDLES: dict[str, ModelBundle] = {}


def load_bundle_for_checkpoint(checkpoint_path: str, tokenizer_dir: str = "data/tokenizer") -> ModelBundle:
    key = f"{checkpoint_path}||{tokenizer_dir}"
    if key in _BUNDLES:
        return _BUNDLES[key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT.from_checkpoint(checkpoint_path, map_location=str(device))
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer(tokenizer_dir)
    eos_id = get_eos_token_id(tokenizer)

    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=model.config.context_length,
        eos_id=eos_id,
    )
    _BUNDLES[key] = bundle
    return bundle


# ------------------------- Routes -------------------------


@app.route("/", methods=["GET"])
def root():
    # Serve the actual UI page from ui/index.html
    ui_path = Path(__file__).resolve().parent / "index.html"
    if not ui_path.exists():
        return "ui/index.html not found\n", 404
    return ui_path.read_text(encoding="utf-8")


@app.route("/generate", methods=["POST"])
def generate_route():
    payload = request.get_json(force=True)

    prompt = payload.get("prompt", "")
    checkpoint = payload.get("checkpoint", "checkpoints/step_004500.pt")

    max_new_tokens = int(payload.get("max_new_tokens", 120))
    temperature = float(payload.get("temperature", 0.7))
    top_k = int(payload.get("top_k", 40))
    top_p = float(payload.get("top_p", 0.9))
    repetition_penalty = float(payload.get("repetition_penalty", 1.2))

    if not prompt.strip():
        return jsonify({"text": ""})

    bundle = load_bundle_for_checkpoint(checkpoint)

    text = generate_text(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        prompt=prompt,
        device=bundle.device,
        context_length=bundle.context_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=bundle.eos_id,
    )

    return jsonify({"text": text})


if __name__ == "__main__":
    # Start: python ui/server.py
    app.run(host="127.0.0.1", port=8000, debug=False)

