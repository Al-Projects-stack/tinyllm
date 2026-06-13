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

from inference_utils import (
    apply_repetition_penalty,
    get_bos_token_id,
    get_eos_token_id,
    load_tokenizer,
    top_k_top_p_filter,
)
from model import GPT

app = Flask(__name__)


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
) -> tuple[str, int]:
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
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, len(new_ids)


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

    text, token_count = generate_text(
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

    return jsonify({"text": text, "tokens": token_count})


@app.route("/checkpoints", methods=["GET"])
def list_checkpoints():
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return jsonify([])
    files = sorted(ckpt_dir.glob("step_*.pt"), reverse=True)
    return jsonify([f.name for f in files])


if __name__ == "__main__":
    # Start: python ui/server.py
    app.run(host="127.0.0.1", port=8000, debug=False)

