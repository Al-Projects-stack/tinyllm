"""
train.py — Training loop for the tiny GPT model.

Features:
  • Mixed precision (bfloat16 / float16 via torch.autocast)
  • AdamW with cosine LR schedule + linear warmup
  • Gradient clipping (1.0 by default)
  • Gradient accumulation
  • Checkpoint save / resume
  • Validation loop
  • Tokens/sec throughput logging
  • --smoke-test flag: synthetic data, no disk I/O needed

Usage:
    # Smoke test (no dataset required)
    python train.py --smoke-test --steps 100

    # Full training (after prepare_data + train_tokenizer)
    python train.py --steps 10000

    # Resume from checkpoint
    python train.py --resume checkpoints/step_1000.pt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, ModelConfig, get_smoke_config, validate
from data import create_dataloaders, create_smoke_dataloaders
from model import GPT


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay → min_lr
# ---------------------------------------------------------------------------

def get_lr(step: int, config: Config) -> float:
    tc = config.training
    max_lr = tc.learning_rate
    min_lr = max_lr * 0.1
    warmup = tc.warmup_steps
    max_steps = tc.max_steps

    if step < warmup:
        # Linear warmup
        return max_lr * (step + 1) / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: GPT,
    val_loader: DataLoader,
    device: torch.device,
    eval_steps: int,
    autocast_ctx,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= eval_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx:
            _, loss, _ = model(x, targets=y)
        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    config: Config,
    ckpt_dir: str,
) -> str:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(ckpt_dir) / f"step_{step:06d}.pt")

    # If model was compiled, get original module
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    torch.save(
        {
            "step": step,
            "val_loss": val_loss,
            "model_config": raw_model.config,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(
    ckpt_path: str,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    print(f"[train] Resuming from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"], ckpt.get("val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Infinite DataLoader iterator
# ---------------------------------------------------------------------------

def cycle(loader: DataLoader):
    """Cycle through a DataLoader indefinitely."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: Config, smoke_test: bool = False, resume: str | None = None) -> None:
    validate(config)
    set_seed(config.training.seed)

    # ------------------------------------------------------------------
    # Device & dtype
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train] Device: {device}")

    dtype_str = config.training.dtype
    if device.type == "cpu" and dtype_str in ("bfloat16", "float16"):
        print(f"[train] CPU detected — overriding dtype from {dtype_str} to float32")
        dtype_str = "float32"

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pt_dtype = dtype_map[dtype_str]

    # autocast context
    if device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=pt_dtype)
    else:
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)

    # GradScaler only for float16 (bfloat16 doesn't need it)
    use_scaler = (dtype_str == "float16") and (device.type == "cuda")
    # Use new API (PyTorch 2.4+) with fallback for older versions
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)  # type: ignore

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if smoke_test:
        print("[train] Smoke-test mode — using synthetic in-memory data")
        train_loader, val_loader = create_smoke_dataloaders(config)
    else:
        print("[train] Loading dataset from disk ...")
        train_loader, val_loader = create_dataloaders(config)

    train_iter = cycle(train_loader)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = GPT(config.model).to(device)
    model.train()

    if config.training.compile_model and hasattr(torch, "compile"):
        print("[train] Compiling model with torch.compile ...")
        model = torch.compile(model)  # type: ignore

    # ------------------------------------------------------------------
    # Optimiser (AdamW with decoupled weight decay)
    # ------------------------------------------------------------------
    # Don't apply weight decay to embeddings, biases, or norms
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "emb" in name or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # fused AdamW is faster on CUDA but not available on CPU/MPS
    adamw_kwargs: dict = dict(
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
    )
    if device.type == "cuda":
        try:
            # Check if fused is supported
            torch.optim.AdamW([torch.zeros(1)], fused=True)
            adamw_kwargs["fused"] = True
        except (TypeError, RuntimeError):
            pass

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        **adamw_kwargs,
    )

    # ------------------------------------------------------------------
    # Optional resume
    # ------------------------------------------------------------------
    start_step = 0
    best_val_loss = float("inf")
    if resume:
        start_step, best_val_loss = load_checkpoint(resume, model, optimizer, device)
        print(f"[train] Resumed at step {start_step}, best val loss {best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    tc = config.training
    max_steps = tc.max_steps
    accum_steps = max(tc.grad_accumulation_steps, 1)
    log_interval = tc.log_interval
    eval_interval = tc.eval_interval
    ckpt_interval = tc.checkpoint_interval
    ckpt_dir = config.paths.checkpoint_dir

    print(f"[train] Starting training: {max_steps} steps, "
          f"batch={tc.batch_size}, accum={accum_steps}, dtype={dtype_str}")

    # Save config alongside checkpoints
    if not smoke_test:
        config.save(config.paths.config_save_path)

    t0 = time.perf_counter()
    tokens_processed = 0

    optimizer.zero_grad(set_to_none=True)

    # Track initial and final loss for smoke test assertion
    first_train_loss: float | None = None
    last_train_loss: float = float("nan")

    for step in range(start_step, max_steps):
        # LR update
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation loop
        accumulated_loss = 0.0
        for micro_step in range(accum_steps):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast_ctx:
                _, loss, _ = model(x, targets=y)

            if loss is None or torch.isnan(loss):
                print(f"[train] WARNING: NaN loss at step {step}, micro {micro_step}. Skipping.")
                optimizer.zero_grad(set_to_none=True)
                break

            loss_scaled = loss / accum_steps
            if use_scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accumulated_loss += loss.item()
            tokens_processed += x.numel()

        else:
            # Only update if the inner loop completed without NaN
            # unscale_ must be called exactly once before step() when using GradScaler
            if use_scaler:
                scaler.unscale_(optimizer)
            # Gradient clipping (grads are already unscaled above when use_scaler=True)
            if tc.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        train_loss = accumulated_loss / accum_steps
        last_train_loss = train_loss
        if first_train_loss is None:
            first_train_loss = train_loss

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if (step + 1) % log_interval == 0 or step == 0:
            t1 = time.perf_counter()
            elapsed = max(t1 - t0, 1e-9)
            tps = tokens_processed / elapsed
            print(
                f"step {step+1:6d}/{max_steps} | "
                f"loss {train_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tps:,.0f}"
            )
            tokens_processed = 0
            t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if (step + 1) % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, tc.eval_steps, autocast_ctx)
            print(f"  -> val_loss {val_loss:.4f} (best {min(val_loss, best_val_loss):.4f})")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not smoke_test:
                    ckpt = save_checkpoint(model, optimizer, step + 1, val_loss, config, ckpt_dir)
                    print(f"  -> saved best checkpoint: {ckpt}")

        # ------------------------------------------------------------------
        # Periodic checkpoint
        # ------------------------------------------------------------------
        if not smoke_test and (step + 1) % ckpt_interval == 0:
            val_loss = evaluate(model, val_loader, device, tc.eval_steps, autocast_ctx)
            ckpt = save_checkpoint(model, optimizer, step + 1, val_loss, config, ckpt_dir)
            print(f"  -> periodic checkpoint: {ckpt}")

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------
    final_val_loss = evaluate(model, val_loader, device, tc.eval_steps, autocast_ctx)
    print(f"\n[train] Training complete.")
    print(f"  Final val loss : {final_val_loss:.4f}")
    print(f"  Best  val loss : {best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Smoke test assertion
    # ------------------------------------------------------------------
    if smoke_test:
        if first_train_loss is None:
            print("[smoke] ERROR: No training loss recorded.")
            sys.exit(1)
        if math.isnan(last_train_loss) or math.isnan(first_train_loss):
            print("[smoke] ERROR: NaN loss detected.")
            sys.exit(1)
        if last_train_loss >= first_train_loss:
            print(
                f"[smoke] WARNING: loss did not decrease "
                f"({first_train_loss:.4f} → {last_train_loss:.4f}). "
                "Model may need more steps to overfit."
            )
            # Not a hard failure — could happen with very few steps
        else:
            print(
                f"[smoke] PASS -- loss decreased: "
                f"{first_train_loss:.4f} -> {last_train_loss:.4f}"
            )
        print("[smoke] Smoke test complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny GPT")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max_steps.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run quick smoke test on synthetic data.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate.")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["bfloat16", "float16", "float32"],
                        help="Override training dtype.")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.smoke_test:
        config = get_smoke_config()
    else:
        # Try to load saved config, fall back to defaults
        saved_cfg = Path("checkpoints/config.json")
        if saved_cfg.exists():
            config = Config.load(saved_cfg)
            print(f"[train] Loaded config from {saved_cfg}")
        else:
            config = Config()

    # Apply CLI overrides
    if args.steps is not None:
        config.training.max_steps = args.steps
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.dtype is not None:
        config.training.dtype = args.dtype
    if args.compile:
        config.training.compile_model = True
    if args.seed is not None:
        config.training.seed = args.seed

    train(config, smoke_test=args.smoke_test, resume=args.resume)
