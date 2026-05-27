"""
data.py — Dataset download, cleaning, tokenisation, PyTorch Dataset & DataLoader.

Pipeline:
  1. Download TinyStories from HuggingFace datasets (train split ~300 MB).
  2. Clean / concatenate text into one flat text file.
  3. Tokenise with a pre-trained BPE tokenizer (see train_tokenizer.py).
  4. Save uint16 numpy arrays (train.bin, val.bin).
  5. Serve batches via TextDataset (memory-mapped) + DataLoader.

For smoke tests, use SmokeDataset which generates deterministic in-memory data.
"""

from __future__ import annotations

import os
import random
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config, PathConfig


# ---------------------------------------------------------------------------
# Step 1 — Download raw text
# ---------------------------------------------------------------------------

def download_tinystories(paths: PathConfig, max_stories: Optional[int] = None) -> str:
    """
    Download the TinyStories dataset (roneneldan/TinyStories on HuggingFace)
    and concatenate all stories into a single text file.

    Returns the path to the raw text file.
    """
    raw_path = Path(paths.raw_data_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        size_mb = raw_path.stat().st_size / 1e6
        print(f"[data] Raw text already exists at {raw_path} ({size_mb:.1f} MB)")
        return str(raw_path)

    print("[data] Downloading TinyStories from HuggingFace...")
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=False)

        print(f"[data] Writing {len(ds):,} stories to {raw_path} ...")
        with open(raw_path, "w", encoding="utf-8") as fh:
            for i, example in enumerate(ds):
                if max_stories is not None and i >= max_stories:
                    break
                text = example.get("text", "").strip()
                if text:
                    fh.write(text + "\n\n")
                if (i + 1) % 50_000 == 0:
                    print(f"  {i+1:,} stories written...")

        size_mb = raw_path.stat().st_size / 1e6
        print(f"[data] Done. Raw file: {raw_path} ({size_mb:.1f} MB)")

    except Exception as exc:
        raise RuntimeError(
            f"Failed to download TinyStories: {exc}\n"
            "Ensure `datasets` and `huggingface-hub` are installed, or place a "
            f"raw text file at {raw_path} manually."
        ) from exc

    return str(raw_path)


# ---------------------------------------------------------------------------
# Step 2 — Clean text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Light cleaning:
      • Collapse repeated blank lines (keep paragraph breaks)
      • Strip leading/trailing whitespace
      • Remove null bytes
    """
    import re
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 3 — Tokenise & save binary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Character-level fallback tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """
    Minimal character-level tokenizer used as a fallback when no BPE
    tokenizer.json is present.

    Vocabulary: all printable ASCII characters (0x20–0x7E) plus newline,
    plus four special tokens: [UNK], [PAD], [BOS], [EOS].
    The vocab is fixed and reproducible (no training required).
    """

    # Special tokens — assigned to indices 0-3
    _SPECIALS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

    def __init__(self) -> None:
        # Printable ASCII: space (32) through tilde (126) + newline
        chars = [chr(i) for i in range(32, 127)] + ["\n"]
        self._vocab: dict[str, int] = {}
        for i, tok in enumerate(self._SPECIALS):
            self._vocab[tok] = i
        for ch in chars:
            if ch not in self._vocab:
                self._vocab[ch] = len(self._vocab)
        self._id2tok = {v: k for k, v in self._vocab.items()}

    # HuggingFace-compatible duck-type API --------------------------------

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def get_vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, text: str) -> "_CharEncoding":
        unk = self._vocab["[UNK]"]
        ids = [self._vocab.get(ch, unk) for ch in text]
        return _CharEncoding(ids)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        specials = set(self._SPECIALS) if skip_special_tokens else set()
        chars = []
        for i in ids:
            tok = self._id2tok.get(i, "")
            if tok not in specials:
                chars.append(tok)
        return "".join(chars)

    # Disable padding / truncation (match tokenizers API stubs) ----------

    def no_padding(self) -> None:
        pass  # no-op — no padding in char tokenizer

    def no_truncation(self) -> None:
        pass  # no-op — no truncation in char tokenizer

    def enable_padding(self, **kwargs) -> None:
        pass  # no-op


class _CharEncoding:
    """Minimal stand-in for tokenizers.Encoding."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


def load_tokenizer(tokenizer_path: str):
    """
    Load a pre-trained BPE tokenizer saved by train_tokenizer.py.

    Falls back automatically to the built-in CharTokenizer when
    tokenizer.json is not present, allowing the full pipeline to run
    on raw text without first training a BPE model.
    """
    tok_file = Path(tokenizer_path) / "tokenizer.json"
    if tok_file.exists():
        from tokenizers import Tokenizer  # type: ignore
        return Tokenizer.from_file(str(tok_file))
    else:
        print(
            f"[data] WARNING: BPE tokenizer not found at {tok_file}. "
            "Falling back to built-in character-level tokenizer. "
            "Run `python train_tokenizer.py` for better quality."
        )
        return CharTokenizer()


def encode_file(
    raw_path: str,
    tokenizer,
    out_train: str,
    out_val: str,
    val_fraction: float = 0.005,
    chunk_size: int = 100_000,
) -> tuple[int, int]:
    """
    Read raw text, encode with BPE tokenizer, split into train/val, save as
    uint16 numpy arrays (.bin files).  Returns (n_train_tokens, n_val_tokens).
    """
    out_train_path = Path(out_train)
    out_val_path   = Path(out_val)
    out_train_path.parent.mkdir(parents=True, exist_ok=True)
    out_val_path.parent.mkdir(parents=True, exist_ok=True)

    if out_train_path.exists() and out_val_path.exists():
        n_train = out_train_path.stat().st_size // 2  # uint16 = 2 bytes
        n_val   = out_val_path.stat().st_size // 2
        print(f"[data] Tokenised data exists: {n_train:,} train, {n_val:,} val tokens")
        return n_train, n_val

    print(f"[data] Tokenising {raw_path} ...")
    all_ids: list[int] = []

    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    tokenizer.no_padding()
    tokenizer.no_truncation()

    raw_path_obj = Path(raw_path)
    texts: list[str] = []

    with open(raw_path_obj, "r", encoding="utf-8") as fh:
        buf: list[str] = []
        for line in fh:
            buf.append(line)
            if len(buf) >= chunk_size:
                chunk = "".join(buf)
                chunk = clean_text(chunk)
                enc = tokenizer.encode(chunk)
                all_ids.extend(enc.ids)
                buf = []
                print(f"  encoded {len(all_ids):,} tokens so far...")
        if buf:
            chunk = clean_text("".join(buf))
            enc = tokenizer.encode(chunk)
            all_ids.extend(enc.ids)

    total = len(all_ids)
    n_val = max(1, int(total * val_fraction))
    n_train = total - n_val

    arr = np.array(all_ids, dtype=np.uint16)
    arr[:n_train].tofile(out_train)
    arr[n_train:].tofile(out_val)

    print(f"[data] Saved {n_train:,} train tokens → {out_train}")
    print(f"[data] Saved {n_val:,} val tokens   → {out_val}")
    return n_train, n_val


# ---------------------------------------------------------------------------
# Step 4 — PyTorch Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Memory-mapped dataset over a flat uint16 token-ID file.

    Each sample is a context_length-token window; the target is the same
    window shifted right by one position.

    Uses np.memmap so large files are never fully loaded into RAM.
    """

    def __init__(self, bin_path: str, context_length: int) -> None:
        self.context_length = context_length
        path = Path(bin_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Token bin file not found: {bin_path}\n"
                "Run the data preparation pipeline first."
            )
        n_bytes = path.stat().st_size
        n_tokens = n_bytes // 2  # uint16
        if n_tokens <= context_length:
            raise ValueError(
                f"Token file has only {n_tokens} tokens; need > {context_length}"
            )
        self._data = np.memmap(str(path), dtype=np.uint16, mode="r")
        self._n_samples = len(self._data) - context_length

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.from_numpy(
            self._data[idx : idx + self.context_length + 1].astype(np.int64)
        )
        x = chunk[:-1]   # input:  tokens [0 … T-1]
        y = chunk[1:]    # target: tokens [1 … T]
        return x, y


# ---------------------------------------------------------------------------
# Smoke / test dataset (no files needed)
# ---------------------------------------------------------------------------

class SmokeDataset(Dataset):
    """
    Deterministic in-memory dataset for smoke tests and unit tests.

    Generates `n_samples` identical sequences so the model can easily overfit
    and we can verify loss is decreasing.

    The pattern is:  position i → token (i * 7 + 13) % vocab_size
    so adjacent tokens have a learnable relationship.
    """

    def __init__(
        self,
        context_length: int,
        vocab_size: int,
        n_samples: int = 512,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Create a random but fixed sequence (same for every sample)
        base = rng.integers(0, vocab_size, size=context_length + 1).astype(np.int64)
        self.data = torch.from_numpy(
            np.tile(base, (n_samples, 1))  # (n_samples, context_length+1)
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data[idx]
        return row[:-1], row[1:]


# ---------------------------------------------------------------------------
# Step 5 — DataLoaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    config: Config,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from pre-tokenised .bin files."""
    train_ds = TextDataset(config.paths.train_data_path, config.model.context_length)
    val_ds   = TextDataset(config.paths.val_data_path,   config.model.context_length)

    print(f"[data] Train dataset: {len(train_ds):,} samples")
    print(f"[data] Val   dataset: {len(val_ds):,} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def create_smoke_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """Create tiny in-memory DataLoaders for smoke testing."""
    train_ds = SmokeDataset(
        context_length=config.model.context_length,
        vocab_size=config.model.vocab_size,
        n_samples=256,
        seed=0,
    )
    val_ds = SmokeDataset(
        context_length=config.model.context_length,
        vocab_size=config.model.vocab_size,
        n_samples=64,
        seed=1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def prepare_data(config: Config, max_stories: Optional[int] = None) -> None:
    """
    Run the full data-preparation pipeline:
      1. Download raw text
      2. Train tokenizer (if not already done)
      3. Encode and save .bin files
    """
    raw_path = download_tinystories(config.paths, max_stories=max_stories)
    tokenizer = load_tokenizer(config.paths.tokenizer_path)
    encode_file(
        raw_path=raw_path,
        tokenizer=tokenizer,
        out_train=config.paths.train_data_path,
        out_val=config.paths.val_data_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from config import Config

    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--max-stories", type=int, default=None,
                        help="Limit number of stories downloaded (for testing)")
    args = parser.parse_args()

    cfg = Config()
    prepare_data(cfg, max_stories=args.max_stories)
