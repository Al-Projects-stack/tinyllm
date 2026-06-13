"""
train_tokenizer.py — Train a Byte-Pair Encoding (BPE) tokenizer on raw text.

Uses the HuggingFace `tokenizers` library (NOT `transformers`).
Saves tokenizer.json to --tokenizer-dir for later use by data.py and train.py.

Usage:
    # Train from TinyStories text file (download first via data.py)
    python train_tokenizer.py --input data/raw/tinystories.txt --vocab-size 32000

    # Quick test on a small file
    python train_tokenizer.py --input data/raw/tinystories.txt --vocab-size 1000 --max-chars 5000000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def train_bpe_tokenizer(
    input_path: str,
    save_dir: str,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    max_chars: int | None = None,
    special_tokens: list[str] | None = None,
    force: bool = False,
) -> None:
    """
    Train a BPE tokenizer on the given text file and save it.

    Args:
        input_path:    Path to a UTF-8 text file (one story / document per paragraph).
        save_dir:      Directory to save tokenizer.json and vocab.txt.
        vocab_size:    Target vocabulary size including special tokens.
        min_frequency: Minimum pair frequency for a merge to be learned.
        max_chars:     If set, only use the first `max_chars` characters for training
                       (useful for quick experiments).
        special_tokens: List of special token strings. Defaults to standard set.
    """
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
        from tokenizers.normalizers import NFD, Lowercase, Sequence as NormSeq, StripAccents
    except ImportError as exc:
        sys.exit(f"[tokenizer] ERROR: `tokenizers` library not installed. "
                 f"Run `pip install tokenizers`. Details: {exc}")

    if special_tokens is None:
        special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]

    input_path = str(input_path)
    if not Path(input_path).exists():
        sys.exit(f"[tokenizer] Input file not found: {input_path}")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    tok_json = save_path / "tokenizer.json"
    if tok_json.exists():
        if not force:
            print(f"[tokenizer] Tokenizer already exists at {tok_json}. Use --force to retrain.")
            return
        print(f"[tokenizer] --force set: overwriting existing tokenizer at {tok_json}.")

    # ------------------------------------------------------------------
    # Build tokenizer
    # ------------------------------------------------------------------
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Pre-tokeniser: split on whitespace + punctuation (byte-level aware)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Byte-level decoder to reconstruct text from tokens
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor: add BOS/EOS
    bos_id = special_tokens.index("[BOS]")
    eos_id = special_tokens.index("[EOS]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[BOS]:0 $A:0 [EOS]:0",
        pair=f"[BOS]:0 $A:0 [SEP]:0 $B:0 [EOS]:0",
        special_tokens=[
            ("[BOS]", bos_id),
            ("[EOS]", eos_id),
            ("[SEP]", special_tokens.index("[SEP]")),
        ],
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # ------------------------------------------------------------------
    # Prepare training data
    # ------------------------------------------------------------------
    if max_chars is not None:
        # Write a trimmed temporary file to avoid loading all text into RAM
        import tempfile
        print(f"[tokenizer] Using first {max_chars:,} chars for training ...")
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                          delete=False, encoding="utf-8")
        try:
            with open(input_path, "r", encoding="utf-8") as fh:
                text = fh.read(max_chars)
            tmp.write(text)
            tmp.close()
            training_files = [tmp.name]
            _train(tokenizer, trainer, training_files)
        finally:
            os.unlink(tmp.name)
    else:
        training_files = [input_path]
        _train(tokenizer, trainer, training_files)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    tokenizer.save(str(tok_json))
    print(f"[tokenizer] Saved tokenizer ({tokenizer.get_vocab_size()} tokens) -> {tok_json}")

    # Also save a human-readable vocab list for inspection
    vocab = tokenizer.get_vocab()
    vocab_txt = save_path / "vocab.txt"
    with open(vocab_txt, "w", encoding="utf-8") as fh:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            fh.write(f"{idx}\t{token}\n")
    print(f"[tokenizer] Saved vocab list -> {vocab_txt}")


def _train(tokenizer, trainer, files: list[str]) -> None:
    """Call tokenizer.train_from_iterator or train depending on file size."""
    print(f"[tokenizer] Training BPE on {len(files)} file(s) ...")
    tokenizer.train(files=files, trainer=trainer)


def load_tokenizer(tokenizer_dir: str):
    """Load a previously saved tokenizer."""
    from tokenizers import Tokenizer  # type: ignore
    tok_json = Path(tokenizer_dir) / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(f"No tokenizer found at {tok_json}")
    return Tokenizer.from_file(str(tok_json))


def encode_text(text: str, tokenizer_dir: str) -> list[int]:
    """Convenience: load tokenizer and encode a string."""
    tok = load_tokenizer(tokenizer_dir)
    return tok.encode(text).ids


def decode_ids(ids: list[int], tokenizer_dir: str) -> str:
    """Convenience: load tokenizer and decode a list of ids."""
    tok = load_tokenizer(tokenizer_dir)
    return tok.decode(ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on a raw text corpus."
    )
    parser.add_argument(
        "--input", type=str, default="data/raw/tinystories.txt",
        help="Path to raw UTF-8 text file.",
    )
    parser.add_argument(
        "--tokenizer-dir", type=str, default="data/tokenizer",
        help="Directory to save tokenizer artefacts.",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32_000,
        help="Target vocabulary size.",
    )
    parser.add_argument(
        "--min-frequency", type=int, default=2,
        help="Minimum pair frequency to learn a merge.",
    )
    parser.add_argument(
        "--max-chars", type=int, default=None,
        help="Only use the first N characters for training (quick test).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing tokenizer.json instead of skipping.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_bpe_tokenizer(
        input_path=args.input,
        save_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_chars=args.max_chars,
        force=args.force,
    )
