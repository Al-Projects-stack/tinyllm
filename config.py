"""
config.py — All hyperparameters for model, training, generation, and paths.
Uses Python dataclasses with JSON serialisation/deserialisation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """GPT-style decoder-only transformer architecture settings."""
    vocab_size: int = 32000
    context_length: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    mlp_multiplier: int = 4
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Optimiser and training-loop knobs."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 10_000
    eval_interval: int = 500
    eval_steps: int = 50
    log_interval: int = 100
    checkpoint_interval: int = 1_000
    grad_accumulation_steps: int = 1
    compile_model: bool = False   # torch.compile for extra speed (PyTorch 2.0+)
    dtype: str = "bfloat16"       # "bfloat16" | "float16" | "float32"
    seed: int = 42


@dataclass
class GenerationConfig:
    """Text-generation defaults."""
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0


@dataclass
class PathConfig:
    """Filesystem locations."""
    data_dir: str = "data"
    tokenizer_path: str = "data/tokenizer"
    checkpoint_dir: str = "checkpoints"
    raw_data_path: str = "data/raw/tinystories.txt"
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    config_save_path: str = "checkpoints/config.json"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Combines all sub-configs.  Supports save/load via JSON."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = cls()
        cfg.model = ModelConfig(**data["model"])
        cfg.training = TrainingConfig(**data["training"])
        cfg.generation = GenerationConfig(**data["generation"])
        cfg.paths = PathConfig(**data["paths"])
        return cfg

    def __str__(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ---------------------------------------------------------------------------
# Quick sanity checks
# ---------------------------------------------------------------------------

def _assert_positive(name: str, val: int | float) -> None:
    if val <= 0:
        raise ValueError(f"{name} must be > 0, got {val}")


def validate(cfg: Config) -> None:
    mc = cfg.model
    tc = cfg.training
    _assert_positive("hidden_dim", mc.hidden_dim)
    _assert_positive("num_layers", mc.num_layers)
    _assert_positive("num_heads", mc.num_heads)
    if mc.hidden_dim % mc.num_heads != 0:
        raise ValueError(
            f"hidden_dim ({mc.hidden_dim}) must be divisible by "
            f"num_heads ({mc.num_heads})"
        )
    _assert_positive("batch_size", tc.batch_size)
    _assert_positive("learning_rate", tc.learning_rate)
    _assert_positive("max_steps", tc.max_steps)
    if tc.dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported dtype: {tc.dtype}")


# ---------------------------------------------------------------------------
# Convenience factory for smoke-test / quick experiments
# ---------------------------------------------------------------------------

def get_smoke_config() -> Config:
    """Minimal config for a fast smoke test without downloading any data."""
    cfg = Config()
    cfg.model.vocab_size = 256
    cfg.model.context_length = 64
    cfg.model.hidden_dim = 128
    cfg.model.num_layers = 2
    cfg.model.num_heads = 4
    cfg.model.dropout = 0.0
    cfg.training.batch_size = 16
    cfg.training.max_steps = 100
    cfg.training.warmup_steps = 10
    cfg.training.eval_interval = 50
    cfg.training.eval_steps = 10
    cfg.training.log_interval = 10
    cfg.training.checkpoint_interval = 999_999  # skip during smoke test
    cfg.training.grad_accumulation_steps = 1
    cfg.training.compile_model = False
    cfg.training.dtype = "float32"  # bfloat16 not always available on CPU
    return cfg


if __name__ == "__main__":
    cfg = Config()
    validate(cfg)
    print(cfg)
    print(f"\nSmoke config:\n{get_smoke_config()}")
