# Tiny LLM — GPT from scratch in pure PyTorch

A complete, ~10 M-parameter decoder-only transformer trained from scratch.
Targets a free-tier Colab T4 GPU (16 GB VRAM).

---

## Project layout

```
tiny-llm/
├── config.py            # All hyperparameters (dataclasses + JSON I/O)
├── model.py             # Pure-PyTorch GPT (RMSNorm, SwiGLU, causal attention)
├── data.py              # Dataset download, tokenisation, DataLoader
├── train_tokenizer.py   # Train & save a BPE tokenizer
├── train.py             # Training loop (mixed precision, cosine LR, checkpoints)
├── inference.py         # Autoregressive generation, interactive CLI
├── requirements.txt
├── Dockerfile
└── data/                # Created at runtime
    ├── raw/
    │   └── tinystories.txt
    ├── tokenizer/
    │   ├── tokenizer.json
    │   └── vocab.txt
    ├── train.bin
    └── val.bin
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Google Colab:** paste these commands in a cell, adding `!` prefix.

---

## Step-by-step: train from scratch

### 3. Download the dataset

```bash
python data.py
```

This downloads **TinyStories** (~300 MB) from HuggingFace and writes it to
`data/raw/tinystories.txt`.  Requires an internet connection on first run.

To limit the download for experimentation:

```bash
python data.py --max-stories 20000
```

### 4. Train the BPE tokenizer

```bash
python train_tokenizer.py \
    --input data/raw/tinystories.txt \
    --tokenizer-dir data/tokenizer \
    --vocab-size 32000
```

Trains a 32 k-token BPE tokenizer (~5–10 min on full dataset).  For a quick
test pass `--max-chars 5000000` to use only the first 5 M characters.

### 5. Tokenise the corpus

Re-run the data pipeline (now that the tokenizer exists):

```bash
python data.py
```

This encodes the full text and saves `data/train.bin` and `data/val.bin` as
uint16 arrays for memory-efficient loading.

### 6. Train the model

```bash
python train.py --steps 10000
```

Checkpoints are saved to `checkpoints/`.  Resume training with:

```bash
python train.py --steps 20000 --resume checkpoints/step_010000.pt
```

Additional options:

| Flag | Default | Description |
|------|---------|-------------|
| `--steps N` | 10 000 | Total training steps |
| `--batch-size N` | 32 | Micro-batch size |
| `--lr F` | 3e-4 | Peak learning rate |
| `--dtype STR` | bfloat16 | Training dtype (`bfloat16/float16/float32`) |
| `--compile` | off | Enable `torch.compile` for ~20 % speedup |
| `--seed N` | 42 | Random seed |

### 7. Generate text

```bash
# Interactive mode
python inference.py --checkpoint checkpoints/best.pt

# Single prompt
python inference.py \
    --checkpoint checkpoints/best.pt \
    --prompt "Once upon a time there was a little girl" \
    --max-new-tokens 300 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95

# Greedy decoding
python inference.py --checkpoint checkpoints/best.pt \
    --prompt "Once upon a time" --temperature 0
```

---

## Smoke test (no dataset required)

Verify the training loop works end-to-end in ~30 seconds on CPU:

```bash
python train.py --smoke-test --steps 100
```

Expected output: loss should decrease, ending with:
```
[smoke] PASS — loss decreased: X.XXXX → Y.YYYY ✓
```

---

## Expected metrics

| Steps | Train loss | Val loss | Notes |
|-------|-----------|----------|-------|
| 1 000 | ~3.5 | ~3.5 | Still mostly random |
| 5 000 | ~2.5 | ~2.6 | Learning grammar |
| 10 000 | ~2.1 | ~2.2 | Coherent short sentences |
| 50 000 | ~1.8 | ~1.9 | Stories with plot |

Val loss ~2.0–2.3 is a realistic target for this model size on TinyStories.

---

## Model architecture

```
GPT (11.4 M params)
├── TokenEmbedding    32000 × 256
├── PositionEmbedding  128  × 256
├── TransformerBlock × 3
│   ├── RMSNorm(256)
│   ├── CausalSelfAttention (4 heads, head_dim=64)
│   │   ├── QKV projection  256 → 768  (no bias)
│   │   └── Out projection  256 → 256  (no bias)
│   ├── RMSNorm(256)
│   └── SwiGLU MLP
│       ├── gate_proj  256 → 1024
│       ├── up_proj    256 → 1024
│       └── down_proj 1024 →  256
├── RMSNorm(256)
└── LMHead  256 → 32000  (weight-tied to TokenEmbedding)
```

---

## Hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| Vocab size | 32 000 |
| Context length | 128 |
| Hidden dim | 256 |
| Layers | 3 |
| Attention heads | 4 |
| MLP multiplier | 4× |
| Dropout | 0.1 |
| Batch size | 32 |
| Peak LR | 3 × 10⁻⁴ |
| LR schedule | Cosine + linear warmup |
| Warmup steps | 100 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Mixed precision | bfloat16 |

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size` (try 8 or 16)
- Enable gradient accumulation: edit `grad_accumulation_steps` in `config.py`
- Reduce `context_length` (e.g. 64) for debugging
- Use `--dtype float16` (slightly more memory-efficient than bfloat16 on some GPUs)

### NaN loss
- Lower the learning rate (`--lr 1e-4`)
- Check for empty batches in data pipeline
- Try `--dtype float32` to rule out half-precision overflow
- Ensure gradient clipping is active (`grad_clip=1.0`)

### Slow training
- Enable `--compile` (torch.compile, requires PyTorch 2.0+)
- Increase batch size (if VRAM allows)
- Set `num_workers > 0` in `create_dataloaders` (Linux / macOS only)
- Reduce `eval_steps` in config

### Loss not decreasing past ~3.5
- Verify the tokenizer was trained before `encode_file` was run
- Check `data/train.bin` is non-empty (`ls -lh data/`)
- Increase warmup steps if LR spikes early

### "Tokenizer not found" error
- Run `python train_tokenizer.py` before `python train.py`
- Alternatively, `data.py` automatically falls back to the built-in
  **character-level tokenizer** (vocab ≈ 100 chars) if no BPE file is found.
  Quality is lower but the pipeline runs without any prior tokenizer training.

### "Token file not found" error
- Run `python data.py` after training the tokenizer

---

## Docker (reproducible GPU environment)

```bash
# Build
docker build -t tiny-llm .

# Train (mount data and checkpoints)
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    tiny-llm python train.py --steps 10000

# Inference
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -it tiny-llm python inference.py \
        --checkpoint checkpoints/best.pt \
        --prompt "Once upon a time"
```

---

## License

MIT — feel free to use and adapt.
