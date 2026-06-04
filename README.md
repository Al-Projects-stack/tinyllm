# Tiny LLM — GPT from scratch in pure PyTorch

A complete, ~11M-parameter decoder-only transformer trained from scratch on the
TinyStories (https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
No Hugging Face Trainer, no Lightning, no Accelerate — every line is plain PyTorch.


**Status:** Training complete for tinyllm (smoke-tested; model training loop validated).

---

## What you are building and learning

This project walks through every layer of a modern LLM, from raw text to generated
stories. Here is what each piece teaches you:

### 1. Tokenization (`train_tokenizer.py`)
I’m learning to convert raw text to numbers.  

A **Byte-Pair Encoding (BPE)** tokenizer is trained directly on TinyStories. It
starts with individual characters and repeatedly merges the most frequent adjacent
pair until it has a 32,000-token vocabulary. The result: common words ("the", "once")
become single tokens; rare words are split into subwords. This is exactly how GPT-2,
LLaMA, and Mistral handle text.

### 2. Data pipeline (`data.py`)
I’m learning to feed the GPU efficiently.  


The 1.84 GB text file is encoded once into a flat array of token IDs and saved as a
binary file (`train.bin`). During training, a memory-mapped `numpy` array lets
PyTorch read random windows without loading everything into RAM. Each sample is a
128-token window; the target is the same window shifted one position to the right
— the model must predict the next token at every position simultaneously.

### 3. Model architecture (`model.py`)
Transformer block used by every modern LLM.

```

Input tokens (B, T)
      |
Token Embedding (32000 x 256)  +  Positional Embedding (128 x 256)
      |
  x 3  TransformerBlock
  |     ├── RMSNorm          — normalise without mean subtraction (faster than LayerNorm)
  |     ├── CausalSelfAttention
  |     |     ├── QKV projection  256 -> 768  (fused, no bias)
  |     |     ├── scaled_dot_product_attention  (causal mask, FlashAttention when available)
  |     |     └── Output projection  256 -> 256
  |     └── SwiGLU MLP       — gated activation used in LLaMA / Mistral
  |           ├── gate_proj  256 -> 1024
  |           ├── up_proj    256 -> 1024
  |           └── down_proj 1024 -> 256
      |
Final RMSNorm
      |
LM Head  256 -> 32000  (weight-tied to token embedding — saves 8M params)
```

This is what I’m working on:


- **Causal masking** — each position can only attend to itself and the past, making
  generation possible one token at a time.
- **Weight tying** — the embedding matrix and the output projection share the same
  weights. This halves the parameter count for large vocabularies and improves
  training stability.
- **RMSNorm** — simpler and faster than LayerNorm; used by LLaMA and Mistral.
- **SwiGLU** — a gated MLP that outperforms plain ReLU/GELU; the gate controls how
  much information flows through the up-projection.
- **KV-cache** — during inference, key/value tensors from past tokens are cached so
  each new token only requires one forward pass through the new position.

### 4. Training loop (`train.py`)
I’m learning engineering for stable, fast training.


- **AdamW with decoupled weight decay** — weight decay is not applied to embeddings,

  norms, or biases (those don't benefit from it).
- **Cosine LR schedule with linear warmup** — the learning rate ramps up for 100
  steps to avoid large early updates, then follows a cosine curve down to 10% of
  peak.
- **Gradient clipping** — gradients are clipped to norm 1.0, preventing the
  exploding gradient problem common in deep networks.
- **Mixed precision (float16 on GTX 1650)** — weights and activations are stored in
  16-bit floats, halving memory and speeding up GPU compute. A `GradScaler` prevents
  underflow in the backward pass.
- **Gradient accumulation** — multiple micro-batches can be summed before an
  optimiser step, simulating a larger effective batch size without extra VRAM.

### 5. Inference (`inference.py`)
This is what I’m doing: generating text with a trained model.


- **Autoregressive decoding** — the model predicts one token at a time, appends it
  to the context, and feeds the extended sequence back in.
- **Temperature** — divides logits before softmax. Low temperature (< 1) makes the
  distribution sharper (more predictable); high temperature (> 1) flattens it
  (more creative/random).
- **Top-k sampling** — keeps only the k most likely next tokens, zeroing out the
  rest. Prevents the model from sampling very unlikely tokens.
- **Top-p (nucleus) sampling** — keeps the smallest set of tokens whose cumulative
  probability exceeds p. Adapts the cutoff dynamically to the shape of the
  distribution.
- **Repetition penalty** — reduces the logit of any token that has already appeared,
  discouraging the model from repeating itself.

---

## Project layout

```
tinyllm/
├── config.py            # All hyperparameters (dataclasses + JSON I/O)
├── model.py             # Pure-PyTorch GPT (RMSNorm, SwiGLU, causal attention, KV-cache)
├── data.py              # Dataset download, BPE tokenisation, DataLoader + char fallback
├── train_tokenizer.py   # Train & save 32k BPE tokenizer on raw text
├── train.py             # Training loop (mixed precision, cosine LR, grad clip, checkpoints)
├── inference.py         # Autoregressive generation, sampling utils, interactive CLI
├── setup_gpu.bat        # One-click CUDA PyTorch installer for Windows
├── requirements.txt     # Pinned dependencies
├── Dockerfile           # Reproducible GPU environment (CUDA 12.1)
└── data/                # Created at runtime (not in git)
    ├── raw/tinystories.txt      # 1.84 GB raw text
    ├── tokenizer/tokenizer.json # 32k BPE vocab
    ├── train.bin                # 464M tokens (uint16)
    └── val.bin                  # 2.3M tokens (uint16)
```

---

## Setup

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA (GTX 1650 or better) — or CPU for smoke tests

### Install dependencies

**GPU (Windows, GTX 1650 / any Turing+ GPU):**
```bat
setup_gpu.bat
pip install -r requirements.txt
```

**CPU only:**
```bash
pip install -r requirements.txt
```

---

## Train from scratch

### Step 1 — Download TinyStories (~300 MB download, 1.84 GB on disk)
```bash
python data.py
```

### Step 2 — Train BPE tokenizer (~10 min on full dataset)
```bash
python train_tokenizer.py --input data/raw/tinystories.txt --vocab-size 32000
```

### Step 3 — Encode corpus to binary
```bash
python data.py
```

### Step 4 — Train
```bash
# GPU
python train.py --steps 10000

# Resume from checkpoint
python train.py --steps 10000 --resume checkpoints/step_005000.pt

# CPU smoke test (no data needed, ~30 sec)
python train.py --smoke-test --steps 100
```

---

## Generate text

```bash
# Interactive mode
python inference.py --checkpoint checkpoints/step_010000.pt

# Single prompt
python inference.py --checkpoint checkpoints/step_010000.pt \
    --prompt "Once upon a time there was a little girl" \
    --max-new-tokens 300 --temperature 0.8 --top-k 50

# Greedy (deterministic)
python inference.py --checkpoint checkpoints/step_010000.pt \
    --prompt "Once upon a time" --temperature 0
```

---

## Expected training metrics

| Steps | Train loss | Val loss | What the model can do |
|-------|-----------|----------|-----------------------|
| 500   | ~3.8      | ~3.9     | Learns basic token frequencies |
| 1 000 | ~3.2      | ~3.3     | Starts forming real words |
| 2 500 | ~2.7      | ~2.8     | Short grammatical phrases |
| 5 000 | ~2.4      | ~2.5     | Simple sentences with story structure |
| 10 000| ~2.1      | ~2.2     | Coherent short stories |

Cross-entropy loss of 2.0–2.3 nats = perplexity of ~7–10, which is a realistic
target for an 11M-parameter model on this dataset.

---

## Model hyperparameters

| Parameter       | Value  | Why |
|-----------------|--------|-----|
| Vocab size      | 32 000 | BPE; covers TinyStories with ~2.5 tokens/word |
| Context length  | 128    | Fits in 4 GB VRAM comfortably; most stories are < 128 tokens |
| Hidden dim      | 256    | Scales parameter count to ~11M |
| Layers          | 3      | Enough depth to learn grammar and simple reasoning |
| Attention heads | 4      | Head dim = 64, a standard efficient size |
| MLP multiplier  | 4x     | Inner dim 1024; standard for SwiGLU |
| Dropout         | 0.1    | Light regularisation |
| Batch size      | 32     | ~4k tokens/batch; fits in 4 GB VRAM with float16 |
| Peak LR         | 3e-4   | AdamW standard for transformers |
| Warmup steps    | 100    | Prevents large early updates |
| Weight decay    | 0.1    | Applied only to weight matrices, not embeddings/norms |
| Grad clip       | 1.0    | Prevents exploding gradients |
| dtype           | float16| GTX 1650 (Turing) — bfloat16 requires Ampere+ |

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size` (try 8 or 16)
- Use `--dtype float16` if on GPU
- Reduce `context_length` in `config.py`

### NaN loss
- Lower the learning rate: `--lr 1e-4`
- Try `--dtype float32` to rule out half-precision overflow
- Gradient clipping is already on (`grad_clip=1.0`)

### Loss stuck above 3.5
- Verify tokenizer was trained before encoding: `data/tokenizer/tokenizer.json` must exist
- Check `data/train.bin` is non-empty
- Try more warmup steps

### "Tokenizer not found"
- Run `python train_tokenizer.py` first
- Or let `data.py` fall back to the built-in character-level tokenizer (lower quality)

### CUDA not available on Windows
- Run `setup_gpu.bat` to install the CUDA-enabled PyTorch wheel
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

---

## What to try next

Once training finishes and you can generate stories, here are natural extensions:

1. **Scale up** — double the hidden dim to 512, layers to 6 → ~50M params. Needs more VRAM or gradient checkpointing.
2. **Rotary embeddings (RoPE)** — replace absolute positional embeddings with RoPE for better length generalisation (used in LLaMA).
3. **Grouped-query attention (GQA)** — share K/V heads across Q heads to reduce KV-cache memory (used in LLaMA 3, Mistral).
4. **torch.compile** — add `--compile` flag; gives ~15–30% speedup on PyTorch 2.x.
5. **Longer context** — increase `context_length` to 256 or 512 to see richer story generation.
6. **Fine-tuning** — load a trained checkpoint and continue training on a custom dataset (instructions, dialogues, code).

---

## License

MIT — build on it freely.
