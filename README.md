# Training Hub on DGX Spark

Testing [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) fine-tuning algorithms on an NVIDIA DGX Spark — documenting what works, what breaks, and what's needed to get it running on ARM + Blackwell.

## Hardware

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GB10 (compute capability 12.1) |
| Memory | 130.7 GB unified CPU/GPU |
| CPU | ARM aarch64 (Grace) |
| CUDA | 13.0 |
| PyTorch | 2.10.0+cu130 |

## Results

| Notebook | Algorithm | Memory Estimate | Status | Notes |
|----------|-----------|-----------------|--------|-------|
| [01_lora_sft_spark.ipynb](01_lora_sft_spark.ipynb) | QLoRA + SFT (4-bit) | 2.8–3.9 GB | **Passed** | Unsloth backend worked out of the box |
| [02_sft_spark.ipynb](02_sft_spark.ipynb) | Full SFT | 44–57 GB | **Not run** | DeepSpeed installed but training not yet executed |
| [03_osft_spark.ipynb](03_osft_spark.ipynb) | OSFT | 34.5–44.8 GB | **Failed** | `mini-trainer` hard-requires `flash_attn`, which can't build on this platform |

All three notebooks fine-tune [ibm-granite/granite-3.3-2b-instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) on the [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset (natural language to SQL, 2000 examples).

### Details

**01 — LoRA+SFT (QLoRA):** The only method that completed successfully. Unsloth 2026.3.4 installed and imported without issues on ARM aarch64. Training Hub's `lora_sft()` ran with 4-bit quantization, LoRA r=16, batch size 8, 1 epoch on 2000 examples. All dependencies (unsloth, peft, bitsandbytes, trl, xformers) were available.

**02 — Full SFT:** DeepSpeed 0.18.7 installed successfully (built from source with no pre-compiled CUDA ops — JIT compilation at runtime). flash-attn failed to build (CUDA toolkit 12.0 vs PyTorch CUDA 13.0 mismatch). The training cell was not executed yet.

**03 — OSFT:** Failed at model setup. The `mini-trainer` backend unconditionally imports `flash_attn` in `setup_model_for_training.py:945`, and flash-attn cannot be compiled on this platform. liger-kernel 0.7.0 installed but had no `__version__` attribute. FSDP (PyTorch built-in) and Triton 3.6.0 were available.

### Blocking Issues

| Issue | Affects | Root Cause | Potential Fix |
|-------|---------|------------|---------------|
| `flash-attn` won't build | OSFT, possibly Full SFT | CUDA toolkit 12.0 vs PyTorch CUDA 13.0 mismatch | Upstream: make `flash_attn` import optional in `mini-trainer` |
| `mini-trainer` hard-requires `flash_attn` | OSFT | Unconditional import in `setup_model_for_training.py` | Patch to fall back to SDPA attention |

## What's Inside

Each notebook follows the same narrative structure:

1. **System Profile** — hardware specs, CUDA version, compute capability
2. **Environment Check** — installed packages, dependency install attempts with full output
3. **Memory Estimation** — Training Hub's `estimate()` to check if it fits
4. **Dataset Preparation** — download, convert to JSONL messages format, preview
5. **Training Execution** — the actual training call with timing
6. **Troubleshooting Log** — what went wrong, what workarounds were tried
7. **Results** — training time, peak memory, observations

## Key Challenges on DGX Spark

The DGX Spark is bleeding-edge hardware. Expect these dependency issues:

- **Unsloth** — installed and works (2026.3.4)
- **DeepSpeed** — installs (0.18.7) but no pre-compiled CUDA ops; JIT compiles at runtime
- **flash-attn** — **won't build** (CUDA toolkit 12.0 vs PyTorch CUDA 13.0 mismatch) — this is the main blocker
- **xformers** — installed and works (0.0.35)
- **liger-kernel** — installed (0.7.0) but missing `__version__` attribute
- **Compute capability 12.1** — PyTorch warns max supported is 12.0 (works anyway)

## Stack

- [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) 0.5.0
- [instructlab-training](https://github.com/instructlab/training) 0.14.1
- [PEFT](https://github.com/huggingface/peft) 0.18.1
- [TRL](https://github.com/huggingface/trl) 0.19.1
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 0.48.1

## Running

Open the notebooks in JupyterLab and run cells top-to-bottom. Some dependency install cells will fail — that's expected and documented. Run in order (01 → 02 → 03) for lowest to highest risk.
