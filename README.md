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

## Notebooks

| Notebook | Algorithm | Risk Level | Memory Estimate |
|----------|-----------|------------|-----------------|
| [01_lora_sft_spark.ipynb](01_lora_sft_spark.ipynb) | QLoRA + SFT (4-bit) | Low | 2.8–3.9 GB |
| [02_sft_spark.ipynb](02_sft_spark.ipynb) | Full SFT | Medium | 44–57 GB |
| [03_osft_spark.ipynb](03_osft_spark.ipynb) | OSFT | High | 37–48 GB |

All three notebooks fine-tune [ibm-granite/granite-3.3-2b-instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) on the [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset (natural language to SQL).

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

- **Unsloth** — no pre-built wheels for ARM aarch64
- **DeepSpeed** — CUDA kernels don't support sm_121; may need `DS_BUILD_OPS=0`
- **flash-attn** — no ARM + Blackwell wheels
- **xformers** — no ARM wheels
- **liger-kernel** — Triton support on ARM is evolving
- **Compute capability 12.1** — PyTorch warns max supported is 12.0 (usually works anyway)

## Stack

- [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) 0.5.0
- [instructlab-training](https://github.com/instructlab/training) 0.14.1
- [PEFT](https://github.com/huggingface/peft) 0.18.1
- [TRL](https://github.com/huggingface/trl) 0.19.1
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 0.48.1

## Running

Open the notebooks in JupyterLab and run cells top-to-bottom. Some dependency install cells will fail — that's expected and documented. Run in order (01 → 02 → 03) for lowest to highest risk.
