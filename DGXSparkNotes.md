# DGX Spark Notes

Workarounds and fixes needed to run [Training Hub](https://github.com/instructlab/training-hub) on the NVIDIA DGX Spark.

## System Profile

| Component | Value |
|-----------|-------|
| Platform | aarch64 (ARM Grace CPU) |
| GPU | NVIDIA GB10 |
| GPU Memory | 121.7 GB (unified/shared with CPU) |
| Compute Capability | 12.1 |
| CUDA Toolkit | 13.0 |
| OS | Ubuntu (Linux 6.17.0-1008-nvidia) |
| PyTorch | 2.10.0+cu130 |
| Transformers | 5.2.0 |
| Unsloth | 2026.3.4 |

## Issues & Fixes

### 1. `report_to=None` crashes with transformers 5.2.0

**Symptom:**
```
ValueError: None is not supported, only azure_ml, comet_ml, mlflow, neptune, tensorboard, ...
```

**Cause:** `training_hub/algorithms/lora.py` line 314 passes Python `None` when no W&B project is configured. Older transformers treated `None` as "no reporting"; transformers 5.2.0 requires the string `"none"`.

**Fix:** Patch the installed library:
```python
# File: .venv/lib/python3.12/site-packages/training_hub/algorithms/lora.py:314
# Before:
report_to="wandb" if params.get('wandb_project') else None,
# After:
report_to="wandb" if params.get('wandb_project') else "none",
```

### 2. PyTorch compute capability warning

**Symptom:**
```
UserWarning: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

**Impact:** Warning only — training proceeds normally. PyTorch 2.10.0 officially supports up to 12.0 but works fine on 12.1.

### 3. Missing libraries (non-blocking)

| Library | Status | Impact |
|---------|--------|--------|
| flash-attn | Not installed (no ARM wheel) | Falls back to xformers / SDPA attention |
| mamba-ssm | Not installed | Not needed for Granite models |
| vLLM | Broken binary extension | Unsloth disables it automatically; not needed for training |

**Unsloth warning on import:**
```
Unsloth: Detected broken vLLM binary extension; disabling vLLM imports and continuing import.
```
This is harmless — vLLM is only needed for inference serving, not fine-tuning.

### 4. Linker warnings during Triton compilation

**Symptom:**
```
/usr/bin/ld: cannot find -laio: No such file or directory
/usr/bin/ld: cannot find -lcufile: No such file or directory
```

**Impact:** These appear during Triton kernel compilation but do not prevent training. Install `libaio-dev` if you want to silence them:
```bash
sudo apt install libaio-dev
```

## What Works

- Unsloth installs and imports successfully (pure-Python + Triton path)
- xformers 0.0.35 works
- BF16 training supported
- QLoRA (4-bit via bitsandbytes) works
- Memory estimator confirms both Granite 2B (~3.3 GB) and 8B (~8.4 GB) fit comfortably in 121.7 GB unified memory

## Tips

- The DGX Spark uses **unified memory** — CPU and GPU share the same 121.7 GB pool. This means you have far more "GPU memory" than typical consumer GPUs, so quantization is less critical for fitting models, though it still speeds up training.
- Set `TORCH_CUDA_ARCH_LIST="12.1"` if you need to compile any CUDA extensions from source.
- Always pass `report_to="none"` (or patch the library) when not using a reporting integration.
