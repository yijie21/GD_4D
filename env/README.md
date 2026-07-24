# Environment — `gd4d5090` (standalone conda env for this repo)

This is the **one conda environment** for GD-4D: it runs the vendored OpenD4RT backbone
(`third_party/Open-d4rt`) and our own code (`src/`). It is named `gd4d5090` because it is
built for this box's **RTX 5090 (Blackwell, compute capability 12.0)**.

> **Why not just use OpenD4RT's `environment.yml`?**
> The vendored repo pins `torch==2.6.0+cu124`. **That build does not run on Blackwell.**
> An sm_120 GPU needs **CUDA ≥ 12.8** wheels; a cu124 wheel installs cleanly but dies at the
> first GPU op with `no kernel image is available for execution on the device`. So this env
> deliberately deviates to a **cu128** PyTorch build. Everything else matches the repo pins.

## Verified working versions (2026-07-24)

| | version |
|---|---|
| Python | 3.10 |
| torch | **2.11.0+cu128** |
| torchvision | 0.26.0+cu128 |
| numpy | 2.2.6 |
| opencv-python-headless | 4.13.0.92 |
| h5py | 3.16.0 (for LIBERO `.hdf5` loading) |

Exact frozen pins: [`requirements-gd4d5090.lock.txt`](requirements-gd4d5090.lock.txt).
Intent-level list: [`requirements-gd4d5090.txt`](requirements-gd4d5090.txt).

## Recreate the environment from scratch

```bash
# 1. create the env
source /workspace/miniconda3/etc/profile.d/conda.sh
conda create -y -n gd4d5090 python=3.10 pip
PY=/workspace/miniconda3/envs/gd4d5090/bin/python
PIP=/workspace/miniconda3/envs/gd4d5090/bin/pip

# 2. PyTorch — cu128 build (REQUIRED for RTX 5090 / Blackwell sm_120; NOT the repo's cu124 pin)
$PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. the rest of OpenD4RT's deps + tooling we add
$PIP install "PyYAML==6.0.3" "opencv-python-headless==4.13.0.92" "imageio==2.37.3" \
             "matplotlib==3.10.8" "tqdm==4.67.3" "tensorboard==2.20.0" "lz4==4.4.5" \
             "huggingface_hub[cli]" "h5py"

# 4. sanity: a real CUDA kernel must run on sm_120 (not just is_available())
$PY - <<'PYEOF'
import torch
assert torch.cuda.is_available()
print("device:", torch.cuda.get_device_name(0), "| cc:", torch.cuda.get_device_capability(0))
print("arch_list:", torch.cuda.get_arch_list())          # must contain 'sm_120'
(torch.randn(2048,2048,device="cuda") @ torch.randn(2048,2048,device="cuda")).sum().item()
print("CUDA matmul OK")
PYEOF
```

Or reproduce the exact pins: `$PIP install -r env/requirements-gd4d5090.lock.txt`
(note the lock file's torch/torchvision lines still require the cu128 index — pass
`--index-url https://download.pytorch.org/whl/cu128` if pip can't find them on PyPI).

## Use it

```bash
/workspace/miniconda3/envs/gd4d5090/bin/python <script>        # explicit interpreter (robust in non-interactive shells)
# or:  conda activate gd4d5090
```
