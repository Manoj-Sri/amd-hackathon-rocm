#!/usr/bin/env bash
# scripts/setup_mi300x.sh — bring up Box A (MI300X) for live audits.
#
# Run this once on a fresh AMD MI300X instance, after cloning the repo,
# to install workload + agent deps and sanity-check that torch is actually
# using the GPU before you start uvicorn.
#
# Why a script instead of a README copy-paste:
#   The two failure modes that have repeatedly bitten this project are
#   (a) `pip install datasets peft transformers ...` quietly clobbering
#       the ROCm torch wheel with a CUDA wheel, dropping training to CPU;
#       and (b) the container starting without /dev/kfd passthrough so
#       torch.cuda.is_available() returns False even though `amd-smi list`
#       on the host sees the GPU.
#   Both fail silently — agent runs, workload runs, but step times jump
#   30-100x. This script catches both before they cost you 20 min of
#   "why is the demo so slow" debugging.
#
# Recommended instance images (pick one consistent ROCm version):
#   * rocm/pytorch:rocm6.2.x_ubuntu22.04_py3.10_pytorch_2.4   (lablab default)
#   * rocm/pytorch:rocm7.x_ubuntu22.04_py*_pytorch_*           (newer)
# Whichever you pick, do NOT mix-and-match — keep system ROCm and torch's
# `torch.version.hip` on the same major version or rocprofv3 will crash.
#
# Usage:
#   bash scripts/setup_mi300x.sh
#   # OR with a specific python:
#   PYTHON=/opt/conda/bin/python bash scripts/setup_mi300x.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"

cd "$(dirname "$0")/.."   # repo root

echo "==> Pre-flight: torch + ROCm runtime"
$PYTHON - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit(
        "ERROR: torch is not installed. The MI300X container should ship "
        "torch preinstalled — if it doesn't, your image is wrong. Pick a "
        "rocm/pytorch:* image and rerun this script."
    )

hip = torch.version.hip
cuda = torch.version.cuda

if hip is None:
    sys.exit(
        f"ERROR: torch=={torch.__version__} is NOT a ROCm build "
        f"(torch.version.hip is None, torch.version.cuda={cuda!r}).\n"
        "Most likely cause: a previous `pip install` resolved torch from "
        "the default PyPI index and overwrote the ROCm wheel with a CUDA "
        "wheel. Reinstall the matching ROCm wheel before continuing:\n"
        "    pip uninstall -y torch\n"
        "    pip install --pre torch --index-url "
        "https://download.pytorch.org/whl/nightly/rocm6.2\n"
        "(Match the ROCm major version of your container; 6.2 shown for "
        "rocm/pytorch:rocm6.2.x_*)."
    )

if not torch.cuda.is_available():
    sys.exit(
        "ERROR: torch.cuda.is_available() is False even though torch is a "
        "ROCm build. The container is probably missing /dev/kfd or "
        "/dev/dri passthrough. When starting the container, pass:\n"
        "    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host"
    )

print(
    f"OK: torch={torch.__version__} hip={hip} "
    f"device_count={torch.cuda.device_count()} "
    f"device_0={torch.cuda.get_device_name(0)}"
)
PY

echo
echo "==> Installing workload deps from requirements-mi300x.txt"
echo "    (datasets, peft, transformers, accelerate — NOT torch)"
$PYTHON -m pip install --no-cache-dir -r requirements-mi300x.txt

echo
echo "==> Installing agent + runner deps (pip install -e .)"
$PYTHON -m pip install --no-cache-dir -e .

echo
echo "==> Post-flight: GPU compute sanity"
$PYTHON - <<'PY'
import sys
import torch

hip = torch.version.hip
if hip is None or not torch.cuda.is_available():
    sys.exit(
        "ERROR: post-install verification failed. The pip step above "
        "almost certainly clobbered the ROCm torch wheel. Reinstall the "
        "matching ROCm wheel and re-run this script."
    )

# Tiny matmul on-device to prove HIP kernels actually launch.
x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
y = (x @ x).sum()
torch.cuda.synchronize()
print(f"OK: matmul on {x.device} -> {y.item():.2f} (hip={hip})")
PY

echo
echo "==> Done. Start the agent server with:"
echo "      export GOBLIN_AGENT_BACKEND=qwen-vllm"
echo "      export GOBLIN_QWEN_VLLM_URL=http://<box-b>:8000/v1"
echo "      export GOBLIN_QWEN_VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct"
echo "      uvicorn agent.server:app --host 0.0.0.0 --port 8000"
echo
echo "    Smoke-test from anywhere:"
echo "      curl http://localhost:8000/healthz | jq ."
