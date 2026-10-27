#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
# CUDA 12.1 wheels for the supplied driver 535.161.07; isolated environment.
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements-gpu.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip freeze > requirements-server.lock.txt
GPU_ID="${GPU_ID:-2}"
CUDA_VISIBLE_DEVICES="$GPU_ID" .venv/bin/python -c 'import torch; print("torch",torch.__version__,"runtime CUDA",torch.version.cuda); print("available",torch.cuda.is_available()); print([(torch.cuda.get_device_name(i),torch.cuda.mem_get_info(i)) for i in range(torch.cuda.device_count())])'
