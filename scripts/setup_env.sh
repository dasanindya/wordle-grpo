#!/bin/bash
# One-time environment setup. Run from the repository root:  bash scripts/setup_env.sh
set -euo pipefail

# On Unity, load conda first (comment out locally):
# module load conda/latest
# conda create -y -p /work/<pi>/<user>/envs/wordle-grpo python=3.11
# conda activate /work/<pi>/<user>/envs/wordle-grpo

pip install -r requirements.txt

# Reconcile packages that break peft/trl imports in some base images (harmless if absent):
pip uninstall -y torchao || true    # avoids: ImportError: incompatible version of torchao
pip uninstall -y vllm    || true    # avoids: ImportError: libcudart.so.13 (unsupported vLLM)

mkdir -p data outputs logs
echo "Environment ready. Verify GPU inside a job:  python -c 'import torch;print(torch.cuda.is_available())'"
