#!/bin/bash
# Open an interactive GPU shell on Unity for quick checks / debugging.
# Usage:  bash slurm/interactive_gpu.sh
srun --pty \
  -p gpu,gpu-preempt \
  --constraint=a100 \
  -G 1 -c 8 --mem=32G -t 02:00:00 \
  /bin/bash
