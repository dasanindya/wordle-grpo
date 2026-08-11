#!/bin/bash
#SBATCH -J wordle-grpo               # job name
#SBATCH -p gpu-long                  # GPU partition: gpu | gpu-long | gpu-preempt
#SBATCH --constraint=a100            # GPU type (or a VRAM constraint, e.g. vram80)
#SBATCH -G 1                         # request 1 GPU (required on GPU partitions)
#SBATCH -c 8                         # CPU cores
#SBATCH --mem=64G                    # host RAM
#SBATCH -t 2-00:00:00                # walltime D-HH:MM:SS (adjust to partition limit)
#SBATCH -o logs/grpo-%j.out          # stdout (%j = job ID)
#SBATCH -e logs/grpo-%j.err          # stderr

set -euo pipefail

module purge
module load conda/latest
# EDIT: point at your env on /work
conda activate /work/<pi>/<user>/envs/wordle-grpo

# EDIT: cache weights/datasets on /work (large quota), not /home
export HF_HOME=/work/<pi>/<user>/hf_cache
export TOKENIZERS_PARALLELISM=false
export WORDLE_QUICK=0                 # 0 = faithful 7B run from the report

cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi -L
python src/train_grpo.py
