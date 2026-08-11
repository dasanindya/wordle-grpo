#!/bin/bash
#SBATCH -J wordle-sft
#SBATCH -p gpu-long
#SBATCH --constraint=a100
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/sft-%j.out
#SBATCH -e logs/sft-%j.err

set -euo pipefail
module purge
module load conda/latest
conda activate /work/<pi>/<user>/envs/wordle-grpo
export HF_HOME=/work/<pi>/<user>/hf_cache
export TOKENIZERS_PARALLELISM=false
export WORDLE_QUICK=0

cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi -L
python src/train_sft.py
