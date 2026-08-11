#!/bin/bash
#SBATCH -J wordle-sft2grpo
#SBATCH -p gpu-long
#SBATCH --constraint=a100
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/sft2grpo-%j.out
#SBATCH -e logs/sft2grpo-%j.err

set -euo pipefail
module purge
module load conda/latest
conda activate /work/<pi>/<user>/envs/wordle-grpo
export HF_HOME=/work/<pi>/<user>/hf_cache
export TOKENIZERS_PARALLELISM=false
export WORDLE_QUICK=0

cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi -L
python src/train_sft_then_grpo.py --sft-checkpoint outputs/wordle-sft/final
