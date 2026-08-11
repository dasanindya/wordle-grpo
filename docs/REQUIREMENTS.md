# Wordle-GRPO — Requirements & Step-by-Step Setup Guide

This guide lists everything needed to run the project end to end: training a
Wordle-playing agent with **Group Relative Policy Optimization (GRPO)**, using the
notebook `Wordle_GRPO_without_Predibase.ipynb`. The base model follows the project
report: **`Qwen/Qwen2.5-7B-Instruct`**. It includes instructions to submit the job to
the **Unity cluster** (Slurm) and capture logs.

---

## 1. Overview

| Item | Value |
|---|---|
| Task | Fine-tune an LLM to play Wordle with GRPO (RL from programmatic rewards) |
| Base model (training) | `Qwen/Qwen2.5-7B-Instruct` |
| Training engine | HuggingFace **TRL** `GRPOTrainer` (open-source; no hosted SDK) |
| Adapter | LoRA (rank 64) via **PEFT** — only the adapter is trained |
| Rewards | 3 functions summed: `output_format_check`, `uses_previous_feedback`, `guess_value` |
| Sampling | 16 candidate guesses per game state, 4096-token budget |
| Cluster | Unity (Slurm scheduler; `sbatch` submission; GPU partitions) |
| Optional | SFT warm-start, then SFT→GRPO continuation |

---

## 2. Project Structure

```text
wordle-grpo/
├── README.md                         # project intro + quickstart
├── requirements.txt                  # pinned dependencies
├── .env.example                      # HF_TOKEN and other env vars (copy to .env)
├── .gitignore                        # ignores data/, outputs/, logs/, .env, caches
│
├── notebooks/
│   └── Wordle_GRPO_without_Predibase.ipynb   # full pipeline (Parts 1–3)
│
├── src/                              # notebook logic factored into modules
│   ├── config.py                    # QUICK, BASE_MODEL, hyperparameters (Step 5)
│   ├── data.py                      # load dataset + cache word list (Step 6)
│   ├── reward_functions.py          # the 3 reward functions (Step 7)
│   ├── reward_adapter.py            # make_trl_reward: single→batched TRL adapter
│   ├── grpo_loss_demo.py            # Part 1: from-scratch GRPO loss (optional)
│   ├── train_grpo.py                # Part 2: GRPO training (Steps 8–10)
│   ├── train_sft.py                 # Part 3: SFT warm-start (optional)
│   ├── train_sft_then_grpo.py       # Part 3: continue GRPO from SFT (optional)
│   └── evaluate.py                  # play a game + score with rewards (Step 11)
│
├── slurm/                           # Unity job scripts (Slurm)
│   ├── train_grpo.sbatch            # submit Part 2 GRPO (7B) to Unity
│   ├── train_sft.sbatch             # submit Part 3 SFT
│   ├── train_sft_then_grpo.sbatch   # submit Part 3 continuation
│   └── interactive_gpu.sh           # salloc/srun for a quick interactive GPU shell
│
├── configs/
│   └── grpo_7b.yaml                 # run settings for the 7B configuration
│
├── data/                            # (gitignored) local, downloaded, or cached data
│   └── wordle_word_list.csv         # cached valid-word dictionary (written by data.py)
│
├── outputs/                         # (gitignored) all training artifacts
│   ├── wordle-grpo/final/           # trained GRPO LoRA adapter
│   └── wordle-sft/final/            # SFT checkpoint (if Part 3 is run)
│
├── logs/                            # (gitignored) Slurm logs: grpo-<jobid>.out/.err
│
├── docs/
│   ├── Wordle_GRPO_Project_Report.docx
│   └── REQUIREMENTS.md              # this file
│
└── scripts/
    └── setup_env.sh                 # install deps + reconcile torchao/vLLM (Steps 1–2)
```

> **Three ways to run.** (a) The **notebook** in `notebooks/` runs everything itself
> (good for a Unity OnDemand JupyterLab session). (b) The **`src/` modules** run headless
> (`python src/train_grpo.py`). (c) On Unity, submit the **`slurm/*.sbatch`** scripts with
> `sbatch`; stdout/stderr land in **`logs/`**. Create `logs/` before submitting.

---

## 3. Hardware Requirements

| Configuration | Requirement |
|---|---|
| **Full GRPO run** (`Qwen2.5-7B-Instruct`, 16 generations) | 1 × GPU, **A100-class (≥ 40 GB VRAM; 80 GB preferred)** |
| **Reduced run** (smaller model / fewer generations) | 1 × GPU (≥ 24 GB helps) |
| **Part 1 only** (`src/grpo_loss_demo.py`, loss mechanics) | Runs on CPU with a small model |

> On Unity, pin the GPU with a Slurm **constraint** (e.g. `--constraint=a100`, or a VRAM
> constraint such as `--constraint=vram80`). If you hit out-of-memory, lower
> `num_generations`, `max_completion_length`, and `per_device_train_batch_size` in
> `src/config.py`.

---

## 4. Software Requirements

- **Python** 3.10 – 3.12
- **PyTorch** with CUDA matching the node's driver (on Unity, via `module load` or conda)
- Core libraries (pinned to a GRPO-compatible combo) in `requirements.txt`:

```text
# requirements.txt
trl>=0.15.0
transformers>=4.48.0
peft>=0.14.0
datasets>=3.0.0
accelerate>=1.2.0
pandas
matplotlib
# optional, GPU-only, MUST be a TRL-supported version:
# vllm>=0.17,<=0.25.1
```

- A **Hugging Face account/token** to download `Qwen/Qwen2.5-7B-Instruct` and the Wordle
  datasets. Put the token in `.env` (`HF_TOKEN=...`) or run `huggingface-cli login` once.
- On Unity, cache models/datasets on `/work` (large quota), not `/home`:
  `export HF_HOME=/work/<pi>/<user>/hf_cache`.

---

## 5. Data & Model Assets

| Asset | Purpose | Location / Notes |
|---|---|---|
| GRPO dataset (Wordle game states) | GRPO training | Loaded by `src/data.py`. Columns: `prompt`, `word_list`, `past_guess_history` |
| SFT dataset (Wordle) | Optional warm-start | Used by `src/train_sft.py` (Part 3) |
| Valid-word dictionary | Reward scoring | Cached to `data/wordle_word_list.csv` by `src/data.py` |
| `Qwen/Qwen2.5-7B-Instruct` | Base policy + frozen reference | Loaded in bfloat16; cache under `HF_HOME` on `/work` |

Each prompt is prefilled with an opening `<think>` tag, so the model continues a
short chain of thought before emitting `<guess>WORD</guess>`.

---

## 6. Step-by-Step Setup (run once, on a login node or interactive job)

All commands are run from the **repository root** (`wordle-grpo/`).

### Step 1 — Create the environment
On Unity, build the env in your `/work` space so it persists and doesn't fill `/home`:

```bash
module load conda/latest
conda create -y -p /work/<pi>/<user>/envs/wordle-grpo python=3.11
conda activate /work/<pi>/<user>/envs/wordle-grpo
pip install -r requirements.txt
```

### Step 2 — Reconcile conflicting preinstalled packages
Only needed if a base image ships incompatible versions (more common on Colab than a
clean Unity env), but harmless to run:

```bash
pip uninstall -y torchao   # avoids: ImportError: incompatible version of torchao
pip uninstall -y vllm      # avoids: ImportError: libcudart.so.13 (unsupported vLLM)
```
> To *use* vLLM for speed, install a supported build (`vllm>=0.17,<=0.25.1`) matching torch/CUDA.

### Step 3 — Verify GPU access (inside a GPU job, not the login node)
```bash
bash slurm/interactive_gpu.sh          # opens an interactive GPU shell (srun/salloc)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
CUDA must be `True`. (Login nodes have no GPU — always verify inside a job.)

### Step 4 — (Optional) Part 1: GRPO loss mechanics — `src/grpo_loss_demo.py`
Illustrates the ratio, PPO clipping, and KL penalty from scratch. Runs on CPU with a
small model. bf16 models are handled (plotting helper casts to float32 before NumPy).

### Step 5 — Configure the 7B run — `src/config.py`
```python
QUICK           = False
BASE_MODEL      = "Qwen/Qwen2.5-7B-Instruct"   # from the project report
NUM_GENERATIONS = 16
MAX_COMPLETION  = 4096
N_TRAIN         = None      # full dataset
MAX_STEPS       = -1        # let epochs decide
USE_VLLM        = False     # set True only if you installed a supported vLLM
```

### Step 6 — Load data & cache the word list — `src/data.py`
Loads the GRPO dataset, caches the valid-word dictionary to `data/wordle_word_list.csv`,
and points every row's `word_list` at that local file (fast, offline reward scoring).

### Step 7 — Rewards + TRL adapter — `src/reward_functions.py`, `src/reward_adapter.py`
`make_trl_reward` maps TRL's batched call onto the per-completion reward logic, returns a
list of floats, and **ignores non-column keyword arguments** the trainer injects (e.g. `trainer_state`).

### Step 8 — Build the trainer — `src/train_grpo.py`
`LoraConfig` (rank 64; attention + MLP projections) + `GRPOConfig` (Step 5 values, `beta`
for KL, `epsilon` for clipping) + `GRPOTrainer(...)`.

---

## 7. Running on the Unity Cluster (Slurm + Logs)

### Step 9 — Write the batch script — `slurm/train_grpo.sbatch`
```bash
#!/bin/bash
#SBATCH -J wordle-grpo               # job name
#SBATCH -p gpu-long                  # GPU partition: gpu | gpu-long | gpu-preempt
#SBATCH --constraint=a100            # GPU type (or e.g. vram80)
#SBATCH -G 1                         # request 1 GPU (required on GPU partitions)
#SBATCH -c 8                         # CPU cores
#SBATCH --mem=64G                    # host RAM
#SBATCH -t 2-00:00:00                # walltime D-HH:MM:SS (adjust to partition limit)
#SBATCH -o logs/grpo-%j.out          # stdout  (%j = job ID)
#SBATCH -e logs/grpo-%j.err          # stderr

module purge
module load conda/latest
conda activate /work/<pi>/<user>/envs/wordle-grpo

export HF_HOME=/work/<pi>/<user>/hf_cache      # cache weights on /work, not /home
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname); GPU:"; nvidia-smi -L
python src/train_grpo.py
```

### Step 10 — Submit and collect logs
```bash
mkdir -p logs                        # Slurm won't create the log dir for you
sbatch slurm/train_grpo.sbatch       # prints: Submitted batch job <jobid>
```
The GRPO adapter is written to `outputs/wordle-grpo/final/`; stdout/stderr stream live to
`logs/grpo-<jobid>.out` and `logs/grpo-<jobid>.err`.

### Step 11 — Monitor the job and its logs
```bash
squeue --me                          # job state: PD (pending) / R (running) / CG (completing)
tail -f logs/grpo-<jobid>.out        # follow training output in real time
scancel <jobid>                      # cancel if needed
```
Watch the **reward** trend in the log, not the loss — GRPO's loss sits near zero by
construction; rising `reward` with modest `kl` means it's learning.

### Step 12 — Evaluate & (optional) SFT→GRPO
- Evaluate with `src/evaluate.py` (greedy decoding, `do_sample=False`), loading the base
  model + `outputs/wordle-grpo/final/`, then scoring with the reward functions.
- For stronger format adherence, submit `slurm/train_sft.sbatch` first (checkpoint →
  `outputs/wordle-sft/final/`), then `slurm/train_sft_then_grpo.sbatch` to continue GRPO
  from it. This is the biggest lever when the base model struggles to emit a valid guess.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Job stuck in `PD` (pending) | Requested GPU/constraint is busy or over-specified | Broaden partition (`-p gpu,gpu-preempt`), relax `--constraint`, or lower `--mem`/`-t` |
| Job killed/requeued after ~2 h | Ran on a preempt partition | Use `-p gpu` or `-p gpu-long` for long GRPO runs |
| `logs/…: No such file or directory` at submit | Log dir missing | `mkdir -p logs` before `sbatch` |
| `torch.cuda.is_available()` is `False` | Running on a login node / no GPU requested | Run inside a GPU job with `-G 1` |
| `ImportError: incompatible version of torchao` | `peft` rejects old `torchao` | `pip uninstall -y torchao`, restart |
| `ImportError: libcudart.so.13 …` | Unsupported `vLLM` for another CUDA | `pip uninstall -y vllm` (or install `vllm>=0.17,<=0.25.1`) |
| `TypeError: 'TrainerState' object is not subscriptable` | Reward adapter indexed a non-column kwarg | Keep only per-completion list columns in `reward_adapter.py` (already handled) |
| `TypeError: Got unsupported ScalarType BFloat16` | NumPy can't convert bf16 | `.float()` before `.numpy()` (already handled) |
| CUDA out-of-memory | 16×4096-token rollouts on 7B | Lower `NUM_GENERATIONS`/`MAX_COMPLETION`/batch size; request `--constraint=vram80` |

---

## 9. Definition of Done

- Env installs on `/work` and `torch.cuda.is_available()` is `True` inside a GPU job.
- `sbatch slurm/train_grpo.sbatch` returns a job ID and the job reaches state `R`.
- `logs/grpo-<jobid>.out` shows training progress with the **reward** trending upward.
- A saved LoRA adapter in `outputs/wordle-grpo/final/` produces well-formed
  `<guess>WORD</guess>` outputs that the reward functions score above zero.
