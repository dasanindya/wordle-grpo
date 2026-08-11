# Wordle-GRPO

Fine-tune a language model to play **Wordle** with **GRPO** (Group Relative Policy
Optimization) — reinforcement learning from programmatic, verifiable rewards. Built on
open-source **HuggingFace TRL / PEFT / Datasets**; no hosted training SDK required. The
base model follows the project report: **`Qwen/Qwen2.5-7B-Instruct`**.

## Quickstart (local, GPU)

```bash
bash scripts/setup_env.sh            # install deps + reconcile torchao/vLLM
WORDLE_QUICK=1 python src/train_grpo.py   # fast smoke test (small model, data subset)
python src/evaluate.py --adapter outputs/wordle-grpo/final --n 3
```

For the faithful 7B run, leave `WORDLE_QUICK` unset (defaults to the report config).

## Quickstart (Unity cluster)

```bash
mkdir -p logs
sbatch slurm/train_grpo.sbatch       # edit the /work paths + partition first
squeue --me                          # PD -> R -> CG
tail -f logs/grpo-<jobid>.out        # follow the reward trend
```

Adapters are written to `outputs/wordle-grpo/final/`; base weights are cached under
`$HF_HOME` (put it on `/work`, not `/home`).

## What's here

| Path | Purpose |
|---|---|
| `notebooks/` | End-to-end notebook (Parts 1–3) |
| `src/config.py` | All settings (env-overridable) |
| `src/data.py` | Load dataset + cache the word list |
| `src/reward_functions.py` | The 3 reward functions |
| `src/reward_adapter.py` | Batched TRL adapter for the rewards |
| `src/train_grpo.py` | GRPO training (Part 2) |
| `src/train_sft.py`, `src/train_sft_then_grpo.py` | SFT warm-start + continuation (Part 3) |
| `src/evaluate.py` | Play a game and score it |
| `src/grpo_loss_demo.py` | GRPO loss from scratch (Part 1) |
| `slurm/` | Unity job scripts (`sbatch`) |
| `configs/grpo_7b.yaml` | Human-readable copy of the 7B settings |
| `docs/` | Project report + full requirements guide |

## Configuration

`src/config.py` is the source of truth; every value has a `WORDLE_*` environment
override (see `configs/grpo_7b.yaml` for the mapping). Common ones:

```bash
export WORDLE_QUICK=0                 # 1 = smoke test
export WORDLE_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
export WORDLE_OUTPUT_ROOT=/work/<pi>/<user>/wordle-grpo/outputs
export WORDLE_USE_VLLM=0              # 1 only with a TRL-supported vLLM installed
```

## Notes

- The GRPO **loss stays near zero by design** — judge progress by the logged **reward**
  trending up (per-function: `output_format_check`, `uses_previous_feedback`,
  `guess_value`), not the loss.
- Using a trained adapter requires the base model **plus** the adapter
  (`PeftModel.from_pretrained`). The adapter alone is not a standalone model.
- The dataset IDs default to the public HuggingFace Wordle datasets and can be swapped
  via `WORDLE_GRPO_DATASET` / `WORDLE_SFT_DATASET`.

See **`docs/REQUIREMENTS.md`** for the full step-by-step setup and Unity submission guide.
