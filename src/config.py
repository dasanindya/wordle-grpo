"""Central configuration for the Wordle-GRPO project.

All knobs live here and can be overridden with environment variables, so the same
code runs locally, in the notebook, and on the Unity cluster without edits.

Set WORDLE_QUICK=1 for a fast smoke test (small model + data subset). Leave it unset
(default) for the faithful run from the project report: Qwen/Qwen2.5-7B-Instruct.
"""
import os


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------- data
GRPO_DATASET_ID = os.environ.get("WORDLE_GRPO_DATASET", "predibase/wordle-grpo")
SFT_DATASET_ID = os.environ.get("WORDLE_SFT_DATASET", "predibase/wordle-sft")
# Local cache for the valid-word dictionary (relative to the repo root / CWD).
WORD_LIST_PATH = os.environ.get("WORDLE_WORD_LIST", "data/wordle_word_list.csv")

# ------------------------------------------------------------------ quick / full
QUICK = _flag("WORDLE_QUICK", "0")

if QUICK:
    BASE_MODEL = os.environ.get("WORDLE_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    NUM_GENERATIONS = int(os.environ.get("WORDLE_NUM_GENERATIONS", "8"))
    MAX_COMPLETION = int(os.environ.get("WORDLE_MAX_COMPLETION", "1024"))
    N_TRAIN = int(os.environ.get("WORDLE_N_TRAIN", "64"))
    MAX_STEPS = int(os.environ.get("WORDLE_MAX_STEPS", "20"))
else:
    # Faithful configuration from the project report.
    BASE_MODEL = os.environ.get("WORDLE_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    NUM_GENERATIONS = int(os.environ.get("WORDLE_NUM_GENERATIONS", "16"))
    MAX_COMPLETION = int(os.environ.get("WORDLE_MAX_COMPLETION", "4096"))
    _n = os.environ.get("WORDLE_N_TRAIN", "")
    N_TRAIN = int(_n) if _n.strip() else None          # None => full dataset
    MAX_STEPS = int(os.environ.get("WORDLE_MAX_STEPS", "-1"))  # -1 => use epochs

# --------------------------------------------------------------- training hypers
LORA_RANK = int(os.environ.get("WORDLE_LORA_RANK", "64"))
LORA_ALPHA = int(os.environ.get("WORDLE_LORA_ALPHA", "128"))
LORA_DROPOUT = float(os.environ.get("WORDLE_LORA_DROPOUT", "0.05"))
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "down_proj", "up_proj"]

LEARNING_RATE = float(os.environ.get("WORDLE_LR", "1e-5"))
NUM_EPOCHS = float(os.environ.get("WORDLE_EPOCHS", "1"))
PER_DEVICE_BATCH = int(os.environ.get("WORDLE_BATCH", str(NUM_GENERATIONS)))
GRAD_ACCUM = int(os.environ.get("WORDLE_GRAD_ACCUM", "1"))
BETA = float(os.environ.get("WORDLE_BETA", "0.04"))      # KL coefficient
EPSILON = float(os.environ.get("WORDLE_EPSILON", "0.2"))  # PPO clip range
LOGGING_STEPS = int(os.environ.get("WORDLE_LOGGING_STEPS", "1"))

# SFT-specific (Part 3)
SFT_EPOCHS = float(os.environ.get("WORDLE_SFT_EPOCHS", "1" if QUICK else "10"))
SFT_LR = float(os.environ.get("WORDLE_SFT_LR", "2e-4"))

# ------------------------------------------------------------------- output dirs
OUTPUT_ROOT = os.environ.get("WORDLE_OUTPUT_ROOT", "outputs")
GRPO_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "wordle-grpo")
SFT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "wordle-sft")
GRPO_FINAL_DIR = os.path.join(GRPO_OUTPUT_DIR, "final")
SFT_FINAL_DIR = os.path.join(SFT_OUTPUT_DIR, "final")


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def use_vllm() -> bool:
    """Only use vLLM if explicitly requested AND a GPU is present."""
    return _flag("WORDLE_USE_VLLM", "0") and cuda_available()


def summary() -> str:
    return (
        f"QUICK={QUICK} BASE_MODEL={BASE_MODEL} num_generations={NUM_GENERATIONS} "
        f"max_completion={MAX_COMPLETION} n_train={N_TRAIN} max_steps={MAX_STEPS} "
        f"use_vllm={use_vllm()} output_root={OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    print(summary())
