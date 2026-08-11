"""Part 2 — GRPO training for the Wordle agent (TRL GRPOTrainer).

Open-source equivalent of a hosted GRPO job. Run from the repository root:

    python src/train_grpo.py
    python src/train_grpo.py --output-dir /work/<pi>/<user>/wordle-grpo/outputs/wordle-grpo

Defaults come from src/config.py (WORDLE_QUICK=1 for a smoke test).
"""
import argparse
import os

import torch
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

import config
from data import load_grpo_dataset
from reward_adapter import build_reward_funcs


def parse_args():
    p = argparse.ArgumentParser(description="GRPO-train a Wordle agent.")
    p.add_argument("--base-model", default=config.BASE_MODEL)
    p.add_argument("--output-dir", default=config.GRPO_OUTPUT_DIR)
    p.add_argument("--num-generations", type=int, default=config.NUM_GENERATIONS)
    p.add_argument("--max-completion", type=int, default=config.MAX_COMPLETION)
    p.add_argument("--epochs", type=float, default=config.NUM_EPOCHS)
    p.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    return p.parse_args()


def main():
    args = parse_args()
    print("[train_grpo]", config.summary())

    dataset = load_grpo_dataset()
    reward_funcs = build_reward_funcs()

    peft_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.TARGET_MODULES,
    )

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        per_device_train_batch_size=config.PER_DEVICE_BATCH,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=config.LOGGING_STEPS,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        beta=config.BETA,
        epsilon=config.EPSILON,
        use_vllm=config.use_vllm(),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    print(f"[train_grpo] saved LoRA adapter to {final_dir}")


if __name__ == "__main__":
    main()
