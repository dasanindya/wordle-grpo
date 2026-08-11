"""Part 3 (optional) — continue GRPO from the SFT checkpoint.

Loads the base model + the SFT LoRA adapter (trainable) and runs GRPO on top, the
open-source analogue of "continue from version". Run from the repo root:

    python src/train_sft_then_grpo.py --sft-checkpoint outputs/wordle-sft/final
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

import config
from data import load_grpo_dataset
from reward_adapter import build_reward_funcs


def parse_args():
    p = argparse.ArgumentParser(description="Continue GRPO from an SFT checkpoint.")
    p.add_argument("--base-model", default=config.BASE_MODEL)
    p.add_argument("--sft-checkpoint", default=config.SFT_FINAL_DIR)
    p.add_argument("--output-dir", default=os.path.join(config.OUTPUT_ROOT, "wordle-sft-then-grpo"))
    p.add_argument("--num-generations", type=int, default=8)   # report's stage-2 value
    p.add_argument("--epochs", type=float, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    print("[sft->grpo]", config.summary(), "| sft_ckpt:", args.sft_checkpoint)

    dataset = load_grpo_dataset()
    reward_funcs = build_reward_funcs()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    model = PeftModel.from_pretrained(base, args.sft_checkpoint, is_trainable=True)

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=config.MAX_COMPLETION,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=args.epochs,
        logging_steps=config.LOGGING_STEPS,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        beta=config.BETA,
        epsilon=config.EPSILON,
        use_vllm=config.use_vllm(),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=dataset,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    print(f"[sft->grpo] saved adapter to {final_dir}")


if __name__ == "__main__":
    main()
