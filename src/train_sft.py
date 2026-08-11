"""Part 3 (optional) — supervised warm-start on the Wordle SFT dataset.

Establishes the <think>/<guess> output format before GRPO. Run from the repo root:

    python src/train_sft.py

The resulting checkpoint (outputs/wordle-sft/final) can seed a GRPO run via
train_sft_then_grpo.py.
"""
import argparse
import os

import torch
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import config
from data import load_sft_dataset


def parse_args():
    p = argparse.ArgumentParser(description="SFT warm-start for the Wordle agent.")
    p.add_argument("--base-model", default=config.BASE_MODEL)
    p.add_argument("--output-dir", default=config.SFT_OUTPUT_DIR)
    p.add_argument("--epochs", type=float, default=config.SFT_EPOCHS)
    p.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    p.add_argument("--lr", type=float, default=config.SFT_LR)
    return p.parse_args()


def main():
    args = parse_args()
    print("[train_sft]", config.summary())

    dataset = load_sft_dataset()

    peft_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.TARGET_MODULES,
    )

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        logging_steps=config.LOGGING_STEPS,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.base_model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    print(f"[train_sft] saved SFT adapter to {final_dir}")


if __name__ == "__main__":
    main()
