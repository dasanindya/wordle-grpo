"""Evaluate a trained adapter: generate a guess for game states and score it.

Loads the base model + a LoRA adapter, decodes greedily (best guess), and reports the
per-function reward breakdown. Run from the repo root:

    python src/evaluate.py --adapter outputs/wordle-grpo/final --n 5
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import config
from data import load_grpo_dataset
from reward_adapter import build_reward_funcs


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Wordle adapter.")
    p.add_argument("--base-model", default=config.BASE_MODEL)
    p.add_argument("--adapter", default=config.GRPO_FINAL_DIR)
    p.add_argument("--n", type=int, default=3, help="number of game states to evaluate")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    dataset = load_grpo_dataset()
    reward_funcs = build_reward_funcs()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(args.base_model)
    mdl = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    mdl = PeftModel.from_pretrained(mdl, args.adapter)
    mdl.eval()

    n = min(args.n, len(dataset))
    for i in range(n):
        sample = dataset[i]
        prompt_text = sample["prompt"]
        inputs = tok(prompt_text, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=args.max_new_tokens,
                               do_sample=False, pad_token_id=tok.eos_token_id)
        completion = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)

        cols = {k: [sample[k]] for k in dataset.column_names if k != "prompt"}
        print(f"\n===== game {i} =====")
        print("COMPLETION:", completion.strip()[:300])
        for rf in reward_funcs:
            score = rf(prompts=[prompt_text], completions=[completion], **cols)[0]
            print(f"  {rf.__name__:24s} {score}")


if __name__ == "__main__":
    main()
