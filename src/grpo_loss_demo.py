"""Part 1 (optional) — the GRPO loss from scratch, for understanding.

Builds a frozen reference model and a LoRA policy, then computes the probability
ratio, the PPO-clipped objective, and the KL penalty on a toy prompt. Saves two
figures to the logs/ directory. Runs on CPU with a small model.

    python src/grpo_loss_demo.py
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser(description="GRPO loss mechanics demo.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="small causal LM; e.g. babylm/babyllama-100m-2024 for CPU")
    p.add_argument("--prompt", default="The quick brown fox jumped over the ")
    p.add_argument("--completion", default="fence and")
    p.add_argument("--logdir", default="logs")
    return p.parse_args()


def prepare_inputs(tokenizer, prompt, completion):
    pt = tokenizer(prompt, return_tensors="pt")
    ct = tokenizer(completion, return_tensors="pt")
    input_ids = torch.cat([pt["input_ids"], ct["input_ids"]], dim=1)
    attn = torch.cat([pt["attention_mask"], ct["attention_mask"]], dim=1)
    plen = pt["input_ids"].shape[1]
    total = input_ids.shape[1]
    mask = torch.zeros(total, dtype=torch.float32)
    mask[plen:] = 1.0
    return input_ids, attn, mask, plen


def compute_log_probs(model, input_ids, attention_mask):
    out = model(input_ids, attention_mask=attention_mask)
    lp = F.log_softmax(out.logits, dim=-1)
    return lp.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def grpo_loss_with_kl(model, ref_model, tokenizer, prompt, completion,
                      advantage, epsilon=0.2, beta=0.1):
    input_ids, attn, mask, _ = prepare_inputs(tokenizer, prompt, completion)
    lp = compute_log_probs(model, input_ids, attn)
    with torch.no_grad():
        ref_lp = compute_log_probs(ref_model, input_ids, attn)
    ratio = torch.exp(lp - ref_lp)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    policy = torch.min(unclipped, clipped)
    delta = lp - ref_lp
    per_token_kl = torch.exp(-delta) + delta - 1
    per_token_loss = -(policy - beta * per_token_kl)
    return (per_token_loss * mask).sum() / mask.sum()


def visualize_clipped_ratios(ru, rc, epsilon, path):
    ru = ru.detach().cpu().float().numpy().ravel()
    rc = rc.detach().cpu().float().numpy().ravel()
    x = np.arange(len(ru))
    clipped = ~np.isclose(ru, rc)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhspan(1 - epsilon, 1 + epsilon, color="green", alpha=0.10,
               label=f"trust region [1±{epsilon}]")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.bar(x - 0.2, ru, width=0.4, label="unclipped", color="#4C72B0")
    ax.bar(x + 0.2, rc, width=0.4, label="clipped", color="#DD8452")
    ax.set_title(f"{clipped.sum()}/{len(ru)} completion tokens clipped")
    ax.set_xlabel("completion token index"); ax.set_ylabel("ratio")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path); plt.close()
    print("[demo] wrote", path)


def kl_curve(path):
    delta = np.linspace(-6, 6, 500)
    kl = np.exp(-delta) + delta - 1
    plt.figure(figsize=(8, 5))
    plt.plot(delta, kl, label=r"$e^{-\Delta}+\Delta-1$")
    plt.axhline(0, color="gray", ls="--", lw=0.5)
    plt.axvline(0, color="gray", ls="--", lw=0.5)
    plt.title("KL penalty vs. log-prob gap")
    plt.xlabel(r"$\Delta=\log\pi-\log\pi_{ref}$"); plt.ylabel("KL")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(path); plt.close()
    print("[demo] wrote", path)


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ref_model = copy.deepcopy(base)
    lora = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.1, init_lora_weights=False, bias="none",
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)

    for beta in (0.0, 0.1, 0.5):
        loss = grpo_loss_with_kl(model, ref_model, tok, args.prompt,
                                 args.completion, advantage=2.0, beta=beta)
        print(f"[demo] beta={beta}  loss={loss.item():.4f}")

    input_ids, attn, _, plen = prepare_inputs(tok, args.prompt, args.completion)
    with torch.no_grad():
        lp = compute_log_probs(model, input_ids, attn)
        ref_lp = compute_log_probs(ref_model, input_ids, attn)
    eps = 0.2
    ratio = torch.exp(lp - ref_lp)
    ratio_clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    visualize_clipped_ratios(ratio[0][plen:], ratio_clipped[0][plen:], eps,
                             os.path.join(args.logdir, "clipped_ratios.png"))
    kl_curve(os.path.join(args.logdir, "kl_curve.png"))


if __name__ == "__main__":
    main()
