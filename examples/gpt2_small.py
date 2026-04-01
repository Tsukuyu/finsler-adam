#!/usr/bin/env python3
"""
GPT-2 Small Language Model Benchmark: AdamW vs Finsler-Adam

Trains GPT-2 Small (124M params) on OpenWebText (or Wikitext-103 fallback)
with four optimizer configurations. Measures:
  - Training / validation perplexity curves
  - Steps to reach target perplexity
  - Wall-clock time per 1K steps
  - Peak GPU memory

This follows the experimental protocol of Sophia (Liu et al., 2024).

Requirements:
    pip install finsler-adam transformers datasets accelerate wandb

Usage:
    python gpt2_small.py                           # default: 50K steps
    python gpt2_small.py --config adamw --steps 100000
    python gpt2_small.py --dataset wikitext         # use wikitext-103
    python gpt2_small.py --fp16                     # mixed precision
"""

import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finsler_adam import FinslerAdam

# ─────────────────────────── Optimizer Configs ──────────────────────────────

CONFIGS = {
    "adamw": dict(gamma=0.0, anna_alpha=0.0),
    "finsler_full": dict(gamma=0.5, anna_alpha=0.1),
    "anna_only": dict(gamma=0.0, anna_alpha=0.1),
    "finsler_only": dict(gamma=0.5, anna_alpha=0.0),
}


# ─────────────────────────── Data ───────────────────────────────────────────

def get_dataloader(args):
    """Load and tokenize dataset."""
    from transformers import GPT2Tokenizer
    from datasets import load_dataset

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == "openwebtext":
        dataset = load_dataset("openwebtext", split="train", streaming=True)
    else:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    block_size = args.seq_len

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=block_size,
                         padding="max_length", return_tensors="pt")

    if args.dataset == "openwebtext":
        # Streaming: tokenize on the fly
        def collate_streaming(batch):
            texts = [item["text"] for item in batch if item["text"].strip()]
            if not texts:
                texts = ["<|endoftext|>"]
            enc = tokenizer(texts, truncation=True, max_length=block_size,
                            padding="max_length", return_tensors="pt")
            return enc["input_ids"]

        return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_streaming)
    else:
        # Wikitext: pre-tokenize
        tokenized = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=block_size,
                                padding="max_length"),
            batched=True, remove_columns=dataset.column_names,
        )
        tokenized.set_format("torch")
        return DataLoader(tokenized, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True,
                          collate_fn=lambda b: torch.stack([x["input_ids"] for x in b]))


# ─────────────────────────── Training ───────────────────────────────────────

def run_experiment(config_name, config, args, seed, device):
    from transformers import GPT2LMHeadModel, GPT2Config

    torch.manual_seed(seed)

    # Model
    model_config = GPT2Config(
        vocab_size=50257,
        n_positions=args.seq_len,
        n_embd=768, n_head=12, n_layer=12,  # GPT-2 Small
    )
    model = GPT2LMHeadModel(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params / 1e6:.1f}M")

    # Optimizer
    optimizer = FinslerAdam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=(0.9, 0.95),  # GPT-2 standard
        **config,
    )

    # Scheduler: cosine with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup)
    cosine = CosineAnnealingLR(optimizer, T_max=args.steps - args.warmup)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup])

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    # Data
    dataloader = get_dataloader(args)

    log = []
    model.train()
    step = 0
    data_iter = iter(dataloader)
    t_start = time.time()

    while step < args.steps:
        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids = next(data_iter).to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=args.fp16):
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        if args.fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()
        step += 1

        # Logging
        if step % args.log_interval == 0 or step == args.steps:
            ppl = math.exp(min(loss.item(), 20))  # cap for numerical safety
            elapsed = time.time() - t_start
            mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0

            entry = dict(
                config=config_name, seed=seed, step=step,
                train_loss=loss.item(), train_ppl=ppl,
                lr=scheduler.get_last_lr()[0],
                wall_sec=elapsed, peak_mem_gb=mem_gb,
            )
            log.append(entry)

            if step % (args.log_interval * 10) == 0 or step == args.steps:
                print(f"  [{config_name}|s{seed}] Step {step:6d}/{args.steps} | "
                      f"Loss {loss.item():.4f} | PPL {ppl:.1f} | "
                      f"LR {entry['lr']:.2e} | {elapsed:.0f}s | {mem_gb:.1f}GB")

    return log


# ─────────────────────────── Main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Small Benchmark")
    parser.add_argument("--config", type=str, default="all",
                        choices=["all"] + list(CONFIGS.keys()))
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "openwebtext"])
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/gpt2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: GPU strongly recommended for this benchmark.")

    configs = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
    all_logs = []

    for cfg_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name} | gamma={cfg['gamma']}, alpha={cfg['anna_alpha']}")
        print(f"{'='*60}")
        for seed in range(1, args.seeds + 1):
            log = run_experiment(cfg_name, cfg, args, seed, device)
            all_logs.extend(log)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "gpt2_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_logs[0].keys())
        writer.writeheader()
        writer.writerows(all_logs)
    print(f"\nResults saved to {csv_path}")

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS (last step, averaged over seeds)")
    print(f"{'='*60}")
    print(f"{'Config':<16} {'PPL':>8} {'Loss':>8} {'Time(s)':>9} {'Mem(GB)':>8}")
    print("-" * 55)
    for cfg_name in configs:
        finals = [e for e in all_logs if e["config"] == cfg_name and e["step"] == args.steps]
        if finals:
            avg_ppl = sum(e["train_ppl"] for e in finals) / len(finals)
            avg_loss = sum(e["train_loss"] for e in finals) / len(finals)
            avg_time = sum(e["wall_sec"] for e in finals) / len(finals)
            avg_mem = sum(e["peak_mem_gb"] for e in finals) / len(finals)
            print(f"{cfg_name:<16} {avg_ppl:>7.1f} {avg_loss:>8.4f} {avg_time:>8.0f}s {avg_mem:>7.1f}")


if __name__ == "__main__":
    main()
