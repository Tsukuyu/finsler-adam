#!/usr/bin/env python3
"""
CIFAR-10 ResNet-20 Benchmark: AdamW vs Finsler-Adam

Compares four optimizer configurations on CIFAR-10 / ResNet-20:
  1. AdamW (baseline)
  2. Finsler-Adam Full (γ=0.5, α=0.1)
  3. AnnaOnly (γ=0, α=0.1)
  4. FinslerOnly (γ=0.5, α=0)

Outputs:
  - Training/validation loss & accuracy curves
  - Final performance table
  - Wall-clock timing per epoch
  - CSV log for further analysis

Requirements:
    pip install finsler-adam torchvision wandb  # wandb is optional

Usage:
    python cifar10_resnet.py                       # run all configs
    python cifar10_resnet.py --config adamw         # run only AdamW
    python cifar10_resnet.py --seeds 5 --epochs 200 # 5 seeds, 200 epochs
    python cifar10_resnet.py --wandb                # log to W&B
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from finsler_adam import FinslerAdam

# ─────────────────────────── ResNet-20 for CIFAR ────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(in_planes, planes, s))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return self.fc(out.view(out.size(0), -1))


# ─────────────────────────── Optimizer Configs ──────────────────────────────

CONFIGS = {
    "adamw": dict(gamma=0.0, anna_alpha=0.0),
    "finsler_full": dict(gamma=0.5, anna_alpha=0.1),
    "anna_only": dict(gamma=0.0, anna_alpha=0.1),
    "finsler_only": dict(gamma=0.5, anna_alpha=0.0),
}


# ─────────────────────────── Training Loop ──────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def run_experiment(config_name, config, args, seed, device):
    """Run a single training experiment and return the log."""
    torch.manual_seed(seed)

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Model + Optimizer
    model = ResNet20().to(device)
    optimizer = FinslerAdam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        **config,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, device)
        val_loss, val_acc = evaluate(model, testloader, device)
        scheduler.step()
        elapsed = time.time() - t0

        entry = dict(
            config=config_name, seed=seed, epoch=epoch,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc,
            lr=scheduler.get_last_lr()[0], wall_sec=elapsed,
        )
        log.append(entry)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  [{config_name}|s{seed}] Ep {epoch:3d}/{args.epochs} | "
                  f"Train {train_acc:.1f}% | Val {val_acc:.1f}% | "
                  f"Loss {val_loss:.4f} | {elapsed:.1f}s")

    return log


# ─────────────────────────── Main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet-20 Benchmark")
    parser.add_argument("--config", type=str, default="all",
                        choices=["all"] + list(CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/cifar10")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    configs = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
    all_logs = []

    for cfg_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name} | gamma={cfg['gamma']}, alpha={cfg['anna_alpha']}")
        print(f"{'='*60}")
        for seed in range(1, args.seeds + 1):
            log = run_experiment(cfg_name, cfg, args, seed, device)
            all_logs.extend(log)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "cifar10_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_logs[0].keys())
        writer.writeheader()
        writer.writerows(all_logs)
    print(f"\nResults saved to {csv_path}")

    json_path = os.path.join(args.output_dir, "cifar10_results.json")
    with open(json_path, "w") as f:
        json.dump(all_logs, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("FINAL RESULTS (last epoch, averaged over seeds)")
    print(f"{'='*60}")
    print(f"{'Config':<16} {'Val Acc':>8} {'Val Loss':>9} {'Train Acc':>10}")
    print("-" * 50)
    for cfg_name in configs:
        finals = [e for e in all_logs if e["config"] == cfg_name and e["epoch"] == args.epochs]
        avg_val_acc = sum(e["val_acc"] for e in finals) / len(finals)
        avg_val_loss = sum(e["val_loss"] for e in finals) / len(finals)
        avg_train_acc = sum(e["train_acc"] for e in finals) / len(finals)
        print(f"{cfg_name:<16} {avg_val_acc:>7.2f}% {avg_val_loss:>9.4f} {avg_train_acc:>9.2f}%")


if __name__ == "__main__":
    main()
