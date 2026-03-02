
"""
CSC4343 Homework 1: ResNet-18 on CIFAR-10 with Gradient Scaling Experiments.

Usage:
    python train.py --part 1        # Hyperparameter tuning
    python train.py --part 2a       # Randomness (non-deterministic)
    python train.py --part 2b       # Reproducibility (deterministic)
    python train.py --part 3        # Gradient scaling n=2,4
    python train.py --part 4        # Gradient scaling n=3,5,7
"""

import argparse
import copy
import itertools
import json
import os
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import (
    ensure_dirs,
    load_metrics,
    plot_accuracy,
    plot_loss,
    plot_lr_schedule,
    plot_multi_runs,
    print_summary_table,
    save_metrics,
    set_seed,
)

# Data

def get_cifar10_loaders(batch_size=128, num_workers=2):
    """Return CIFAR-10 train and val (test set) data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val)

    use_pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin)

    return train_loader, val_loader


# Model

def build_resnet18(num_classes=10):
    """
    Build a ResNet-18 adapted for CIFAR-10 (32x32 images).
    - Replace first conv: 3x3, stride=1, padding=1 (instead of 7x7).
    - Remove the initial max-pool layer.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    # Replace first conv layer
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool by replacing with identity
    model.maxpool = nn.Identity()
    return model

# Training / Evaluation


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


def train_one_epoch_scaled(model, loader, criterion, optimizer, device, n,
                           base_lr, scheduler):
    """
    Train one epoch with gradient scaling by factor n.

    Implements the paper's perturbed SGD: θ = θ - (η/n) * (n * g)

    To avoid the optimizer's internal weight decay being scaled by the
    modified LR, we temporarily zero out weight_decay and apply it manually
    at the original LR.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Get current LR from scheduler (the "base" LR for this epoch)
    current_lr = optimizer.param_groups[0]["lr"]

    # Store original weight decay so we can apply it manually
    original_wd = optimizer.param_groups[0]["weight_decay"]

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Multiply gradients by n: (n * g)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(n)

        # Manually apply weight decay at the original LR before stepping,
        # so it isn't affected by the scaled LR
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    p.data.add_(p.data, alpha=-current_lr * original_wd)

        # Set LR to η/n and disable optimizer's weight decay
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr / n
            pg["weight_decay"] = 0.0

        optimizer.step()

        # Restore original LR and weight decay
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
            pg["weight_decay"] = original_wd

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


# Full training loop

def train_model(config, device, deterministic=False, seed=42, grad_scale_n=1,
                quiet=False):
    """
    Train ResNet-18 on CIFAR-10 with the given config.

    config keys: lr, momentum, weight_decay, batch_size, epochs, scheduler
    Returns a metrics dict with per-epoch stats.
    """
    if deterministic:
        set_seed(seed)

    train_loader, val_loader = get_cifar10_loaders(
        batch_size=config["batch_size"])

    model = build_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    epochs = config["epochs"]
    if config["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif config["scheduler"] == "multistep":
        m1 = int(0.6 * epochs)
        m2 = int(0.8 * epochs)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m1, m2],
                                                    gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")

    metrics = {
        "config": config,
        "grad_scale_n": grad_scale_n,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    iterator = range(1, epochs + 1)
    if not quiet:
        iterator = tqdm(iterator, desc="Training", unit="epoch")

    for epoch in iterator:
        current_lr = optimizer.param_groups[0]["lr"]
        metrics["lr"].append(current_lr)

        if grad_scale_n == 1:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device)
        else:
            train_loss, train_acc = train_one_epoch_scaled(
                model, train_loader, criterion, optimizer, device,
                n=grad_scale_n, base_lr=current_lr, scheduler=scheduler)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        best_val_acc = max(best_val_acc, val_acc)

        if not quiet and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                tr_loss=f"{train_loss:.4f}",
                tr_acc=f"{train_acc:.2f}",
                val_acc=f"{val_acc:.2f}",
                lr=f"{current_lr:.6f}",
            )

    metrics["best_val_acc"] = best_val_acc
    return metrics


# Part 1: Hyperparameter Tuning

def part1(device, sweep_epochs=30, full_epochs=150):
    """
    Two-phase hyperparameter tuning:
      Phase 1: Short sweep (sweep_epochs) across all combos to rank them.
      Phase 2: Full training (full_epochs) with the best config.
    """
    print("=" * 60)
    print("Part 1: Hyperparameter Tuning")
    print("=" * 60)

    # Sweep grid covering the required ranges.
    # Full grid: lr=[0.01,0.05,0.1] x mom=[0.9,0.95] x wd=[1e-4,5e-4]
    #            x bs=[128,256] x sched=[cosine,multistep] = 48 combos.
    # Using representative subset (16 configs) for practical runtime.
    learning_rates = [0.01, 0.1]
    momentums = [0.9, 0.95]
    weight_decays = [1e-4, 5e-4]
    batch_sizes = [128, 256]
    schedulers = ["cosine"]

    combos = list(itertools.product(
        learning_rates, momentums, weight_decays, batch_sizes, schedulers))

    print(f"Phase 1: Quick sweep — {len(combos)} configs x {sweep_epochs} epochs")

    all_results = []
    best_sweep_acc = 0.0
    best_sweep_config = None

    for i, (lr, mom, wd, bs, sched) in enumerate(combos):
        config = {
            "lr": lr,
            "momentum": mom,
            "weight_decay": wd,
            "batch_size": bs,
            "scheduler": sched,
            "epochs": sweep_epochs,
        }
        tag = (f"lr={lr}_mom={mom}_wd={wd}_bs={bs}"
               f"_sched={sched}_ep={sweep_epochs}")
        print(f"\n[{i+1}/{len(combos)}] {tag}")

        metrics = train_model(config, device, deterministic=True, seed=42)

        result_entry = {**config, "best_val_acc": metrics["best_val_acc"]}
        all_results.append(result_entry)

        save_metrics(metrics, f"part1_sweep_{tag}.json")

        if metrics["best_val_acc"] > best_sweep_acc:
            best_sweep_acc = metrics["best_val_acc"]
            best_sweep_config = config

        print(f"  Best val acc this run: {metrics['best_val_acc']:.2f}%")

    # Sweep summary table (top 10)
    sorted_results = sorted(all_results, key=lambda x: x["best_val_acc"],
                            reverse=True)
    headers = ["Rank", "LR", "Mom", "WD", "BS", "Sched", "Sweep Val Acc"]
    rows = []
    for rank, r in enumerate(sorted_results[:10], 1):
        rows.append([
            rank, r["lr"], r["momentum"], r["weight_decay"],
            r["batch_size"], r["scheduler"], f"{r['best_val_acc']:.2f}%",
        ])
    print("\n--- Phase 1 Sweep Results ---")
    print_summary_table(headers, rows)

    # Save sweep summary
    save_metrics({
        "all_results": all_results,
        "best_sweep_config": best_sweep_config,
        "best_sweep_acc": best_sweep_acc,
    }, "part1_sweep_summary.json")

    # Phase 2: full training with the best config
    best_config = {**best_sweep_config, "epochs": full_epochs}
    print(f"\nPhase 2: Full training with best config for {full_epochs} epochs")
    print(f"Config: {best_config}")

    best_metrics = train_model(best_config, device, deterministic=True, seed=42)
    save_metrics(best_metrics, "part1_best_full.json")

    # Save best config separately for later parts
    save_metrics(best_config, "best_config.json")

    # Plots for best config
    ep = best_config["epochs"]
    plot_lr_schedule(ep, best_metrics["lr"],
                     title="Part 1: LR Schedule (Best Config)",
                     filename="part1_lr_schedule.png")
    plot_accuracy(ep, best_metrics["train_acc"], best_metrics["val_acc"],
                  title="Part 1: Accuracy (Best Config)",
                  filename="part1_accuracy.png")
    plot_loss(ep, best_metrics["train_loss"], best_metrics["val_loss"],
              title="Part 1: Loss (Best Config)",
              filename="part1_loss.png")

    print(f"\nBest config: {best_config}")
    print(f"Best validation accuracy: {best_metrics['best_val_acc']:.2f}%")


# Part 2a: Randomness

def part2a(device):
    print("=" * 60)
    print("Part 2a: Randomness (Non-Deterministic)")
    print("=" * 60)

    config = load_metrics("best_config.json")
    print(f"Using config: {config}")

    runs = []
    for run_id in range(1, 3):
        print(f"\n--- Run {run_id} (no seed) ---")
        # Explicitly disable determinism
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

        metrics = train_model(config, device, deterministic=False)
        save_metrics(metrics, f"part2a_run{run_id}.json")
        runs.append((f"Run {run_id}", metrics["val_acc"]))
        print(f"  Best val acc: {metrics['best_val_acc']:.2f}%")

    plot_multi_runs(runs,
                    ylabel="Val Accuracy (%)",
                    title="Part 2a: Non-Deterministic Runs",
                    filename="part2a_val_acc.png")

    # Show divergence
    diff = [abs(a - b) for a, b in zip(runs[0][1], runs[1][1])]
    print(f"\nMax val acc difference between runs: {max(diff):.4f}%")
    print(f"Mean val acc difference: {np.mean(diff):.4f}%")


# Part 2b: Reproducibility

def part2b(device):
    print("=" * 60)
    print("Part 2b: Reproducibility (Deterministic)")
    print("=" * 60)

    config = load_metrics("best_config.json")
    print(f"Using config: {config}")

    runs = []
    for run_id in range(1, 3):
        print(f"\n--- Run {run_id} (seed=42, fully deterministic) ---")
        metrics = train_model(config, device, deterministic=True, seed=42)
        save_metrics(metrics, f"part2b_run{run_id}.json")
        runs.append((f"Run {run_id} (seed=42)", metrics["val_acc"]))
        print(f"  Best val acc: {metrics['best_val_acc']:.2f}%")

    plot_multi_runs(runs,
                    ylabel="Val Accuracy (%)",
                    title="Part 2b: Deterministic Runs (seed=42)",
                    filename="part2b_val_acc.png")

    # Verify exact match
    loss_diff = [abs(a - b) for a, b in
                 zip(runs[0][1], runs[1][1])]
    max_diff = max(loss_diff)
    print(f"\nMax val acc difference: {max_diff}")
    if max_diff == 0.0:
        print("PASS: Runs are exactly identical.")
    else:
        print("WARNING: Runs differ — determinism may not be fully enforced.")

    # Also check loss
    m1 = load_metrics("part2b_run1.json")
    m2 = load_metrics("part2b_run2.json")
    loss_diffs = [abs(a - b) for a, b in zip(m1["val_loss"], m2["val_loss"])]
    print(f"Max val loss difference: {max(loss_diffs)}")


# Part 3: Gradient Scaling n=2,4

def part3(device):
    print("=" * 60)
    print("Part 3: Gradient Scaling n=2, 4 (Powers of 2)")
    print("=" * 60)

    config = load_metrics("best_config.json")
    print(f"Using config: {config}")

    # Baseline n=1 (reuse part2b_run1 if available)
    try:
        baseline = load_metrics("part2b_run1.json")
        print("Loaded baseline (n=1) from Part 2b.")
    except FileNotFoundError:
        print("Training baseline (n=1)...")
        baseline = train_model(config, device, deterministic=True, seed=42,
                               grad_scale_n=1)
        save_metrics(baseline, "part2b_run1.json")

    runs = [("n=1 (baseline)", baseline["val_acc"])]

    for n in [2, 4]:
        print(f"\n--- Gradient scaling n={n} ---")
        metrics = train_model(config, device, deterministic=True, seed=42,
                              grad_scale_n=n)
        save_metrics(metrics, f"part3_n{n}.json")
        runs.append((f"n={n}", metrics["val_acc"]))
        print(f"  Best val acc: {metrics['best_val_acc']:.2f}%")

        # Compare to baseline
        acc_diff = [abs(a - b) for a, b in
                    zip(baseline["val_acc"], metrics["val_acc"])]
        loss_diff = [abs(a - b) for a, b in
                     zip(baseline["val_loss"], metrics["val_loss"])]
        print(f"  Max val acc diff vs baseline: {max(acc_diff):.6f}")
        print(f"  Max val loss diff vs baseline: {max(loss_diff):.10f}")

    plot_multi_runs(runs,
                    ylabel="Val Accuracy (%)",
                    title="Part 3: Gradient Scaling n=1,2,4",
                    filename="part3_val_acc.png")

    # Summary
    headers = ["n", "Best Val Acc", "Max Acc Diff vs n=1", "Max Loss Diff vs n=1"]
    rows = [["1", f"{baseline['best_val_acc']:.2f}%", "0", "0"]]
    for n in [2, 4]:
        m = load_metrics(f"part3_n{n}.json")
        acc_diff = max(abs(a - b) for a, b in
                       zip(baseline["val_acc"], m["val_acc"]))
        loss_diff = max(abs(a - b) for a, b in
                        zip(baseline["val_loss"], m["val_loss"]))
        rows.append([str(n), f"{m['best_val_acc']:.2f}%",
                     f"{acc_diff:.6f}", f"{loss_diff:.10f}"])
    print_summary_table(headers, rows)


# Part 4: Gradient Scaling n=3,5,7

def part4(device):
    print("=" * 60)
    print("Part 4: Gradient Scaling n=3, 5, 7 (Odd Numbers)")
    print("=" * 60)

    config = load_metrics("best_config.json")
    print(f"Using config: {config}")

    # Baseline
    try:
        baseline = load_metrics("part2b_run1.json")
        print("Loaded baseline (n=1) from Part 2b.")
    except FileNotFoundError:
        print("Training baseline (n=1)...")
        baseline = train_model(config, device, deterministic=True, seed=42,
                               grad_scale_n=1)
        save_metrics(baseline, "part2b_run1.json")

    runs = [("n=1 (baseline)", baseline["val_acc"])]
    divergence_data = {}

    for n in [3, 5, 7]:
        print(f"\n--- Gradient scaling n={n} ---")
        metrics = train_model(config, device, deterministic=True, seed=42,
                              grad_scale_n=n)
        save_metrics(metrics, f"part4_n{n}.json")
        runs.append((f"n={n}", metrics["val_acc"]))
        print(f"  Best val acc: {metrics['best_val_acc']:.2f}%")

        # Compare to baseline
        acc_diff = [abs(a - b) for a, b in
                    zip(baseline["val_acc"], metrics["val_acc"])]
        loss_diff = [abs(a - b) for a, b in
                     zip(baseline["val_loss"], metrics["val_loss"])]
        print(f"  Max val acc diff vs baseline: {max(acc_diff):.6f}")
        print(f"  Max val loss diff vs baseline: {max(loss_diff):.10f}")
        print(f"  Final epoch val loss diff:     "
              f"{abs(baseline['val_loss'][-1] - metrics['val_loss'][-1]):.10f}")

        divergence_data[f"n={n}"] = {
            "per_epoch_loss_diff": loss_diff,
            "per_epoch_acc_diff": acc_diff,
            "max_loss_diff": max(loss_diff),
            "max_acc_diff": max(acc_diff),
            "final_loss_diff": abs(
                baseline["val_loss"][-1] - metrics["val_loss"][-1]),
        }

    save_metrics(divergence_data, "part4_divergence.json")

    plot_multi_runs(runs,
                    ylabel="Val Accuracy (%)",
                    title="Part 4: Gradient Scaling n=1,3,5,7",
                    filename="part4_val_acc.png")

    # Plot loss divergence over epochs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in [3, 5, 7]:
        diffs = divergence_data[f"n={n}"]["per_epoch_loss_diff"]
        ax.plot(range(1, len(diffs) + 1), diffs, label=f"n={n}", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|Loss Difference| vs Baseline")
    ax.set_title("Part 4: Loss Divergence from Baseline (n=3,5,7)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    from utils import save_plot
    save_plot(fig, "part4_loss_divergence.png")

    # Summary table
    headers = ["n", "Best Val Acc", "Max Acc Diff", "Max Loss Diff",
               "Final Loss Diff"]
    rows = [["1", f"{baseline['best_val_acc']:.2f}%", "0", "0", "0"]]
    for n in [3, 5, 7]:
        d = divergence_data[f"n={n}"]
        m = load_metrics(f"part4_n{n}.json")
        rows.append([
            str(n),
            f"{m['best_val_acc']:.2f}%",
            f"{d['max_acc_diff']:.6f}",
            f"{d['max_loss_diff']:.10f}",
            f"{d['final_loss_diff']:.10f}",
        ])
    print_summary_table(headers, rows)

    print("\nKey insight: n=3,5,7 involve division by numbers not exactly")
    print("representable in IEEE 754 (1/3, 1/5, 1/7), causing tiny floating")
    print("point errors that accumulate over training, leading to divergent")
    print("trajectories despite mathematically equivalent updates.")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="CSC4343 HW1: ResNet-18 CIFAR-10 Gradient Scaling")
    parser.add_argument("--part", type=str, required=True,
                        choices=["1", "2a", "2b", "3", "4"],
                        help="Which part to run")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    parser.add_argument("--sweep-epochs", type=int, default=30,
                        help="Epochs per config during Part 1 sweep (default: 30)")
    parser.add_argument("--full-epochs", type=int, default=150,
                        help="Epochs for full training in Part 1 (default: 150)")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else
                              "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    ensure_dirs()

    start = time.time()

    if args.part == "1":
        part1(device, sweep_epochs=args.sweep_epochs,
              full_epochs=args.full_epochs)
    elif args.part == "2a":
        part2a(device)
    elif args.part == "2b":
        part2b(device)
    elif args.part == "3":
        part3(device)
    elif args.part == "4":
        part4(device)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()