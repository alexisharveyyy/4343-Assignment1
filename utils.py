"""
Utility functions for CSC4343 Homework 1.
Seeding, plotting, logging, and data helpers.
"""

import os
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed=42):
    """Enforce full determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def save_metrics(metrics, filename):
    """Save metrics dict to JSON in results/."""
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")


def load_metrics(filename):
    """Load metrics dict from JSON in results/."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)


def save_plot(fig, filename):
    """Save a matplotlib figure to plots/."""
    ensure_dirs()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")

# Plotting helpers

def plot_lr_schedule(epochs, lrs, title="Learning Rate Schedule", filename="lr_schedule.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, epochs + 1), lrs, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save_plot(fig, filename)


def plot_accuracy(epochs, train_acc, val_acc, title="Accuracy", filename="accuracy.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = range(1, epochs + 1)
    ax.plot(ep, train_acc, label="Train Acc", linewidth=2)
    ax.plot(ep, val_acc, label="Val Acc", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, filename)


def plot_loss(epochs, train_loss, val_loss, title="Loss", filename="loss.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = range(1, epochs + 1)
    ax.plot(ep, train_loss, label="Train Loss", linewidth=2)
    ax.plot(ep, val_loss, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, filename)


def plot_multi_runs(runs, ylabel="Val Accuracy (%)", title="Multiple Runs",
                    filename="multi_runs.png"):
    """
    runs: list of (label, values) tuples.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, vals in runs:
        ax.plot(range(1, len(vals) + 1), vals, label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, filename)


def print_summary_table(headers, rows):
    """Print a nicely formatted summary table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    print()
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print()
