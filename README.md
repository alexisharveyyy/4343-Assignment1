# CSC4343 Homework 1: ResNet-18 on CIFAR-10 with Gradient Scaling
# Author: Alexis Harvey

## Setup

```bash
pip install torch torchvision matplotlib tqdm numpy
```

## Usage

Run each part sequentially:

```bash
# Part 1: Hyperparameter sweep 
python train.py --part 1

# Part 2a: Two non-deterministic runs to show randomness
python train.py --part 2a

# Part 2b: Two deterministic runs to verify reproducibility
python train.py --part 2b

# Part 3: Gradient scaling with n=2,4 
python train.py --part 3

# Part 4: Gradient scaling with n=3,5,7 
python train.py --part 4
```

Specify a device explicitly if needed:

```bash
python train.py --part 1 --device cuda
python train.py --part 1 --device mps   # Apple Silicon
python train.py --part 1 --device cpu
```

## Project Structure

```
.
├── train.py      # Main training script (all parts)
├── utils.py      # Helpers: seeding, plotting, logging
├── plots/        # Generated figures (PNG)
├── results/      # Saved metrics (JSON)
└── README.md
```

## Output

- **results/**: JSON files with per-epoch metrics for every training run.
- **plots/**: PNG figures for each part (LR schedules, accuracy curves, loss curves, comparisons).
- **results/best_config.json**: The best hyperparameter config from Part 1, used by all subsequent parts.

## Key Findings

- **Part 2a**: Without seeding, two runs diverge due to non-deterministic GPU operations.
- **Part 2b**: With full determinism enforced, two runs produce identical results.
- **Part 3**: Gradient scaling by n=2,4 (powers of 2) yields identical training because 1/2 and 1/4 are exact in IEEE 754 floating point.
- **Part 4**: Gradient scaling by n=3,5,7 causes tiny floating point errors (1/3, 1/5, 1/7 are non-terminating in binary) that accumulate over training, producing measurably different loss trajectories despite mathematically equivalent updates.
