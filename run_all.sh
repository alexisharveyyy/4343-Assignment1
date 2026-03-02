#!/bin/bash
# Run all parts of HW1 sequentially.
# Estimated total time on MPS: ~10-12 hours.
set -e

cd "$(dirname "$0")"
PYTHON=/usr/local/bin/python3

echo "========================================"
echo "  CSC4343 HW1 — Full Run"
echo "  Started: $(date)"
echo "========================================"

echo ""
echo "=== Part 1: Hyperparameter Tuning ==="
$PYTHON train.py --part 1 --sweep-epochs 5 --full-epochs 100

echo ""
echo "=== Part 2a: Randomness ==="
$PYTHON train.py --part 2a

echo ""
echo "=== Part 2b: Reproducibility ==="
$PYTHON train.py --part 2b

echo ""
echo "=== Part 3: Gradient Scaling n=2,4 ==="
$PYTHON train.py --part 3

echo ""
echo "=== Part 4: Gradient Scaling n=3,5,7 ==="
$PYTHON train.py --part 4

echo ""
echo "========================================"
echo "  ALL PARTS COMPLETE"
echo "  Finished: $(date)"
echo "========================================"
