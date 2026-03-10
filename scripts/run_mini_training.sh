#!/bin/bash
# Run mini training for pipeline validation

set -e

echo "=========================================="
echo "ReCogDrive Mini Training Pipeline Test"
echo "=========================================="

# Create necessary directories
mkdir -p data/mini
mkdir -p outputs/mini

# Run training with mock model
echo "Running training with mock mode..."
python3 run_mini_train.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "Check outputs/mini/training_summary.json"
echo "=========================================="
