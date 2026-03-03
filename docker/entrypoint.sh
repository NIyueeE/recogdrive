#!/bin/bash
# ReCogDrive Entrypoint Script
# This script is executed when the container starts

set -e

echo "=========================================="
echo "ReCogDrive Training Container"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "=========================================="

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPUs available:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
else
    echo "Warning: nvidia-smi not found"
fi

# Check NCCL
if python -c "import torch; torch.cuda.nccl.version()" &> /dev/null; then
    echo "NCCL version: $(python -c 'import torch.cuda.nccl; print(torch.cuda.nccl.version())')"
fi

# Execute the command passed to docker run
exec "$@"
