# Installation for ReCogDrive

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 8+ GPUs with 80GB memory (recommended for training)
- Docker (recommended for environment setup)

## Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Build the container
cd docker
docker build -t recogdrive:latest .

# Run with GPU support
docker run --gpus all -v /path/to/data:/data -it recogdrive:latest
```

The Docker image includes:
- Python 3.10
- CUDA 11.8 + cuDNN 8
- just command runner
- All required dependencies

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/xiaomi-research/recogdrive.git
cd recogdrive

# Install basic dependencies
pip install -r requirements.txt

# Install VLM dependencies
pip install -r configs/internvl_chat.txt

# Install the package in development mode
pip install -e .

# Install just (optional, for convenience)
# macOS: brew install just
# Linux: curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | sudo bash
```

## Project Structure

```
recogdrive/
├── src/recogdrive/           # Core source code
│   ├── cli.py                # CLI entry point
│   ├── vlm/                  # VLM backend
│   ├── dit/                  # DiT diffusion planner
│   └── training/             # Training entry points
├── configs/                  # Configuration files
├── scripts/                  # Scripts
├── docker/                   # Containerization
├── docs/                     # Documentation
└── navsim/                   # NAVSIM simulation
```

## Usage

### Using just (Recommended)

```bash
# Training
just train-vlm --vlm-path /path/to/InternVL3-8B --data-path /path/to/data
just train-dit --vlm-path /path/to/vlm --data-path /path/to/data
just train-rl --vlm-path /path/to/vlm --dit-path /path/to/dit

# Data download
just download-navtrain
just download-all

# Evaluation
just eval-2b
just eval-8b

# View all commands
just --list
```

### Using Python CLI

```bash
python -m src.recogdrive.cli train --stage 1 --vlm-path /path/to/model --data-path /path/to/data
python -m src.recogdrive.cli download --dataset navsim
```

### Using Python Modules

```bash
python -m src.recogdrive.training.stage1_vlm --vlm-path /path/to/model --data-path /path/to/data
```

## Next Steps

See [Train_Eval.md](Train_Eval.md) for detailed training and evaluation instructions.
