# ReCogDrive Training Container
# Build: docker build -t recogdrive:latest .
# Run: docker run --gpus all -v /path/to/data:/data recogdrive:latest

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VERSION=11.8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install just command runner
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# Working directory
WORKDIR /app

# Copy requirements files first for better caching
COPY requirements.txt .
COPY configs/internvl_chat.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r internvl_chat.txt

# Install additional training dependencies
RUN pip install deepspeed>=0.9.0 \
    transformers>=4.30.0 \
    accelerate>=0.20.0 \
    bitsandbytes>=0.41.0 \
    xformers>=0.0.20

# Copy project files
COPY . .

# Create directories for data, models and cache
RUN mkdir -p /app/data /app/models /app/cache /app/outputs

# Set working directory
WORKDIR /app

# Default entrypoint
ENTRYPOINT ["/bin/bash"]
