#!/bin/bash
# ReCogDrive Multi-Node Training Script
# Usage:
#   # Start master node (node 0)
#   NODE_NAME=node0 MASTER_ADDR=192.168.1.1 NODE_RANK=0 WORLD_SIZE=2 docker-compose up training-node
#
#   # Start worker node (node 1)
#   NODE_NAME=node1 MASTER_ADDR=192.168.1.1 NODE_RANK=1 WORLD_SIZE=2 docker-compose up training-node

set -e

echo "Starting ReCogDrive multi-node training..."
echo "Node: ${NODE_NAME:-node0}"
echo "Master: ${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500}"
echo "Rank: ${NODE_RANK:-0}"
echo "World size: ${WORLD_SIZE:-1}"

# Export environment variables for distributed training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# If this is not the master node, wait for master to be ready
if [ "${NODE_RANK:-0}" -ne 0 ]; then
    echo "Waiting for master node to be ready..."
    sleep 10
fi

# Run the training command
# Example: torchrun --nproc_per_node=8 training_script.py
echo "Starting training process..."
exec "$@"
