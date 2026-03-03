# ReCogDrive Training and Evaluation

## Three-Stage Training

ReCogDrive uses a unified three-stage training architecture:

1. **Stage 1**: VLM Supervised Fine-tuning
2. **Stage 2**: DiT Imitation Learning
3. **Stage 3**: DiffGRPO Reinforcement Learning

---

## Stage 1: VLM Supervised Fine-tuning

### Download Data and Prepare

```bash
# Download NAVSIM data
just download-navtrain
just download-test

# Generate trajectory data
just data-generate-trajectory
```

### Training

```bash
# Using just (recommended)
just train-vlm --vlm-path /path/to/InternVL3-8B --data-path /path/to/data --num_gpus 8

# Or using Python
python -m src.recogdrive.training.stage1_vlm \
    --vlm-path /path/to/InternVL3-8B \
    --data-path /path/to/data \
    --output-dir ./outputs/stage1_vlm \
    --num_gpus 8 \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 4e-5
```

---

## Stage 2: DiT Imitation Learning

### Cache Hidden States

```bash
just cache-hidden-state
```

### Training

```bash
# Using just
just train-dit --vlm-path /path/to/finetuned_vlm --data-path /path/to/navsim_data --num_gpus 8

# Or using Python
python -m src.recogdrive.training.stage2_dit \
    --vlm-path /path/to/finetuned_vlm \
    --data-path /path/to/navsim_data \
    --output-dir ./outputs/stage2_dit \
    --num_gpus 8 \
    --num_epochs 200 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --dit-type small \
    --vlm-size large
```

---

## Stage 3: DiffGRPO Reinforcement Learning

### Cache Metrics

```bash
just cache-metrics-train
just cache-metrics-test
```

### Training

```bash
# Using just
just train-rl --vlm-path /path/to/vlm --dit-path /path/to/dit --metric-cache /path/to/metrics

# Or using Python
python -m src.recogdrive.training.stage3_rl \
    --vlm-path /path/to/finetuned_vlm \
    --dit-path /path/to/stage2_checkpoint \
    --metric-cache /path/to/metrics \
    --output-dir ./outputs/stage3_rl \
    --num_gpus 8 \
    --num_epochs 10
```

---

## Evaluation

```bash
# 2B model
just eval-2b

# 8B model
just eval-8b
```

---

## Training Parameters

| Stage | Model | Epochs | Batch Size | Learning Rate | GPUs |
|-------|-------|--------|------------|---------------|------|
| 1 | VLM (InternVL3-8B) | 3 | 8 | 4e-5 | 8 |
| 2 | DiT | 200 | 16 | 1e-4 | 8 |
| 3 | DiT (GRPO) | 10 | 8 | 1e-5 | 8 |

---

## Docker

```bash
# Build
cd docker
docker build -t recogdrive:latest .

# Run
docker run --gpus all -v /path/to/data:/data recogdrive:latest \
    just train-vlm --vlm-path /path/to/model --data-path /data
```
