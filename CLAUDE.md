# ReCogDrive 项目架构文档

## 项目概述

ReCogDrive 是一个面向自动驾驶的视觉-语言模型(VLM)训练框架，采用三阶段训练架构：
1. **阶段一**: VLM监督微调 (VLM SFT)
2. **阶段二**: DiT模仿学习 (Diffusion Transformer Imitation Learning)
3. **阶段三**: DiffGRPO强化学习

## 项目结构

```
recogdrive/
├── src/recogdrive/           # 核心源代码
│   ├── cli.py                # 统一CLI入口
│   ├── vlm/                   # VLM后端 (InternVL, Qwen)
│   │   ├── backends/         # 模型实现
│   │   ├── preprocessing/    # 数据预处理
│   │   └── registry.py       # VLM注册表
│   ├── dit/                  # DiT扩散规划器
│   │   └── blocks/           # DiT基础模块
│   ├── agents/recogdrive/    # ReCogDrive Agent
│   ├── training/             # 三阶段训练入口
│   │   ├── stage1_vlm.py    # VLM监督微调
│   │   ├── stage2_dit.py    # DiT模仿学习
│   │   └── stage3_rl.py     # DiffGRPO强化学习
│   ├── evaluation/           # 评估模块 (DriveBench, DriveLM, LingoQA)
│   └── data/                 # 数据处理 (navsim_common)
├── scripts/                   # 脚本
│   ├── download/             # 数据下载
│   ├── cache_dataset/        # 特征缓存
│   ├── generate_dataset/     # 数据生成
│   └── evaluation/           # 评估脚本
├── configs/                   # 配置文件
├── docker/                    # Docker容器
├── docs/                      # 文档
└── justfile                   # 命令入口
```

## 用户接口

### 使用 just (推荐)
```bash
# 训练
just train-vlm --vlm-path /path/to/InternVL3-8B --data-path /path/to/data
just train-dit --vlm-path /path/to/vlm --data-path /path/to/data
just train-rl --vlm-path /path/to/vlm --dit-path /path/to/dit

# 数据下载
just download-navtrain
just download-all

# 评估
just eval-2b
just eval-8b
```

### 使用 Python CLI
```bash
python -m src.recogdrive.cli train --stage 1 --vlm-path /path/to/model
python -m src.recogdrive.cli download --dataset navsim
```

---

## 阶段一: VLM监督微调

### 模型架构
- **基础模型**: InternVL3-8B (视觉-语言模型)
  - 视觉编码器: InternViT-6B，动态分辨率支持 (448×448 tiles)
  - 语言模型: InternLM2-8B (32层，隐藏维度4096)
  - 投影层: 3层MLP，连接视觉和语言模态，维度3584
- **关键配置参数**:
  - `hidden_size: 3584` - 投影层输出维度
  - `vision_select_layer: -1` - 使用最后一层视觉特征
  - `force_image_size: 448` - 基础tile尺寸
  - `max_dynamic_patch: 16` - 最大动态分块数

### 训练入口
```bash
# just (推荐)
just train-vlm --vlm-path /path/to/InternVL3-8B --data-path /path/to/data

# 或 Python模块
python -m src.recogdrive.training.stage1_vlm --vlm-path /path/to/model
```

### 输出目录
- `outputs/stage1_vlm/` - VLM微调后的模型

---

## 阶段二: DiT模仿学习

### DiT模型配置
- **small配置** (35M参数): num_heads=8, head_dim=48, num_layers=16
- **large配置** (~140M参数): num_heads=32, head_dim=48, num_layers=16
- **输入维度映射**: VLM输出(3584/1536) → feature_encoder → DiT输入(384/1536)

### 训练入口
```bash
just train-dit --vlm-path /path/to/finetuned_vlm --data-path /path/to/data
```

---

## 阶段三: DiffGRPO强化学习

### GRPO算法
- 优化目标: `L_grpo = E[log(π(a|s) / π_ref(a|s)) * A(s,a)] + β * KL(π||π_ref)`
- PDM评分: 综合考虑碰撞、时间TTC、舒适度等因素

### 训练入口
```bash
just train-rl --vlm-path /path/to/vlm --dit-path /path/to/dit
```

---

## 关键技术创新

1. **层次化认知架构**: VLM层 → DiT层 → GRPO层
2. **高效训练系统**: 特征缓存、评分缓存、分布式训练
3. **评估体系**: PDM评分、安全指标、泛化能力
