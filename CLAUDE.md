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
│   ├── __init__.py
│   ├── cli.py                # 统一CLI入口
│   ├── config/               # 配置管理
│   │   ├── base.py          # 配置类定义 (VLMConfig, DiTConfig, RLConfig, DataConfig)
│   │   └── loader.py        # 配置加载器
│   ├── vlm/                  # VLM后端 (注册表模式)
│   │   ├── __init__.py
│   │   ├── base.py          # VLM基类
│   │   ├── factory.py       # VLM工厂
│   │   ├── registry.py     # VLM注册表
│   │   ├── internvl.py     # InternVL实现
│   │   ├── qwen.py         # QwenVL实现
│   │   ├── conversation.py # 对话模板
│   │   ├── dist_utils.py   # 分布式工具
│   │   ├── backends/model/  # 模型后端
│   │   │   ├── internvl_chat/
│   │   │   ├── internlm2/
│   │   │   └── phi3/
│   │   └── preprocessing/patch/  # 训练补丁
│   │       └── patch/       # 各种训练补丁 (flash_attn, packed_training等)
│   ├── dit/                  # DiT扩散规划器
│   │   ├── __init__.py
│   │   ├── recogdrive_dit.py       # DiT主模型
│   │   ├── recogdrive_diffusion_planner.py
│   │   └── blocks/          # DiT基础模块
│   │       ├── attention.py  # 注意力机制
│   │       ├── encoder.py    # 编码器
│   │       ├── rmsnorm.py    # RMSNorm
│   │       └── rope.py       # RoPE位置编码
│   ├── training/             # 三阶段训练入口
│   │   ├── __init__.py
│   │   ├── stage1_vlm.py    # VLM监督微调
│   │   ├── stage2_dit.py    # DiT模仿学习
│   │   └── stage3_rl.py     # DiffGRPO强化学习
│   ├── agents/recogdrive/    # ReCogDrive Agent
│   │   ├── recogdrive_agent.py
│   │   ├── recogdrive_backbone.py
│   │   ├── recogdrive_features.py
│   │   └── utils/
│   ├── data/navsim_common/   # 数据处理
│   │   ├── dataclasses.py
│   │   ├── dataloader.py
│   │   └── enums.py
│   ├── evaluation/           # 评估模块
│   │   └── vqa_evaluation/
│   │       ├── DriveBench/
│   │       ├── DriveLM/
│   │       └── LingoQA/
│   └── utils/internvl_tools/ # 工具脚本
├── scripts/                   # 脚本
│   ├── download/             # 数据下载
│   ├── cache_dataset/        # 特征缓存
│   ├── generate_dataset/     # 数据生成
│   └── evaluation/           # 评估脚本
├── configs/                   # 配置文件
│   ├── __init__.py
│   ├── default.yaml          # 默认配置
│   ├── stage1.yaml           # Stage1配置
│   ├── stage2.yaml           # Stage2配置
│   ├── stage3.yaml          # Stage3配置
│   ├── internvl_chat.txt     # VLM依赖
│   ├── deepspeed/            # DeepSpeed配置
│   │   ├── zero_stage1_config.json
│   │   ├── zero_stage2_config.json
│   │   ├── zero_stage3_config.json
│   │   └── ...
│   ├── vlm/                  # VLM模型配置
│   │   ├── internvl3_8b.yaml
│   │   └── qwen2_vl_7b.yaml
│   └── data/
├── docker/                    # Docker容器
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── entrypoint.sh
│   └── run_multi_node.sh
├── docs/                      # 文档
├── navsim/                    # NAVSIM数据集模块
├── requirements.txt           # Python依赖
├── setup.py                   # 安装脚本
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

# 环境设置
just env-install
just env-install-vlm
```

### 使用 Python CLI

```bash
python -m src.recogdrive.cli train --stage 1 --vlm-path /path/to/model --data-path /path/to/data
python -m src.recogdrive.cli download --dataset navsim
python -m src.recogdrive.cli eval --model-path /path/to/model
```

### 使用 Python 模块

```bash
python -m src.recogdrive.training.stage1_vlm --vlm-path /path/to/model --data-path /path/to/data
python -m src.recogdrive.training.stage2_dit --vlm-path /path/to/vlm --data-path /path/to/data
python -m src.recogdrive.training.stage3_rl --vlm-path /path/to/vlm --dit-path /path/to/dit
```

---

## 核心模块详解

### 1. 配置管理 (config/)

采用层次化配置设计：

- **TrainingConfig**: 统一训练配置，包含所有子配置
- **VLMConfig**: VLM模型配置 (vlm_type, vlm_model_path, force_image_size等)
- **DiTConfig**: DiT模型配置 (dit_type, input_embedding_dim, action_dim等)
- **RLConfig**: 强化学习配置 (num_samples, kl_coef等)
- **DataConfig**: 数据配置 (data_path, cache_dir, max_seq_length等)

### 2. VLM注册表 (vlm/)

采用注册表模式，支持多VLM后端：

```python
from src.recogdrive.vlm import create_vlm

# 创建VLM实例
vlm = create_vlm("internvl", model_path="/path/to/model")
vlm = create_vlm("qwen", model_path="/path/to/model")
```

支持的VLM:
- **InternVL**: InternVL3-8B (视觉-语言模型)
- **Qwen**: Qwen2-VL

### 3. DiT扩散规划器 (dit/)

- **ReCogDriveDiT**: DiT主模型
- **ReCogDriveDiffusionPlanner**: 扩散规划器
- **Blocks**: 基础模块 (attention, encoder, rmsnorm, rope)

---

## 阶段一: VLM监督微调

### 模型架构
- **基础模型**: InternVL3-8B (视觉-语言模型)
  - 视觉编码器: InternViT-6B，动态分辨率支持 (448×448 tiles)
  - 语言模型: InternLM2-8B (32层，隐藏维度4096)
  - 投影层: 3层MLP，连接视觉和语言模态，维度3584

### 关键配置参数
- `hidden_size: 3584` - 投影层输出维度
- `vision_select_layer: -1` - 使用最后一层视觉特征
- `force_image_size: 448` - 基础tile尺寸
- `max_dynamic_patch: 16` - 最大动态分块数

### 训练参数
- `num_epochs`: 训练轮数 (默认3)
- `batch_size`: 批大小 (默认8)
- `learning_rate`: 学习率 (默认4e-5)
- `weight_decay`: 权重衰减 (默认0.05)

### 输出目录
- `outputs/stage1_vlm/` - VLM微调后的模型

---

## 阶段二: DiT模仿学习

### DiT模型配置
- **small配置** (35M参数): num_heads=8, head_dim=48, num_layers=16
- **large配置** (~140M参数): num_heads=32, head_dim=48, num_layers=16
- **输入维度映射**: VLM输出(3584/1536) → feature_encoder → DiT输入(384/1536)

### 扩散设置
- **采样方法**: DDPM, DDIM, Flow
- **推理步数**: 默认5步

---

## 阶段三: DiffGRPO强化学习

### GRPO算法
- 优化目标: `L_grpo = E[log(π(a|s) / π_ref(a|s)) * A(s,a)] + β * KL(π||π_ref)`
- PDM评分: 综合考虑碰撞、时间TTC、舒适度等因素

### 关键参数
- `num_samples`: 轨迹采样数 (默认8)
- `kl_coef`: KL散度系数 (默认0.1)
- `gamma_denoising`: 去噪优势估计 (默认0.6)

---

## 测试与开发

### 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-cov pytest-mock

# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_recogdrive.py::TestConfig -v

# 生成覆盖率报告
python -m pytest tests/ --cov=src.recogdrive
```

### 低配置测试

项目支持在CPU/显存不足的环境下进行Mock测试，验证代码逻辑正确性，不需要真实模型权重。

---

## 关键技术创新

1. **层次化认知架构**: VLM层 → DiT层 → GRPO层
2. **VLM注册表模式**: 支持多VLM后端动态切换
3. **高效训练系统**: 特征缓存、评分缓存、分布式训练
4. **评估体系**: PDM评分、安全指标、泛化能力
