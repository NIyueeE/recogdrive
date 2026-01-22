# ReCogDrive 项目概述

## 项目目的
ReCogDrive（Reinforced Cognitive Framework for End-to-End Autonomous Driving）是一个用于端到端自动驾驶的强化认知框架。它通过整合自回归模型和扩散规划器，统一了驾驶理解和规划。

## 核心创新
1. **层次化数据管道**：模拟人类驾驶员的顺序认知过程（生成、精炼、质量控制）
2. **语言-动作对齐**：将VLM学习的驾驶先验注入扩散规划器，生成连续稳定的轨迹
3. **Diffusion Group Relative Policy Optimization (DiffGRPO)**：强化学习阶段提升驾驶安全性和舒适性

## 三阶段训练流程
1. **阶段一：VLM监督微调** - 在驾驶QA数据上微调InternVL3模型
2. **阶段二：DIT模仿学习** - 使用扩散规划器进行轨迹预测模仿学习
3. **阶段三：DiffGRPO强化学习** - 基于模仿学习策略进行强化学习优化

## 技术栈
- **Python 3.9**
- **PyTorch** + **PyTorch Lightning**
- **InternVL3** (视觉语言模型)
- **Diffusion Transformer (DiT)** (扩散规划器)
- **NAVSIM** (自动驾驶仿真平台)
- **DeepSpeed** (分布式训练)
- **Hydra** (配置管理)

## 项目结构
```
recogdrive/
├── internvl_chat/          # VLM微调相关代码
├── navsim/                # NAVSIM集成和规划器
├── scripts/              # 训练、评估、缓存脚本
├── docs/                 # 文档
├── vqa_evaluation/       # 视觉问答评估
└── tutorial/            # 教程
```

## 数据集
使用15个驾驶相关数据集，包括：
- NAVSIM-Traj, NAVSIM-QA (自有生成)
- DriveLM, LingoQA, NuInstruct, NuScenes-QA
- Omnidrive, SUTD, Talk2Car, DriveGPT4
- Senna, Drama, MAPLM, CODA-LM
- LLaVA (通用VLM数据，采样率0.2)

## 硬件要求
- **GPU**: 8+ GPUs (用于分布式训练)
- **存储**: 1-2 TB (用于特征缓存)
- **内存**: 大容量RAM用于数据处理