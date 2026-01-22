# ReCogDrive 常用命令

## 环境设置
```bash
# 安装基础依赖
pip install -e .

# 安装InternVL依赖 (VLM微调)
pip install -r internvl_chat/internvl_chat.txt
```

## 阶段一：VLM监督微调
### 数据准备
```bash
# 生成NAVSIM轨迹数据
cd scripts
sh generate_dataset/generate_internvl_dataset.sh

# 生成QA数据 (需要部署VLM服务)
sh generate_dataset/generate_internvl_dataset_pipeline.sh
```

### 训练VLM
```bash
# 8B模型训练
cd internvl_chat
sh ./shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh

# 配置说明：
# - 修改 model_name_or_path 为InternVL3预训练权重路径
# - 修改 meta_path 指向数据集配置文件
# - 调整GPUS、BATCH_SIZE等硬件参数
```

## 阶段二：DIT模仿学习
### 缓存VLM特征 (节省训练时间)
```bash
# 缓存隐藏状态 (需1-2TB空间)
sh scripts/cache_dataset/run_caching_recogdrive_hidden_state.sh
```

### 训练扩散规划器
```bash
# 8B模型训练
sh scripts/training/run_recogdrive_train_multi_node.sh

# 2B模型训练  
sh scripts/training/run_recogdrive_train_multi_node_2b.sh

# 使用EMA的训练 (更快收敛)
sh scripts/training/run_recogdrive_train_multi_node_ema.sh
```

### 评估
```bash
# PDM分数评估
sh scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh
```

## 阶段三：DiffGRPO强化学习
### 指标缓存
```bash
# 缓存训练集指标
sh scripts/cache_dataset/run_metric_caching_train.sh

# 缓存测试集指标
sh scripts/cache_dataset/run_metric_caching.sh
```

### RL训练
```bash
# 8B模型RL训练
sh scripts/training/run_recogdrive_train_multi_node_rl.sh

# 2B模型RL训练
sh scripts/training/run_recogdrive_train_multi_node_rl_2b.sh
```

## 配置说明
### VLM训练配置 (internvl_chat_finetune.py)
- `model_name_or_path`: InternVL3预训练权重
- `meta_path`: 数据集配置文件路径
- `force_image_size`: 448 (图像分辨率)
- `max_dynamic_patch`: 16 (动态分块数量)
- `num_train_epochs`: 3
- `learning_rate`: 4e-5
- `deepspeed`: "zero_stage1_config.json"

### DIT训练配置 (recogdrive_agent.yaml)
- `dit_type`: 'small' 或 'large' (DiT模型大小)
- `grpo`: False (模仿学习阶段)
- `vlm_type`: 'internvl'
- `sampling_method`: 'ddim'
- `lr`: 1e-4
- `cache_hidden_state`: True (使用缓存特征)

### RL训练配置
- `grpo`: True (启用GRPO)
- `metric_cache_path`: 指标缓存路径
- `reference_policy_checkpoint`: 参考策略检查点路径

## 实用工具命令
```bash
# 查看训练日志
tail -f train_recogdrive_exp.txt

# TensorBoard可视化
tensorboard --logdir internvl_chat/work_dirs/ReCogDrive_pretrain/

# 检查GPU状态
nvidia-smi

# 查看进程
ps aux | grep torchrun
```