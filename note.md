## 阶段一: vlm监督微调

### 模型架构
- **基础模型**: InternVL3-8B (视觉-语言模型)
  - 视觉编码器: InternViT-6B，动态分辨率支持 (448×448 tiles)
  - 语言模型: InternLM2-8B (32层，隐藏维度4096)
  - 投影层: 3层MLP，连接视觉和语言模态，维度3584
- **关键配置参数详解**:
  ```python
  hidden_size: 3584              # 投影层输出维度
  vision_select_layer: -1        # 使用最后一层视觉特征 (layer norm后)
  force_image_size: 448          # 基础tile尺寸
  max_dynamic_patch: 16          # 最大动态分块数 (1×1到4×4)
  down_sample_ratio: 0.5         # 视觉token压缩率 (像素重组)
  drop_path_rate: 0.1            # 视觉编码器DropPath率
  vision_hidden_size: 2560       # InternViT输出维度
  ```
- **动态分辨率机制**:
  - 输入图像根据宽高比动态分割为1-16个448×448 tiles
  - 支持缩略图生成 (`use_thumbnail=True`)
  - 像素重组(pixel unshuffle)减少4倍视觉token

### 可使用数据集
#### 15个开源驾驶QA数据集 (通过`recogdrive_pretrain.json`配置)
1. **Navsim** - NAVSIM轨迹数据 (85,109样本)
2. **Navsim_QA** - NAVSIM生成的QA数据 (85,109样本)
3. **CODA-LM** - 驾驶场景语言建模 (20,318样本)
4. **DriveLM** - 驾驶语言模型基准 (4,072样本)
5. **LingoQA** - 驾驶问答数据集 (26,824样本)
6. **MAPLM** - 地图语言模型 (10,612样本)
7. **Nuinstruct** - NuScenes指令数据集 (57,317样本)
8. **Omnidrive** - 全方位驾驶数据集 (28,010样本)
9. **SUTD** - 交通视频问答 (9,916样本)
10. **Talk2Car** - 指代理解数据集 (8,079样本)
11. **NuScenes-QA** - NuScenes问答 (24,988样本)
12. **Drivegpt4** - 驾驶GPT-4数据 (26,319样本)
13. **Senna** - 驾驶场景理解 (27,813样本)
14. **Drama** - 危险场景分析 (16,404样本)
15. **llava** - 通用VLM数据 (665,298样本，采样率0.2)

#### 数据分布策略详解
```json
"repeat_time": 2,       # NAVSIM轨迹数据重复2次，增强驾驶场景权重
"repeat_time": 1,       # 其他驾驶数据集重复1次
"repeat_time": 0.2,     # LLaVA通用数据重复0.2次，防止过拟合
"data_augment": false   # 所有数据集均未开启数据增强
```

#### 数据格式标准化
```jsonl
{
  "image": "relative/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n驾驶相关问题"},
    {"from": "gpt", "value": "详细答案"}
  ],
  "image_size": [width, height],
  "scene_id": "navsim_001"
}
```

#### NAVSIM生成QA数据流程详解
1. **轨迹预测生成** (`sh scripts/generate_dataset/generate_internvl_dataset.sh`)
   - 逻辑实现: `navsim/planning/script/run_generate_dataset.py`
   - 输入: NAVSIM原始传感器数据 (相机、LiDAR、定位)
   - 输出: 85,109个轨迹样本，包含8步未来轨迹 (4秒，0.5秒间隔)
   - 数据格式:
     ```json
     {
       "scene_id": "scene_001",
       "timestamp": 1234567890,
       "camera_front": "sensor/camera_front.jpg",
       "trajectory": [[x1,y1,yaw1], ..., [x8,y8,yaw8]],
       "velocity": 5.2,
       "acceleration": 0.1
     }
     ```

2. **QA数据生成** (`sh scripts/generate_dataset/generate_internvl_dataset_pipeline.sh`)
   - 逻辑实现: `navsim/planning/script/run_generate_dataset_pipeline.py`
   - VLM服务: Qwen2.5VL-72B (通过vLLM/SGLang部署)
   - 提示工程: 4阶段认知提示 (感知→理解→推理→决策)
   - 质量控制: GPT-4评分过滤，保留评分>7的样本
   - 输出: 85,109个高质量QA对，涵盖驾驶场景理解

### 训练过程
#### 用户接口
- 主要脚本: `sh internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh`
- 逻辑实现: `internvl_chat/internvl/train/internvl_chat_finetune.py`

#### 输入部分
- **预训练模型**: InternVL3-8B (本地路径或HuggingFace仓库)
- **驾驶数据集**: 通过`recogdrive_pretrain.json`配置的15个数据集混合

#### 关键训练参数详解
```bash
# 硬件配置优化
GPUS=8                           # 使用8个A100/H800 GPU
BATCH_SIZE=128                   # 总批量大小 (梯度累积后)
PER_DEVICE_BATCH_SIZE=1          # 每个GPU的批量大小 (受限于序列长度)
GRADIENT_ACC=16                  # 梯度累积步数 (128 / 1 / 8)

# 模型架构配置
--model_name_or_path "/path/to/ckpt/InternVL3-8B"
--conv_style "internvl2_5"       # 对话模板 (支持<image>占位符)
--use_fast_tokenizer False       # 使用慢速但更准确的分词器
--force_image_size 448           # 强制图像处理尺寸
--max_dynamic_patch 16           # 动态分块最大数量
--down_sample_ratio 0.5          # 视觉token下采样比例
--drop_path_rate 0.1             # ViT DropPath正则化
--vision_select_layer -1         # 使用最后一层视觉特征

# 训练优化配置
--num_train_epochs 3             # 训练3个epoch (约360k步)
--learning_rate 4e-5             # AdamW学习率
--weight_decay 0.05              # 权重衰减 (L2正则化)
--warmup_ratio 0.1               # 10%训练步数预热
--lr_scheduler_type "cosine"     # 余弦退火学习率调度
--bf16 True                      # 使用bfloat16混合精度
--grad_checkpoint True           # 梯度检查点 (减少30%显存)
--group_by_length True           # 按序列长度分组 (提升20%吞吐量)

# 数据处理配置
--dynamic_image_size True        # 启用动态图像尺寸处理
--use_thumbnail True             # 为多tile图像生成缩略图
--max_seq_length 12288           # 最大序列长度 (包括图像token)
--dataloader_num_workers 32      # 数据加载工作进程数

# 分布式训练配置
--deepspeed "zero_stage1_config.json"  # DeepSpeed ZeRO Stage 1
--report_to "tensorboard"        # 日志记录到TensorBoard
--logging_steps 1                # 每步记录日志

# 多节点训练配置 (脚本中设置)
--nnodes=8                       # 8个计算节点
--node_rank=$MLP_ROLE_INDEX      # 节点排名 (0-7)
--master_addr=$MLP_WORKER_0_HOST # 主节点地址
--nproc_per_node=8               # 每节点8个GPU进程
```

#### DeepSpeed配置详解 (`zero_stage1_config.json`)
```json
{
  "zero_optimization": {
    "stage": 1,                    # ZeRO Stage 1 (优化器状态分区)
    "allgather_partitions": true,  # 启用分区收集
    "allgather_bucket_size": 1e9,  # 参数收集桶大小 (1GB)
    "overlap_comm": true,          # 重叠通信与计算
    "reduce_scatter": true,        # 启用reduce-scatter操作
    "reduce_bucket_size": 1e9,     # 梯度规约桶大小 (1GB)
    "contiguous_gradients": true   # 连续梯度内存
  },
  "fp16": {
    "enabled": "auto",             # 自动检测FP16支持
    "auto_cast": true,             # 自动类型转换
    "loss_scale": 0,               # 动态损失缩放
    "initial_scale_power": 32,     # 初始缩放因子
    "loss_scale_window": 1000,     # 损失缩放窗口大小
    "hysteresis": 2,               # 滞后阈值
    "min_loss_scale": 1            # 最小损失缩放
  },
  "bf16": {
    "enabled": "auto"              # 自动检测BF16支持
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",                # 学习率由训练脚本指定
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": "auto"       # 权重衰减由训练脚本指定
    }
  },
  "gradient_accumulation_steps": "auto",  # 自动推断梯度累积步数
  "gradient_clipping": "auto",    # 自动梯度裁剪
  "steps_per_print": 2000,        # 每2000步打印日志
  "train_batch_size": "auto",     # 自动推断训练批量大小
  "train_micro_batch_size_per_gpu": "auto",  # 自动推断每GPU微批量大小
  "wall_clock_breakdown": true    # 启用时间分解分析
}
```

#### 训练策略深度分析
1. **参数冻结策略**:
   ```python
   freeze_llm=False     # 训练整个语言模型 (8B参数)
   freeze_mlp=False     # 训练MLP投影层 (10M参数)
   freeze_backbone=False # 训练视觉编码器 (6B参数)
   # 总可训练参数: ~14B (全参数微调)
   ```

2. **动态分辨率训练**:
   - 支持448×448 tiles，适应不同尺寸输入图像
   - 视觉token处理流程:
     ```
     原始图像 → 动态分块 → 448×448 tiles → ViT编码 → 像素重组 → 投影层 → LLM
     ```

3. **序列打包优化**:
   - `group_by_length=True`: 将相似长度样本打包，减少填充token
   - 提升训练效率: 128序列长度提升20%吞吐量

4. **混合精度训练**:
   - BF16精度: 保留动态范围，避免梯度下溢
   - 梯度检查点: 用计算换显存，支持更长序列

5. **多节点分布式训练**:
   - 8节点×8GPU = 64 GPU并行训练
   - ZeRO Stage 1: 优化器状态分区，减少内存占用
   - NCCL优化: IB/RoCE网络，高速GPU间通信

#### 输出目录
- 路径: `internvl_chat/work_dirs/ReCogDrive_pretrain/internvl3_8b_finetune_full_recogdrive_pretrain`
- 输出内容: 驾驶领域适配的VLM模型

#### 输出目录结构详解
```
internvl_chat/work_dirs/ReCogDrive_pretrain/internvl3_8b_finetune_full_recogdrive_pretrain/
├── config.json                          # InternVLChatConfig (hidden_size: 3584)
├── pytorch_model.bin                    # 微调后的模型权重 (14B参数)
├── tokenizer.json                       # 分词器模型 (包含特殊token <image>)
├── tokenizer_config.json                # 分词器配置 (use_fast=False)
├── special_tokens_map.json              # 特殊token映射 (bos_token, eos_token)
├── training_args.bin                    # 训练参数序列化 (包含所有超参数)
├── trainer_state.json                   # 训练器状态 (global_step: 360000)
├── metrics.json                         # 训练指标 (loss曲线, 学习率变化)
├── training_log.txt                     # 完整训练日志 (Stdout重定向)
├── checkpoint-200/                      # 检查点目录 (每200步保存)
│   ├── pytorch_model.bin
│   ├── optimizer.pt                     # 优化器状态
│   └── scheduler.pt                     # 调度器状态
└── tensorboard/                         # TensorBoard日志
    ├── events.out.tfevents.*
    └── scalars.json                     # 标量指标记录
```

## 阶段二: DIT模仿学习

### 数据集架构详解
- **NAVSIM-Traj 轨迹数据**:
  - 来源: NAVSIM仿真数据集中的车辆轨迹 (85,109个样本)
  - 格式: JSONL文件，包含历史轨迹和未来轨迹标签
  ```json
  {
    "scene_id": "scene_001",
    "timestamp": 1234567890,
    "camera_front": "sensor/camera_front.jpg",
    "ego_pose": {"x": 10.5, "y": 20.3, "yaw": 1.57},
    "trajectory": [  # 8步未来轨迹 (4秒，0.5秒间隔)
      {"x": 10.8, "y": 20.5, "yaw": 1.57},
      {"x": 11.1, "y": 20.7, "yaw": 1.56},
      ...
    ],
    "velocity": 5.2,
    "acceleration": 0.1
  }
  ```

- **缓存特征数据系统**:
  - VLM特征提取: `last_hidden_state` (维度: small-1536, large-3584) - 此为VLM输出维度，DiT输入维度不同
  - 用户接口: `sh scripts/cache_dataset/run_caching_recogdrive_hidden_state.sh`
  - 预计算流程:
    ```python
    # 特征提取流程
    image → VLM视觉编码器 → 视觉特征 → MLP投影 → 隐藏状态 → 缓存
    # 存储格式: NumPy .npy文件 + JSON元数据
    ```
  - 存储需求: 1-2 TB (85k样本 × 1.5KB × 16倍压缩)
  - 加速效果: 训练速度提升100倍 (跳过图像编码)

### 训练过程
- 用户接口 `sh scripts/training/run_recogdrive_train_multi_node.sh` (8B模型) 或 `sh scripts/training/run_recogdrive_train_multi_node_2b.sh` (2B模型)

- 逻辑实现 `navsim/planning/script/run_training_recogdrive.py`

- 输入部分:
  - 阶段一微调后的VLM模型 (作为特征提取器)
  - NAVSIM轨迹数据集
  - 可选: 预计算的VLM特征缓存

- **配置参数深度解析** (`navsim/planning/script/config/common/agent/recogdrive_agent.yaml`):
  ```yaml
  _target_: navsim.agents.recogdrive.recogdrive_agent.ReCogDriveAgent
  _convert_: 'all'
  
  # 模型架构配置
  dit_type: 'small'               # DiT模型大小: 'small'(8头) 或 'large'(32头)，训练脚本中使用dit_type='small'
  vlm_type: 'internvl'            # VLM类型: 'internvl' 或 'qwen'
  sampling_method: 'ddim'         # 扩散采样方法: 'ddim'(5步) 或 'dpm'(10步)
  
  # 训练配置
  lr: 1e-4                        # 学习率 (AdamW优化器)
  grpo: False                     # 模仿学习阶段设为False
  cache_hidden_state: True        # 使用预计算VLM特征
  
  # 路径配置
  vlm_path: ''                    # VLM模型路径 (必需)
  checkpoint_path: ''             # DiT检查点路径 (恢复训练)
  metric_cache_path: ''           # PDM评分缓存路径 (RL阶段使用)
  
  # 相机配置
  cam_type: 'single'              # 相机类型: 'single'(前视) 或 'multi'(多视角)
  vlm_size: 'small'               # VLM尺寸: 'small'(2B) 或 'large'(8B)，训练脚本中8B模型使用vlm_size='large'
  train_backbone: false           # 是否训练VLM骨干 (默认false)
  ```

- **DIT模型配置详解** (`navsim/agents/recogdrive/recogdrive_agent.py:make_recogdrive_config`):
  ```python
  # small配置 (35M参数)
  num_heads: 8, head_dim: 48, num_layers: 16, output_dim: 512
  
  # large配置 (~140M参数)
  num_heads: 32, head_dim: 48, num_layers: 16, output_dim: 1536
  
  # 通用参数
  dropout: 0.0                   # 无Dropout (扩散模型特性)
  attention_bias: True           # 注意力偏置
  norm_eps: 1e-5                 # LayerNorm epsilon
  interleave_attention: True     # 交错注意力机制
  ```

- **VLM架构约束详解** (`navsim/agents/recogdrive/recogdrive_diffusion_planner.py`):
  ```python
  # 代码中的硬编码维度映射
  if config.vlm_size == "large":
      self.feature_encoder = nn.Linear(3584, config.input_embedding_dim)  # 3584→1536
  else:
      self.feature_encoder = nn.Linear(1536, config.input_embedding_dim)  # 1536→384
  ```

  - **实际架构约束**:
    ```
    大型配置 (large):
    VLM输出维度: 3584 → feature_encoder投影 → 1536 → DiT输入

    小型配置 (small):
    VLM输出维度: 1536 → feature_encoder投影 → 384 → DiT输入
    ```

  - **配置参数对应关系**:
    ```yaml
    # 必须保持一致的参数对
    dit_type: "large"     # DiT输入1536维
    vlm_size: "large"     # VLM输出3584维

    dit_type: "small"     # DiT输入384维
    vlm_size: "small"     # VLM输出1536维
    ```

  **训练脚本参数组合**:
  - 8B模型: `dit_type="small"`, `vlm_size="large"`
  - 2B模型: `dit_type="small"`, `vlm_size="small"`

  - **对新VLM的适配要求**:
    1. **输出维度匹配**: 新VLM的`last_hidden_state`维度必须恰好为3584或1536
    2. **接口兼容**: 必须实现相同的前向传播接口，返回包含`hidden_states[-1]`的对象
    3. **特征提取**: `last_hidden_state`形状必须为`[batch_size, seq_len, hidden_dim]`
    4. **配置一致**: `vlm_type`, `vlm_size`参数必须与模型实际尺寸对应

- **扩散规划器架构**:
  ```python
  class ReCogDriveDiffusionPlannerConfig:
      action_dim: 3              # 动作维度 (x, y, yaw)
      action_horizon: 8          # 动作步数 (8步=4秒)
      input_embedding_dim: 384   # 输入嵌入维度 (small-384, large-1536) - 注意：此为DiT输入维度，VLM输出维度不同
      sampling_method: 'ddim'    # 采样方法
      num_inference_steps: 5     # 推理步数 (DDIM-5步, DPM-10步)
      model_dtype: 'float16'     # 模型精度
  ```

- **输出目录结构**:
  ```
  output_dir/                    # 由训练脚本指定
  ├── version_0/                 # 实验版本
  │   ├── checkpoints/
  │   │   ├── epoch=199-step=20000.ckpt    # 最终检查点
  │   │   └── last.ckpt                    # 最后检查点 (EMA)
  │   ├── hparams.yaml           # 超参数配置
  │   └── metrics.csv            # 训练指标记录
  ├── config.yaml                # Hydra配置合并
  └── train_recogdrive_exp.txt   # 训练日志
  ```

- **训练流程优化**:
  1. **数据加载**: 从缓存加载VLM特征，跳过图像编码
  2. **前向传播**: VLM特征 → 条件编码器 → DiT → 轨迹预测
  3. **损失计算**: MSE损失 between 预测轨迹和真实轨迹
  4. **反向传播**: 仅训练DiT参数，VLM参数冻结
  5. **采样验证**: 每epoch使用DDIM采样验证轨迹质量

## 阶段三: DiffGRPO强化学习

### 训练目标深度解析
- **Diffusion Group Relative Policy Optimization (DiffGRPO)**:
  - 基于阶段二的模仿学习策略进行强化学习优化
  - 优化目标: 提高驾驶安全性和舒适性，减少碰撞
  - 核心思想: 在模仿学习基础上进行策略提升

- **GRPO算法公式**:
  ```python
  # 优化目标函数
  L_grpo = E[log(π(a|s) / π_ref(a|s)) * A(s,a)] + β * KL(π||π_ref)
  
  # 其中:
  # π: 当前策略 (DiT规划器)
  # π_ref: 参考策略 (模仿学习checkpoint)
  # A(s,a): 优势函数 (基于PDM评分计算)
  # β: KL散度系数 (控制策略偏离程度)
  ```

### 训练过程
- 用户接口 `sh scripts/training/run_recogdrive_train_multi_node_rl.sh` (8B模型) 或 `sh scripts/training/run_recogdrive_train_multi_node_rl_2b.sh` (2B模型)

- 逻辑实现 `navsim/planning/script/run_training_recogdrive_rl.py`

- 输入部分:
  - 阶段二训练的扩散规划器checkpoint
  - NAVSIM轨迹数据集
  - 预计算的metric缓存 (用于PDM评分)

- **配置参数详解** (GRPO相关):
  ```yaml
  # GRPO特定配置
  grpo: True                     # 启用GRPO训练
  metric_cache_path: "/path/to/metric_cache_dir"  # PDM评分缓存路径
  reference_policy_checkpoint: "/path/to/IL_Model.ckpt"  # 参考策略checkpoint
  
  # 训练调整
  trainer.params.max_epochs: 10  # RL训练轮数 (相比IL的200轮大幅减少)
  dataloader.params.batch_size: 8  # 更小的批量大小 (策略梯度需求)
  ```

- **PDM评分系统**:
  ```python
  # PDM评分公式 (来自代码实现)
  PDM Score = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12

  # 各指标说明
  NC: No at-fault Collisions          # 无责任碰撞 (避免与其他物体/车辆碰撞)
  DAC: Drivable Area Compliance       # 可行驶区域合规性 (保持在可行驶区域内)
  TTC: Time to Collision              # 碰撞时间 (与其他车辆保持安全距离)
  EP: Ego Progress                    # 自我车辆进度 (确保车辆前进不被卡住)
  C: Comfort                          # 舒适度 (避免急转弯和突然减速)
  DDC: Driving Direction Compliance   # 行驶方向合规性 (与预期行驶方向对齐)

  # 评分计算优化
  - 离线预计算: 避免在线计算开销
  - 缓存格式: {scene_id: {timestamp: pdm_score}}
  - 更新策略: 每RL迭代重新评分
  ```

- **训练策略调整**:
  1. **KL系数调度**: 初始β=0.1，随训练衰减
  2. **优势估计**: 使用GAE (Generalized Advantage Estimation)
  3. **策略更新**: PPO-style裁剪，避免过大更新
  4. **价值函数**: 添加价值头估计状态价值

- **输出目录结构**:
  ```
  rl_output_dir/                 # RL训练输出目录
  ├── version_0/
  │   ├── checkpoints/
  │   │   ├── epoch=9-step=1000.ckpt      # RL最终检查点
  │   │   └── best_pdm_score.ckpt         # 最佳PDM评分检查点
  │   ├── rewards.csv           # 奖励曲线记录
  │   ├── kl_divergence.csv     # KL散度监控
  │   └── advantages.csv        # 优势函数统计
  ├── config.yaml               # RL专用配置
  └── rl_training_log.txt       # RL训练日志
  ```

### 关键技术创新总结
1. **层次化认知架构**:
   - VLM层: 场景理解和语义表示
   - DiT层: 连续轨迹生成和优化
   - GRPO层: 安全强化和策略提升

2. **高效训练系统**:
   - 特征缓存: 解耦VML训练和规划器训练
   - 评分缓存: 加速RL奖励计算
   - 分布式训练: 支持多节点大规模训练

3. **评估体系**:
   - PDM评分: 综合性能评估
   - 安全指标: 碰撞率、舒适度
   - 泛化能力: 跨场景、跨数据集评估
