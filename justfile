# ReCogDrive 项目命令集
# 使用 `just <recipe>` 运行命令，例如 `just train-vlm`
# 查看所有可用命令: `just --list`

# ===========================================================================
# 变量定义
# ===========================================================================

project_root := "."
scripts_dir := "scripts"

# 默认路径 (根据实际情况修改)
vlm_model_path := "/path/to/ckpt/InternVL3-8B"
data_path := "/path/to/data"
output_dir := "./outputs"

# ===========================================================================
# 默认命令
# ===========================================================================

default: help

# ===========================================================================
# 环境设置
# ===========================================================================

# 安装项目依赖
env-install:
    @echo "安装基础依赖..."
    pip install -r requirements.txt

# 安装VLM依赖
env-install-vlm:
    @echo "安装VLM依赖..."
    pip install -r configs/internvl_chat.txt

# 安装项目
env-dev:
    @echo "安装项目 (开发模式)..."
    pip install -e .

# ===========================================================================
# 训练命令 (阶段一、二、三)
# ===========================================================================

# Stage 1: VLM监督微调
train-vlm ARGS='':
    @echo "运行 Stage 1: VLM监督微调..."
    python -m src.recogdrive.training.stage1_vlm {{ARGS}}

# Stage 2: DiT模仿学习
train-dit ARGS='':
    @echo "运行 Stage 2: DiT模仿学习..."
    python -m src.recogdrive.training.stage2_dit {{ARGS}}

# Stage 3: DiffGRPO强化学习
train-rl ARGS='':
    @echo "运行 Stage 3: DiffGRPO强化学习..."
    python -m src.recogdrive.training.stage3_rl {{ARGS}}

# ===========================================================================
# 数据下载
# ===========================================================================

# 下载NAVSIM训练数据
download-navtrain:
    @echo "下载NAVSIM训练数据..."
    bash {{scripts_dir}}/download/download_navtrain.sh

# 下载NAVSIM测试数据
download-test:
    @echo "下载NAVSIM测试数据..."
    bash {{scripts_dir}}/download/download_test.sh

# 下载训练验证数据
download-trainval:
    @echo "下载训练验证数据..."
    bash {{scripts_dir}}/download/download_trainval.sh

# 下载小规模数据
download-mini:
    @echo "下载小规模数据..."
    bash {{scripts_dir}}/download/download_mini.sh

# 下载地图数据
download-maps:
    @echo "下载地图数据..."
    bash {{scripts_dir}}/download/download_maps.sh

# 下载所有数据
download-all:
    @echo "下载所有数据..."
    just download-navtrain
    just download-test
    just download-trainval

# ===========================================================================
# 数据处理
# ===========================================================================

# 生成NAVSIM轨迹数据
data-generate-trajectory:
    @echo "生成NAVSIM轨迹数据..."
    bash {{scripts_dir}}/generate_dataset/generate_internvl_dataset.sh

# 生成QA数据 (需要VLM服务)
data-generate-qa:
    @echo "生成QA数据 (需要部署VLM服务)..."
    bash {{scripts_dir}}/generate_dataset/generate_internvl_dataset_pipeline.sh

# 缓存VLM隐藏状态
cache-hidden-state:
    @echo "缓存VLM隐藏状态 (需要1-2TB存储)..."
    bash {{scripts_dir}}/cache_dataset/run_caching_recogdrive_hidden_state.sh

# 缓存训练集指标
cache-metrics-train:
    @echo "缓存训练集指标..."
    bash {{scripts_dir}}/cache_dataset/run_metric_caching_train.sh

# 缓存测试集指标
cache-metrics-test:
    @echo "缓存测试集指标..."
    bash {{scripts_dir}}/cache_dataset/run_metric_caching.sh

# ===========================================================================
# 评估
# ===========================================================================

# 评估模型 (2B)
eval-2b:
    @echo "评估模型 (2B)..."
    bash {{scripts_dir}}/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh

# 评估模型 (8B)
eval-8b:
    @echo "评估模型 (8B)..."
    bash {{scripts_dir}}/evaluation/run_recogdrive_agent_pdm_score_evaluation_8b.sh

# ===========================================================================
# 实用工具
# ===========================================================================

# GPU状态
gpu:
    nvidia-smi

# 清理缓存
clean:
    @echo "清理缓存文件..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 项目信息
info:
    @echo "ReCogDrive - Vision-Language Model for Autonomous Driving"
    @echo ""
    @echo "三阶段训练:"
    @echo "  1. just train-vlm   # VLM监督微调"
    @echo "  2. just train-dit   # DiT模仿学习"
    @echo "  3. just train-rl     # DiffGRPO强化学习"

# ===========================================================================
# 帮助
# ===========================================================================

help:
    @echo "ReCogDrive 命令帮助"
    @echo "===================="
    @echo ""
    @echo "训练:"
    @echo "  just train-vlm [ARGS]    # Stage 1: VLM监督微调"
    @echo "  just train-dit [ARGS]    # Stage 2: DiT模仿学习"
    @echo "  just train-rl [ARGS]     # Stage 3: DiffGRPO强化学习"
    @echo ""
    @echo "下载数据:"
    @echo "  just download-navtrain   # NAVSIM训练数据"
    @echo "  just download-test       # 测试数据"
    @echo "  just download-all        # 所有数据"
    @echo ""
    @echo "数据处理:"
    @echo "  just data-generate-trajectory  # 生成轨迹"
    @echo "  just cache-hidden-state        # 缓存特征"
    @echo ""
    @echo "评估:"
    @echo "  just eval-2b   # 评估2B模型"
    @echo "  just eval-8b   # 评估8B模型"
    @echo ""
    @echo "查看所有命令: just --list"
