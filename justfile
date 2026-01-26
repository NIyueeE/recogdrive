# ReCogDrive 项目命令集
# 使用 `just <recipe>` 运行命令，例如 `just data-generate-trajectory`
# 查看所有可用命令: `just --list`

# ===========================================================================
# 变量定义
# ===========================================================================

# 项目根目录 (相对于justfile位置)
project_root := "."

# 脚本目录
scripts_dir := "scripts"

# VLM微调脚本目录
vlm_finetune_dir := "internvl_chat/shell/internvl3.0/2nd_finetune"

# VLM微调主脚本
vlm_finetune_script := "internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh"

# QwenVL微调脚本目录
qwen_vlm_finetune_dir := "qwenvl_chat/shell/qwen2.5_vl/finetune"

# QwenVL微调主脚本
qwen_vlm_finetune_script := "qwen2.5_vl_7b_finetune_recogdrive_pretrain.sh"

# 数据集配置
dataset_config := "recogdrive_pretrain.json"

# 默认VLM模型路径 (根据实际情况修改)
vlm_model_path := "/path/to/ckpt/InternVL3-8B"

# 默认QwenVL模型路径
qwen_vlm_model_path := "Qwen/Qwen2.5-VL-7B-Instruct"

# 默认输出目录
output_dir := "internvl_chat/work_dirs/ReCogDrive_pretrain"

# 默认QwenVL输出目录
qwen_output_dir := "qwenvl_chat/work_dirs/ReCogDrive_pretrain"

# ===========================================================================
# 默认命令
# ===========================================================================

# 默认命令: 直接运行 `just` 时显示帮助信息
default: help

# ===========================================================================
# 环境设置
# ===========================================================================

# 安装项目依赖
env-install:
    @echo "安装基础依赖..."
    pip install -e .

# 安装VLM相关依赖
env-install-vlm:
    @echo "安装InternVL依赖..."
    pip install -r internvl_chat/internvl_chat.txt

# 安装QwenVL相关依赖
env-install-qwen-vlm:
    @echo "安装QwenVL依赖..."
    pip install -r qwenvl_chat/qwenvl_chat.txt

# 显示项目结构
env-structure:
    @echo "项目结构:"
    @tree -d -L 3

# ===========================================================================
# 数据生成
# ===========================================================================

# 生成NAVSIM轨迹数据
data-generate-trajectory:
    @echo "生成NAVSIM轨迹数据..."
    cd {{scripts_dir}} && sh generate_dataset/generate_internvl_dataset.sh

# 生成QA数据 (需要部署VLM服务)
data-generate-qa:
    @echo "生成QA数据 (需要部署VLM服务)..."
    cd {{scripts_dir}} && sh generate_dataset/generate_internvl_dataset_pipeline.sh

# 列出可用的数据集
data-list-datasets:
    @echo "ReCogDrive使用的15个驾驶数据集:"
    @echo "1. Navsim - NAVSIM轨迹数据"
    @echo "2. Navsim_QA - NAVSIM生成的QA数据"
    @echo "3. CODA-LM - 驾驶场景语言建模"
    @echo "4. DriveLM - 驾驶语言模型基准"
    @echo "5. LingoQA - 驾驶问答数据集"
    @echo "6. MAPLM - 地图语言模型"
    @echo "7. Nuinstruct - NuScenes指令数据集"
    @echo "8. Omnidrive - 全方位驾驶数据集"
    @echo "9. SUTD - 交通视频问答"
    @echo "10. Talk2Car - 指代理解数据集"
    @echo "11. NuScenes-QA - NuScenes问答"
    @echo "12. Drivegpt4 - 驾驶GPT-4数据"
    @echo "13. Senna - 驾驶场景理解"
    @echo "14. Drama - 危险场景分析"
    @echo "15. llava - 通用VLM数据 (采样率0.2)"

# ===========================================================================
# 阶段一: VLM监督微调
# ===========================================================================

# VLM微调训练 (8B模型)
vlm-finetune:
    @echo "运行VLM微调训练 (InternVL3-8B)..."
    @echo "注意: 请确保已设置正确的模型路径和数据集配置"
    cd {{vlm_finetune_dir}} && sh {{vlm_finetune_script}}

# 显示VLM训练配置
vlm-show-config:
    @echo "VLM训练关键配置:"
    @echo "模型路径: {{vlm_model_path}}"
    @echo "数据集配置: {{dataset_config}}"
    @echo "图像尺寸: 448"
    @echo "动态分块: 16"
    @echo "训练轮数: 3"
    @echo "学习率: 4e-5"
    @echo "分布式训练: DeepSpeed ZeRO Stage 1"

# 启动VLM训练TensorBoard
vlm-tensorboard:
    @echo "启动TensorBoard监控VLM训练..."
    tensorboard --logdir {{output_dir}}

# 查看VLM训练日志
vlm-logs:
    @echo "查看VLM训练日志..."
    @if [ -f "{{output_dir}}/internvl3_8b_finetune_full_recogdrive_pretrain/training_log.txt" ]; then \
        tail -f {{output_dir}}/internvl3_8b_finetune_full_recogdrive_pretrain/training_log.txt; \
    else \
        echo "训练日志文件不存在"; \
    fi

# QwenVL微调训练 (7B模型)
qwen-vlm-finetune:
    @echo "运行QwenVL微调训练 (Qwen2.5-VL-7B)..."
    @echo "注意: 请确保已设置正确的模型路径和数据集配置"
    cd {{qwen_vlm_finetune_dir}} && sh {{qwen_vlm_finetune_script}}

# 显示QwenVL训练配置
qwen-vlm-show-config:
    @echo "QwenVL训练关键配置:"
    @echo "模型路径: {{qwen_vlm_model_path}}"
    @echo "数据集配置: {{dataset_config}}"
    @echo "训练轮数: 3"
    @echo "学习率: 4e-5"
    @echo "分布式训练: DeepSpeed ZeRO Stage 1"
    @echo "序列长度: 12288"

# 启动QwenVL训练TensorBoard
qwen-vlm-tensorboard:
    @echo "启动TensorBoard监控QwenVL训练..."
    tensorboard --logdir {{qwen_output_dir}}

# 查看QwenVL训练日志
qwen-vlm-logs:
    @echo "查看QwenVL训练日志..."
    @if [ -f "{{qwen_output_dir}}/qwen2.5_vl_7b_finetune_full_recogdrive_pretrain/training_log.txt" ]; then \
        tail -f {{qwen_output_dir}}/qwen2.5_vl_7b_finetune_full_recogdrive_pretrain/training_log.txt; \
    else \
        echo "训练日志文件不存在"; \
    fi

# ===========================================================================
# 特征缓存
# ===========================================================================

# 缓存VLM隐藏状态特征 (节省训练时间)
cache-hidden-state:
    @echo "缓存VLM隐藏状态特征 (需要1-2TB存储空间)..."
    sh {{scripts_dir}}/cache_dataset/run_caching_recogdrive_hidden_state.sh

# 缓存训练集指标 (用于RL阶段)
cache-metrics-train:
    @echo "缓存训练集指标..."
    sh {{scripts_dir}}/cache_dataset/run_metric_caching_train.sh

# 缓存测试集指标 (用于RL阶段)
cache-metrics-test:
    @echo "缓存测试集指标..."
    sh {{scripts_dir}}/cache_dataset/run_metric_caching.sh

# 查看缓存状态
cache-status:
    @echo "检查缓存目录状态..."
    @find . -name "*.npy" -type f | head -20 | xargs -I {} du -h {} | sort -hr

# ===========================================================================
# 阶段二: DIT模仿学习
# ===========================================================================

# DIT模仿学习训练 (8B模型)
dit-train-8b:
    @echo "运行DIT模仿学习训练 (8B模型)..."
    sh {{scripts_dir}}/training/run_recogdrive_train_multi_node.sh

# DIT模仿学习训练 (2B模型)
dit-train-2b:
    @echo "运行DIT模仿学习训练 (2B模型)..."
    sh {{scripts_dir}}/training/run_recogdrive_train_multi_node_2b.sh

# DIT模仿学习训练 with EMA (8B模型)
dit-train-ema-8b:
    @echo "运行DIT模仿学习训练 with EMA (8B模型)..."
    sh {{scripts_dir}}/training/run_recogdrive_train_multi_node_ema.sh

# DIT模仿学习训练 with EMA (2B模型)
dit-train-ema-2b:
    @echo "运行DIT模仿学习训练 with EMA (2B模型)..."
    sh {{scripts_dir}}/training/run_recogdrive_train_multi_node_ema_2b.sh

# 显示DIT训练配置
dit-show-config:
    @echo "DIT模仿学习关键配置:"
    @echo "模型类型: small (8头注意力)"
    @echo "VLM类型: internvl"
    @echo "采样方法: ddim (5步推理)"
    @echo "学习率: 1e-4"
    @echo "缓存特征: true"
    @echo "GRPO: false (模仿学习阶段)"

# ===========================================================================
# 阶段三: DiffGRPO强化学习
# ===========================================================================

# DiffGRPO强化学习训练 (8B模型)
rl-train-8b:
    @echo "运行DiffGRPO强化学习训练 (8B模型)..."
    sh {{scripts_dir}}/training/run_recogdrive_train_multi_node_rl.sh

# DiffGRPO强化学习训练 (2B模型)
rl-train-2b:
    @echo "运行DiffGRPO强化学习训练 (2B模型)..."
    sh {{scripts_dir}}/training/run_recogdrive_train_multi_node_rl_2b.sh

# 显示RL训练配置
rl-show-config:
    @echo "DiffGRPO强化学习关键配置:"
    @echo "GRPO: true"
    @echo "参考策略: 模仿学习检查点"
    @echo "指标缓存: 必需"
    @echo "训练轮数: 10 (相比IL的200轮大幅减少)"
    @echo "批量大小: 8 (更小的批量大小)"

# ===========================================================================
# 评估
# ===========================================================================

# 评估ReCogDrive代理 (8B模型)
eval-recogdrive-8b:
    @echo "评估ReCogDrive代理 (8B模型)..."
    sh {{scripts_dir}}/evaluation/run_recogdrive_agent_pdm_score_evaluation_8b.sh

# 评估ReCogDrive代理 (2B模型)
eval-recogdrive-2b:
    @echo "评估ReCogDrive代理 (2B模型)..."
    sh {{scripts_dir}}/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh

# 评估InternVL代理
eval-internvl:
    @echo "评估InternVL代理..."
    sh {{scripts_dir}}/evaluation/run_internvl_agent_pdm_score_evaluation.sh

# 评估Qwen代理
eval-qwen:
    @echo "评估Qwen代理..."
    sh {{scripts_dir}}/evaluation/run_qwen_agent_pdm_score_evaluation.sh

# 评估TransFuser
eval-transfuser:
    @echo "评估TransFuser..."
    sh {{scripts_dir}}/evaluation/run_transfuser.sh

# 评估Ego-MLP代理
eval-ego-mlp:
    @echo "评估Ego-MLP代理..."
    sh {{scripts_dir}}/evaluation/run_ego_mlp_agent_pdm_score_evaluation.sh

# 评估Human代理
eval-human:
    @echo "评估Human代理..."
    sh {{scripts_dir}}/evaluation/run_human_agent_pdm_score_evaluation.sh

# 交叉验证PDM分数评估
eval-cv-pdm:
    @echo "交叉验证PDM分数评估..."
    sh {{scripts_dir}}/evaluation/run_cv_pdm_score_evaluation.sh

# ===========================================================================
# 可视化
# ===========================================================================

# 绘制轨迹分数
vis-trajectory-score:
    @echo "绘制轨迹分数..."
    sh {{scripts_dir}}/visualization/plt_traj_score_one.sh

# 生成视频可视化
vis-video:
    @echo "生成视频可视化..."
    sh {{scripts_dir}}/visualization/plt_video.sh

# ===========================================================================
# 实用工具
# ===========================================================================

# 检查GPU状态
utils-gpu:
    @echo "检查GPU状态..."
    nvidia-smi

# 查看训练进程
utils-processes:
    @echo "查看训练进程..."
    ps aux | grep torchrun | grep -v grep || echo "没有找到torchrun进程"

# 清理pycache文件
utils-clean:
    @echo "清理pycache文件..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 显示项目信息
utils-info:
    @echo "ReCogDrive项目信息:"
    @echo "项目根目录: {{project_root}}"
    @echo "三阶段训练流程:"
    @echo "  1. VLM监督微调 (阶段一)"
    @echo "  2. DIT模仿学习 (阶段二)"
    @echo "  3. DiffGRPO强化学习 (阶段三)"
    @echo "使用 `just --list` 查看所有可用命令"

# 列出所有可用数据集
utils-list-all:
    @just --list

# ===========================================================================
# 工作流命令
# ===========================================================================

# 完整工作流: 数据生成 → VLM微调 → 特征缓存 → DIT训练 → RL训练
workflow-full:
    @echo "=== 开始完整ReCogDrive工作流 ==="
    @echo "1. 数据生成..."
    @just data-generate-trajectory
    @echo "2. VLM微调..."
    @just vlm-finetune
    @echo "3. 特征缓存..."
    @just cache-hidden-state
    @echo "4. DIT模仿学习..."
    @just dit-train-8b
    @echo "5. 指标缓存..."
    @just cache-metrics-train
    @echo "6. DiffGRPO强化学习..."
    @just rl-train-8b
    @echo "=== 工作流完成 ==="

# 快速测试工作流 (跳过数据生成)
workflow-test:
    @echo "=== 开始测试工作流 ==="
    @echo "1. VLM微调 (使用现有数据)..."
    @just vlm-finetune
    @echo "2. DIT模仿学习..."
    @just dit-train-2b
    @echo "3. 评估..."
    @just eval-recogdrive-2b
    @echo "=== 测试工作流完成 ==="

# 评估工作流
workflow-eval:
    @echo "=== 开始评估工作流 ==="
    @echo "1. 评估ReCogDrive (8B)..."
    @just eval-recogdrive-8b
    @echo "2. 评估ReCogDrive (2B)..."
    @just eval-recogdrive-2b
    @echo "3. 评估基线模型..."
    @just eval-internvl
    @just eval-qwen
    @just eval-transfuser
    @echo "=== 评估工作流完成 ==="

# ===========================================================================
# 帮助信息
# ===========================================================================

# 显示帮助信息
help:
    @echo "ReCogDrive项目命令帮助"
    @echo "======================"
    @echo ""
    @echo "基本使用:"
    @echo "  just <recipe>      # 运行特定命令"
    @echo "  just --list        # 列出所有可用命令"
    @echo ""
    @echo "主要工作流:"
    @echo "  just workflow-full    # 完整工作流 (数据→VLM→DIT→RL)"
    @echo "  just workflow-test    # 测试工作流 (跳过数据生成)"
    @echo "  just workflow-eval    # 评估工作流"
    @echo ""
    @echo "阶段一 (VLM微调):"
    @echo "  just vlm-finetune     # VLM监督微调训练 (InternVL)"
    @echo "  just vlm-show-config  # 显示VLM训练配置"
    @echo "  just vlm-tensorboard  # 启动TensorBoard"
    @echo "  just qwen-vlm-finetune     # QwenVL监督微调训练"
    @echo "  just qwen-vlm-show-config  # 显示QwenVL训练配置"
    @echo "  just qwen-vlm-tensorboard  # 启动QwenVL TensorBoard"
    @echo ""
    @echo "阶段二 (DIT模仿学习):"
    @echo "  just dit-train-8b     # DIT训练 (8B模型)"
    @echo "  just dit-train-2b     # DIT训练 (2B模型)"
    @echo "  just dit-show-config  # 显示DIT训练配置"
    @echo ""
    @echo "阶段三 (DiffGRPO强化学习):"
    @echo "  just rl-train-8b      # RL训练 (8B模型)"
    @echo "  just rl-train-2b      # RL训练 (2B模型)"
    @echo "  just rl-show-config   # 显示RL训练配置"
    @echo ""
    @echo "数据与缓存:"
    @echo "  just data-generate-trajectory  # 生成轨迹数据"
    @echo "  just cache-hidden-state        # 缓存VLM特征"
    @echo "  just cache-metrics-train       # 缓存训练集指标"
    @echo ""
    @echo "评估:"
    @echo "  just eval-recogdrive-8b  # 评估ReCogDrive (8B)"
    @echo "  just eval-recogdrive-2b  # 评估ReCogDrive (2B)"
    @echo "  just eval-internvl       # 评估InternVL基线"
    @echo "  just eval-qwen           # 评估QwenVL基线"
    @echo ""
    @echo "更多命令请使用: just --list"