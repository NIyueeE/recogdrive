# QwenVL Chat - Stage 1 Fine-tuning

This directory contains the code for Stage 1 supervised fine-tuning of Qwen2.5-VL models for the ReCogDrive project.

## Structure

- `qwenvl/train/qwenvl_chat_finetune.py` - Main training script for Qwen2.5-VL fine-tuning
- `shell/qwen2.5_vl/finetune/qwen2.5_vl_7b_finetune_recogdrive_pretrain.sh` - Distributed training shell script
- `shell/data_info/recogdrive_pretrain.json` - Dataset configuration (shared with InternVL)
- `qwenvl_chat.txt` - Python dependencies
- `zero_stage1_config.json` - DeepSpeed configuration

## Usage

### 1. Install dependencies
```bash
pip install -r qwenvl_chat.txt
```

### 2. Prepare dataset
Ensure the dataset paths in `shell/data_info/recogdrive_pretrain.json` are correctly set.

### 3. Run fine-tuning
```bash
cd qwenvl_chat
sh shell/qwen2.5_vl/finetune/qwen2.5_vl_7b_finetune_recogdrive_pretrain.sh
```

## Model Compatibility

The fine-tuned Qwen2.5-VL-7B-Instruct model has `hidden_size: 3584`, which matches the requirement for Stage 2 (DiT imitation learning) when using `vlm_size='large'`.

## Notes

- This implementation is independent of the original InternVL training code.
- The training script uses Hugging Face's `transformers` library with `Qwen2_5_VLForConditionalGeneration`.
- Data preprocessing follows the same format as InternVL but adapts to Qwen's chat template.
- The fine-tuned model can be directly used as the VLM backbone in Stage 2 and Stage 3 of ReCogDrive.