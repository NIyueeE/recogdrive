# Requirements Document: VLA Model Training and Simulation Setup

## 1. Background
We are developing a Vision-Language-Action (VLA) model for robotic manipulation tasks. The model integrates visual perception, natural language understanding, and motor command generation. To validate the training pipeline and ensure compatibility with our development environment, we need to establish a lightweight training setup that can run on a local virtual machine (Pop!_OS 24.04) and later be containerized using Docker.

This project is a refactoring of an upstream project located at `../input/recogdrive`, which can be referenced for implementation details and baseline configurations.

## 2. Objectives
- **Enable training initiation**: Configure the existing codebase to start training on a small subset of data (not full dataset) without requiring a full-scale training run.
- **Host environment validation**: Successfully execute a training test on the host VM (Pop!_OS 24.04) with aggressive optimizations allowed (bypass permissions, YOLO mode).
- **Dockerization**: After host validation, create a Dockerfile that replicates the training environment and can run the same training test inside a container.
- **Ensure reproducibility**: The Docker container should produce identical results (or functionally equivalent) to the host run.
- **(Optional) Qwen2.5-VL model test**: After completing the primary tasks, attempt to configure and test the pipeline with `Qwen2.5-VL-7B-Instruct` as the first-stage VLM. **Important**: Due to severe VM resource constraints, do **not** actually download the full model weights or use the complete dataset during this test. The goal is to verify configuration compatibility and code paths without incurring heavy resource usage (e.g., by using stub weights or skipping weight loading entirely).

## 3. Constraints
- **No full training or full dataset**: Due to resource and time limitations, the training test must use a minimal dataset (e.g., a few samples) and a reduced model configuration (e.g., smaller backbone, fewer epochs). The goal is to verify the pipeline, not to train a performant model.
- **Host environment specifics**: 
  - OS: Pop!_OS 24.04 (Ubuntu-based)
  - Virtual machine with full privileges; “unconditionally allow aggressive operations” means we can bypass safety checks, use root, install packages freely, etc.
- **Docker target**: The final Dockerfile must work on any Linux host with Docker installed; it should not rely on host-specific paths or settings.
- **Resource limitations**: The VM does **not** support downloading large model weights (e.g., 7B) or processing full datasets. Any test involving large models must be simulated or limited to configuration validation only.

## 4. Scope
This task covers:
- Setting up the training code and dependencies on the host.
- Modifying configuration files to use a minimal dataset and reduced training parameters.
- Running a short training iteration (e.g., 1-5 steps) to confirm that forward/backward passes work, logging occurs, and checkpoints are saved.
- Writing a Dockerfile that installs all dependencies, copies the code and minimal data, and can execute the same training command.
- Testing the Docker build and run to ensure consistency.
- Optionally, verifying the ability to switch the first-stage VLM to `Qwen2.5-VL-7B-Instruct` without actually downloading weights or running full training (e.g., by stubbing the model loading or using a tiny placeholder).

Out of scope:
- Full dataset preparation.
- Model performance evaluation.
- Production deployment.
- Actual download or use of the full Qwen2.5-VL-7B model.

## 5. Detailed Requirements

### 5.1 Host Environment Setup (Pop!_OS 24.04 VM)
- **System packages**: Install essential build tools, CUDA drivers (if GPU available), and Python 3.10+.
- **Python environment**: Use a virtual environment (venv/conda) to manage dependencies listed in `requirements.txt` (or similar). Reference the upstream project at `../input/recogdrive` for dependency hints if needed.
- **Codebase**: Clone the latest VLA model repository from the designated internal GitLab.
- **Data**: Place a minimal dataset sample (e.g., 5 episodes) in a known location (e.g., `/home/user/vla_data/mini`). The dataset format should match what the dataloader expects (e.g., RLDS, TFDS, or custom).
- **Configuration**: Create a training config override (e.g., YAML/JSON) that:
  - Sets `dataset_path` to the mini dataset.
  - Reduces `batch_size` to 1 or 2.
  - Limits `num_epochs` to 1 and `max_steps` to 10.
  - Uses a smaller model variant (e.g., tiny ViT, smaller LLM) if supported.
  - Disables any heavy augmentation or evaluation.
- **Execution**: Run the training command (e.g., `python train.py --config configs/mini.yaml`). Monitor logs to ensure no errors.

### 5.2 Aggressive Permissions (YOLO mode)
Since the host VM allows unconditional operations, we can:
- Disable any safety prompts (e.g., `--dangerously-skip-permissions` in Claude Code if used for automation).
- Run training with `--allow-root` if needed.
- Use `--force` flags to overwrite existing outputs.
- Ensure all file writes are permitted.

### 5.3 Dockerfile Development
- **Base image**: Use an official CUDA-enabled image (e.g., `nvidia/cuda:12.1-runtime-ubuntu22.04`) or a Python slim image, depending on GPU requirements.
- **Dependencies**: Copy `requirements.txt` and install Python packages. Include system libraries (e.g., ffmpeg, libgl1) if required by OpenCV or video processing. Refer to `./input/recogdrive/Dockerfile` (if exists) for potential base configurations.
- **Code**: Copy the entire repository (or only necessary modules) into the container.
- **Data**: Either copy the mini dataset into the image (if small) or mount it as a volume at runtime.
- **Working directory**: Set `WORKDIR` to the code location.
- **Entrypoint/CMD**: Define the command to run the training with the mini config. Use `CMD` to allow overriding.

### 5.4 Validation Criteria
- **Host test**: Training script runs without errors for at least 10 steps. Loss values are computed and logged. Checkpoint files are created.
- **Docker build**: `docker build -t vla-mini-test .` succeeds.
- **Docker run**: `docker run --gpus all vla-mini-test` (if GPU) runs the same training steps without errors. Logs should be similar to host run (allowing for minor numerical differences due to environment).
- **Reproducibility**: If possible, compare final loss values or checkpoint contents between host and container (optional but desirable).

### 5.5 Optional: Testing with Qwen2.5-VL-7B-Instruct (Configuration Validation Only)
- **Goal**: Ensure the training pipeline can be configured to use `Qwen2.5-VL-7B-Instruct` as the VLM without actually downloading the 7B weights or training with it.
- **Method**:
  - Modify the training config to specify `model: "Qwen2.5-VL-7B-Instruct"`.
  - Implement a mock or stub for model loading that bypasses actual weight download (e.g., by using a tiny dummy model or skipping the `from_pretrained` call).
  - Run the training script with `--dry-run` or a similar flag to verify that configuration parsing and data flow work up to the point of model instantiation.
  - Ensure no network requests are made to download the large model.
- **Note**: This step is only required if time permits and does not replace the primary mini training test. The VM's resource constraints must be strictly observed—no actual 7B model loading.

## 6. Deliverables
- A working training configuration for mini dataset.
- A documented procedure (or script) for running the training on host.
- A `Dockerfile` that builds a container capable of running the same training.
- A brief report (or comments in the Dockerfile) explaining any deviations or assumptions.
- (Optional) Notes on how to configure the pipeline for Qwen2.5-VL-7B-Instruct without actual weight download, if attempted.

## 7. Assumptions
- The VLA codebase is modular and allows easy config overrides.
- The dataset format is consistent and can be subsampled without breaking the dataloader.
- GPU support is available on the host; if not, CPU-only training is acceptable for the test.
- The host has internet access to download dependencies during setup.
- The upstream project at `../input/recogdrive` provides useful reference for architecture and dependencies.

## 8. Timeline & Milestones
- **Day 1**: Host environment setup and mini training test.
- **Day 2**: Dockerfile creation and initial build.
- **Day 3**: Docker validation and final adjustments.
- **(Optional) Day 4**: Attempt Qwen2.5-VL configuration validation if time allows.

## 9. Risks and Mitigations
- **Dependency conflicts**: Use a clean Python environment and pin versions in `requirements.txt`. Consider using `conda` lock files.
- **Data path issues**: Ensure the dataset location in config is either absolute or relative to the code root; for Docker, use a consistent mount point.
- **GPU incompatibility**: If the host GPU drivers differ from the Docker base image, the container may fail to access GPU. Use `nvidia-docker` runtime and ensure driver versions match.
- **Accidental large model download**: For the optional Qwen test, explicitly disable internet access during the run or implement checks to prevent downloading weights. Use environment variables or flags to force stub mode.

## 10. Glossary
- **VLA**: Vision-Language-Action model.
- **Mini dataset**: A small, manually curated subset of the full dataset, used only for pipeline testing.
- **YOLO mode**: Running with all safety checks disabled (from Claude Code context).
- **Pop!_OS 24.04**: Ubuntu-based Linux distribution.
- **Qwen2.5-VL-7B-Instruct**: A 7-billion parameter vision-language model by Alibaba; referenced here for optional testing.

---

**Prepared by**: [@NIyueeE]  
**Date**: [2026/03/10]  
**Version**: 1.1 (updated with upstream reference and optional Qwen test)
