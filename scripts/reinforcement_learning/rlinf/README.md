# RLinf Integration for IsaacLab

This module integrates [RLinf](https://github.com/RLinf/RLinf.git)'s distributed RL training framework with IsaacLab, enabling **reinforcement learning fine-tuning of Vision-Language-Action (VLA) models** (e.g., GR00T, OpenVLA) on IsaacLab simulation tasks.

## Overview

RLinf is a flexible and scalable open-source RL infrastructure designed for Embodied and Agentic AI. This integration allows IsaacLab users to:

- Fine-tune pretrained VLA models on IsaacLab tasks using PPO / Actor-Critic / SAC
- Leverage RLinf's FSDP-based distributed training across multiple GPUs/nodes
- Define observation/action mappings from IsaacLab to GR00T format via YAML config
- Register IsaacLab tasks into RLinf without modifying RLinf source code

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         RLinf Runner                           │
│                 (EmbodiedRunner / EvalRunner)                  │
├────────────────┬──────────────────────┬────────────────────────┤
│  Actor Worker  │   Rollout Worker     │      Env Worker        │
│  (FSDP)        │  (HF Inference)      │  (IsaacLab Sim)        │
│                │                      │                        │
│ Policy         │  Multi-step rollout  │ IsaacLabGenericEnv     │
│ Update         │  with VLA model      │  ├─ _make_env_function │
│                │                      │  ├─ _wrap_obs          │
│                │                      │  └─ _wrap_action       │
└────────────────┴──────────────────────┴────────────────────────┘
```

**Data flow:**
1. `EnvWorker` runs IsaacLab simulation and converts observations to RLinf format
2. `RolloutWorker` runs VLA model inference (e.g., GR00T) to produce actions
3. Actions are converted back to IsaacLab format and stepped in the environment
4. `ActorWorker` updates the VLA model with PPO/actor-critic loss via FSDP

## Directory Structure

```
scripts/reinforcement_learning/rlinf/
├── README.md               # This file
├── __init__.py              # Module docstring and entry point examples
├── train.py                 # Training entry point (launches RLinf distributed training)
├── play.py                  # Evaluation entry point (launches RLinf eval runner)
├── run.sh                   # Shell wrapper for training (sets env vars, PYTHONPATH)
├── play.sh                  # Shell wrapper for evaluation
├── cli_args.py              # CLI argument definitions and config override utilities
├── config/                  # Hydra YAML configs
│   ├── isaaclab_ppo_gr00t_assemble_trocar.yaml   # Full task config (top-level)
│   ├── env/
│   │   └── isaaclab_assemble_trocar.yaml          # Env defaults (base)
│   └── model/
│       └── gr00t.yaml                             # GR00T model defaults
├── policy/                  # GR00T data config for custom embodiments
│   ├── __init__.py
│   └── gr00t_config.py      # IsaacLabDataConfig (modality, transforms)
└── logs/                    # Training/eval logs (auto-generated)
```

**Extension module** (in `source/isaaclab_rl/`):

```
source/isaaclab_rl/isaaclab_rl/rlinf/
├── __init__.py
└── extension.py             # RLinf extension: task registration, obs/action conversion
```

## Prerequisites

- **IsaacLab** installed and configured
- **RLinf** available in the parent repo (expected at `../../rlinf` relative to IsaacLab root)
- **GR00T** model (for VLA inference and data transforms)
- A **pretrained VLA checkpoint** in HuggingFace format (e.g., GR00T fine-tuned model)
- Multi-GPU setup recommended (FSDP requires at least 1 GPU)

## Quick Start

### Training

```bash
# Basic training with a config file
bash run.sh --config_name isaaclab_ppo_gr00t_assemble_trocar

# Training with task override
bash run.sh --config_name isaaclab_ppo_gr00t_assemble_trocar \
    --task Isaac-Install-Trocar-G129-Dex3-RLinf-v0

# List available tasks
bash run.sh --list_tasks
```

### Evaluation

```bash
# Evaluate a trained checkpoint
bash play.sh --config_name isaaclab_ppo_gr00t_assemble_trocar \
    --model_path /path/to/checkpoint

# Evaluate with video recording
bash play.sh --config_name isaaclab_ppo_gr00t_assemble_trocar \
    --model_path /path/to/checkpoint --video
```

## Configuration

Configuration uses [Hydra](https://hydra.cc/) with YAML composition. The top-level config file composes from `env/` and `model/` defaults.

### Top-Level Config Structure

```yaml
defaults:
  - env/isaaclab_assemble_trocar@env.train    # Env defaults → env.train
  - env/isaaclab_assemble_trocar@env.eval     # Env defaults → env.eval
  - model/gr00t@actor.model                   # Model defaults → actor.model

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all                    # Co-locate all workers

runner:
  max_epochs: 1000
  logger:
    logger_backends: ["tensorboard"]          # Options: tensorboard, wandb, swanlab

algorithm:
  loss_type: actor_critic                     # Options: actor_critic, embodied_sac
  update_epoch: 4
  clip_ratio_high: 0.2
  gamma: 0.99
  gae_lambda: 0.95

env:
  train:
    total_num_envs: 4
    max_episode_steps: 256
    isaaclab: &isaaclab_config                # IsaacLab ↔ RLinf mapping (see below)
      ...
  eval:
    isaaclab: *isaaclab_config                # Reuse via YAML anchor

actor:
  training_backend: "fsdp"
  micro_batch_size: 2
  global_batch_size: 4
  optim:
    lr: 5e-6

rollout:
  backend: "huggingface"
  model:
    model_path: "/path/to/pretrained/model"
```

### IsaacLab ↔ RLinf Observation/Action Mapping

The `env.train.isaaclab` section defines how IsaacLab observations are converted to GR00T format. This is the key configuration block for adapting new tasks:

```yaml
isaaclab:
  # Task description for language conditioning
  task_description: "assemble trocar from box"

  # --- IsaacLab → RLinf observation mapping ---
  main_images: "front_camera"               # Single main camera → (B, H, W, C)
  extra_view_images:                         # Extra cameras → (B, N, H, W, C)
    - "left_wrist_camera"
    - "right_wrist_camera"
  states:                                    # State specs with optional slicing
    - key: "robot_joint_state"
      slice: [15, 29]                        # Take indices 15..29
    - key: "robot_dex3_joint_state"          # Full tensor (no slice)

  # --- RLinf → GR00T format conversion ---
  gr00t_mapping:
    video:
      main_images: "video.room_view"
      extra_view_images:
        - "video.left_wrist_view"
        - "video.right_wrist_view"
    state:                                   # Slice concatenated state into GR00T keys
      - gr00t_key: "state.left_arm"
        slice: [0, 7]
      - gr00t_key: "state.right_arm"
        slice: [7, 14]
      - gr00t_key: "state.left_hand"
        slice: [14, 21]
      - gr00t_key: "state.right_hand"
        slice: [21, 28]

  # --- GR00T → IsaacLab action conversion ---
  action_mapping:
    prefix_pad: 15                           # Pad zeros for uncontrolled joints
    suffix_pad: 0

  # --- GR00T model config ---
  obs_converter_type: "isaaclab"
  embodiment_tag: "new_embodiment"
  embodiment_tag_id: 31
  data_config_class: "policy.gr00t_config:IsaacLabDataConfig"
```

## CLI Arguments

### Common Arguments

| Argument | Description |
|---|---|
| `--config_name` | Name of the Hydra config file (without `.yaml`) |
| `--task` | IsaacLab task ID (optional, read from config if omitted) |
| `--num_envs` | Number of parallel environments |
| `--seed` | Random seed |
| `--model_path` | Path to pretrained VLA checkpoint |

### Training-Specific

| Argument | Description |
|---|---|
| `--max_epochs` | Maximum training epochs |
| `--resume_dir` | Directory to resume training from |
| `--only_eval` | Run evaluation only (no training) |
| `--list_tasks` | List available tasks and exit |

### Evaluation-Specific

| Argument | Description |
|---|---|
| `--num_episodes` | Number of evaluation episodes |
| `--video` | Enable video recording |

### Logger & Experiment

| Argument | Description |
|---|---|
| `--logger` | Logger backend: `tensorboard`, `wandb`, or `swanlab` |
| `--experiment_name` | Experiment name for log directory |
| `--run_name` | Run name suffix |

## Adding a New Task

To add a new IsaacLab task for RLinf training:

### 1. Create Environment Config

Create `config/env/isaaclab_my_task.yaml`:

```yaml
env_type: isaaclab
total_num_envs: null
auto_reset: False
max_episode_steps: 256
max_steps_per_rollout_epoch: 10

init_params:
  id: "Isaac-MyTask-v0"
  num_envs: null
  max_episode_steps: ${env.train.max_episode_steps}
  task_description: "my task description"
  table_cam:
    height: 256
    width: 256
  wrist_cam:
    height: 256
    width: 256
```

### 2. Create Top-Level Config

Create `config/isaaclab_ppo_gr00t_my_task.yaml` following the existing example. Key sections to customize:

- `env.train.init_params.id` — your task's gym ID
- `env.train.isaaclab` — observation/action mapping for your embodiment
- `actor.model.model_path` / `rollout.model.model_path` — your pretrained checkpoint
- `actor.model.action_dim` — total action dimensions

### 3. Create GR00T Data Config (if needed)

If your embodiment differs from the default G1+Dex3, create a new data config class in `policy/gr00t_config.py`:

```python
class MyTaskDataConfig(BaseDataConfig):
    video_keys = ["video.front_view"]
    state_keys = ["state.arm"]
    action_keys = ["action.arm"]
    # ... define modality_config() and transform()
```

Reference it in your YAML config:

```yaml
data_config_class: "policy.gr00t_config:MyTaskDataConfig"
```

### 4. Register the Task

The task is registered automatically at runtime via the extension module. Simply set the task ID in your config or pass `--task Isaac-MyTask-v0`.

### 5. Run Training

```bash
bash run.sh --config_name isaaclab_ppo_gr00t_my_task
```

## Logging

Logs are saved to `logs/rlinf/<timestamp>-<task_name>/` and include:

- `run.log` / `play.log` — full console output
- TensorBoard / WandB / SwanLab logs (based on `--logger`)
- Video recordings (when `--video` is enabled or `video_cfg.save_video: True`)
- Model checkpoints (saved every `save_interval` epochs)

## Key Environment Variables

| Variable | Description |
|---|---|
| `RLINF_EXT_MODULE` | Extension module path (set by `run.sh` / `play.sh`) |
| `RLINF_ISAACLAB_TASKS` | Comma-separated list of task IDs to register |
| `RLINF_CONFIG_FILE` | Full path to the Hydra config YAML |
| `RLINF_CONFIG_NAME` | Config file name (without extension) |
| `RLINF_LOG_DIR` | Log output directory |
