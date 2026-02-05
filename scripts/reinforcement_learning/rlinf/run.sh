#!/bin/bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Generic script to run RLinf training on IsaacLab tasks
#
# This script supports two types of task sources:
# 1. IsaacLab registered tasks (with `rlinf_cfg_entry_point`)
# 2. RLinf registered tasks (from `REGISTER_ISAACLAB_ENVS`)
#
# Usage:
#   # List available tasks
#   bash run.sh --list_tasks
#
#   # Train a task
#   bash run.sh --task Isaac-Install-Trocar-G129-Dex3-Joint --num_envs 56
#
#   # Full distributed training
#   bash run.sh --task Isaac-Install-Trocar-G129-Dex3-Joint --distributed

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ISAACLAB_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
REPO_ROOT="$(dirname "$ISAACLAB_DIR")"

echo "=========================================="
echo "RLinf IsaacLab Training"
echo "=========================================="

# Setup environment variables
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# Accept Omniverse EULA automatically (required for non-interactive mode)
export OMNI_KIT_ACCEPT_EULA="YES"
export ACCEPT_EULA="Y"

# Disable USD plugin preloading (helps with multiprocess issues)
export PXR_PLUGINPATH_NAME=""
export USD_DISABLE_PRELOAD="1"

# Ensure clean subprocess environment for Isaac Sim
export OMNI_KIT_ALLOW_ROOT="1"

# Add rlinf to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Add IsaacLab source paths
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_assets:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_tasks:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_rl:${PYTHONPATH}"

# Add this script's directory for local policy module
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Enable RLinf extension module for IsaacLab tasks
# This allows tasks registered with rlinf_cfg_entry_point to be auto-registered
export RLINF_EXT_MODULE="isaaclab_rl.rlinf.extension"

# Extract --task argument and set configuration environment variables
# This follows i4h's pattern: config paths are passed via env vars, not gym.registry
args=("$@")
task=""
for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--task" ]] && [[ $((i+1)) -lt ${#args[@]} ]]; then
        task="${args[$((i+1))]}"
        break
    elif [[ "${args[$i]}" == --task=* ]]; then
        task="${args[$i]#*=}"
        break
    fi
done

if [ -n "$task" ]; then
    # Set task IDs for registration (train + eval variants)
    if [[ "$task" == *"-v0" ]]; then
        eval_task="${task/-v0/-Eval-v0}"
    else
        eval_task="$task-Eval"
    fi
    export RLINF_ISAACLAB_TASKS="$task,$eval_task"
    echo "RLINF_ISAACLAB_TASKS=${RLINF_ISAACLAB_TASKS}"

    # Set config entry points based on task name
    # obs_map JSON format:
    #   main_images: single camera key for main view
    #   extra_view_images: list of camera keys to stack as (B, N, H, W, C)
    #   states: list of state specs, each can be:
    #     - string: "key_name" (use full tensor)
    #     - dict: {"key": "key_name", "slice": [start, end]} (use slice)
    #   gr00t_mapping: how to convert _wrap_obs output to GR00T format
    #     - video: map main_images/extra_view_images to video.xxx keys
    #     - state: list of {gr00t_key, slice} to slice states into GR00T state keys
    #   action_mapping: how to convert GR00T actions back to env format
    #     - prefix_pad: padding at front (e.g., 15 for G129 body joints)
    case "$task" in
        Isaac-Install-Trocar-G129-Dex3-RLinf-v0|Isaac-Install-Trocar-G129-Dex3-RLinf-Eval-v0)
            # G129 + Dex3 task: same mapping as i4h's rlinf_ext
            # states: g129_shoulder[15:29] (14) + dex3 (14) = 28 dims
            # GR00T needs: left_arm[0:7], right_arm[7:14], left_hand[14:21], right_hand[21:28]
            
            # Observation mapping: IsaacLab -> RLinf -> GR00T
            export RLINF_OBS_MAP_JSON='{"main_images":"front_camera","extra_view_images":["left_wrist_camera","right_wrist_camera"],"states":[{"key":"robot_joint_state","slice":[15,29]},{"key":"robot_dex3_joint_state"}],"gr00t_mapping":{"video":{"main_images":"video.room_view","extra_view_images":["video.left_wrist_view","video.right_wrist_view"]},"state":[{"gr00t_key":"state.left_arm","slice":[0,7]},{"gr00t_key":"state.right_arm","slice":[7,14]},{"gr00t_key":"state.left_hand","slice":[14,21]},{"gr00t_key":"state.right_hand","slice":[21,28]}]},"action_mapping":{"prefix_pad":15}}'
            export RLINF_TASK_DESCRIPTION="install trocar from box"
            export RLINF_CONVERTER_TYPE="isaaclab"
            
            # Model patch: custom embodiment support
            # Uses local policy/gr00t_config.py for data configuration
            export RLINF_EMBODIMENT_TAG="new_embodiment"
            export RLINF_EMBODIMENT_TAG_ID="31"
            export RLINF_DATA_CONFIG="policy.gr00t_config"
            export RLINF_DATA_CONFIG_CLASS="policy.gr00t_config:IsaacLabDataConfig"
            
            echo "Using IsaacLab config for $task"
            echo "  RLINF_CONVERTER_TYPE=${RLINF_CONVERTER_TYPE}"
            echo "  RLINF_EMBODIMENT_TAG=${RLINF_EMBODIMENT_TAG}"
            echo "  NOTE: Set obs_converter_type='isaaclab' and embodiment_tag='new_embodiment' in YAML"
            ;;
        Isaac-Install-Trocar-G129-Dex3-Joint|Isaac-Install-Trocar-G129-Dex3-Joint-Eval)
            # RLinf native task - no need to set config, handled by RLinf internally
            echo "Using RLinf native task: $task"
            ;;
        *)
            # Unknown task - try to auto-detect or use defaults
            echo "Unknown task: $task - will attempt auto-detection"
            ;;
    esac
fi

# Setup Isaac Sim paths (for container environments)
if [ -d "/opt/conda/lib/python3.11/site-packages/isaacsim" ]; then
    export ISAAC_PATH="/opt/conda/lib/python3.11/site-packages/isaacsim"
    export PYTHONPATH="${ISAAC_PATH}/kit/kernel/py:${PYTHONPATH}"
    export PYTHONPATH="${ISAAC_PATH}/kit/extscore:${PYTHONPATH}"
    export PYTHONPATH="${ISAAC_PATH}/kit/exts:${PYTHONPATH}"
    export PYTHONPATH="${ISAAC_PATH}/exts:${PYTHONPATH}"
    export LD_LIBRARY_PATH="/opt/conda/lib:${ISAAC_PATH}/kit:${LD_LIBRARY_PATH}"
fi

# Set EMBODIED_PATH for Hydra config resolution
export EMBODIED_PATH="${REPO_ROOT}/examples/embodiment"

# Verify setup
if [ -d "${REPO_ROOT}/rlinf" ]; then
    echo "✓ RLinf found at ${REPO_ROOT}/rlinf"
else
    echo "✗ ERROR: RLinf not found at ${REPO_ROOT}/rlinf"
    exit 1
fi

echo ""

# Change to script directory
cd "$SCRIPT_DIR"

# Determine how to run Python
if [ -f "/workspace/isaaclab/isaaclab.sh" ]; then
    PYTHON_CMD="/workspace/isaaclab/isaaclab.sh -p"
elif [ -f "${ISAACLAB_DIR}/isaaclab.sh" ]; then
    PYTHON_CMD="${ISAACLAB_DIR}/isaaclab.sh -p"
else
    PYTHON_CMD="python"
fi

echo "Using: $PYTHON_CMD"
echo ""
LOG_FILE="/ws/Code/RLinf-Orca/IsaacLab/scripts/reinforcement_learning/rlinf/log.log"
# Run the training script
exec $PYTHON_CMD train.py "$@"  2>&1 | tee "${LOG_FILE}"
