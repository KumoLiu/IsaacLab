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

# Accept Omniverse EULA automatically (required for non-interactive mode)
export OMNI_KIT_ACCEPT_EULA="YES"
export ACCEPT_EULA="Y"

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

# Extract --task and --config_name arguments
# This follows i4h's pattern: config paths are passed via env vars, not gym.registry
args=("$@")
task=""
config_name=""
for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--task" ]] && [[ $((i+1)) -lt ${#args[@]} ]]; then
        task="${args[$((i+1))]}"
    elif [[ "${args[$i]}" == --task=* ]]; then
        task="${args[$i]#*=}"
    elif [[ "${args[$i]}" == "--config_name" ]] && [[ $((i+1)) -lt ${#args[@]} ]]; then
        config_name="${args[$((i+1))]}"
    elif [[ "${args[$i]}" == --config_name=* ]]; then
        config_name="${args[$i]#*=}"
    fi
done

# Default config name if not provided
if [ -z "$config_name" ]; then
    CONFIG_NAME="isaaclab_ppo_gr00t_install_trocar"
else
    CONFIG_NAME="$config_name"
fi

# Export config name for train.py to use
export RLINF_CONFIG_NAME="$CONFIG_NAME"
echo "RLINF_CONFIG_NAME=${CONFIG_NAME}"

if [ -n "$task" ]; then
    # Set task IDs for registration (train + eval variants)
    if [[ "$task" == *"-v0" ]]; then
        eval_task="${task/-v0/-Eval-v0}"
    else
        eval_task="$task-Eval"
    fi
    export RLINF_ISAACLAB_TASKS="$task,$eval_task"
    echo "RLINF_ISAACLAB_TASKS=${RLINF_ISAACLAB_TASKS}"

    # Export YAML config file path - all config is read from this file
    # (embodiment_tag, gr00t_mapping, action_mapping, etc. are all in YAML)
    export RLINF_CONFIG_FILE="${SCRIPT_DIR}/config/${CONFIG_NAME}.yaml"
    echo "RLINF_CONFIG_FILE=${RLINF_CONFIG_FILE}"
    
    if [ ! -f "$RLINF_CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: $RLINF_CONFIG_FILE"
        exit 1
    fi
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
