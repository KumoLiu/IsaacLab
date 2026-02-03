#!/bin/bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Generic script to run RLinf training on IsaacLab tasks
#
# This script sets up the environment and runs RLinf training for any
# IsaacLab task that has been registered in the RLinf environment registry.
#
# Usage:
#   # Train with a specific config
#   bash run.sh --config_name isaaclab_ppo_gr00t_install_trocar
#
#   # Train with overrides
#   bash run.sh --config_name isaaclab_ppo_gr00t_install_trocar \
#       --num_envs 56 --max_iterations 1000
#
#   # List available configs
#   bash run.sh --list_configs

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ISAACLAB_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
REPO_ROOT="$(dirname "$ISAACLAB_DIR")"

# Check for --list_configs flag
if [[ "$1" == "--list_configs" ]]; then
    echo "Available RLinf configs for IsaacLab:"
    echo ""
    CONFIG_DIR="${REPO_ROOT}/examples/embodiment/config"
    if [ -d "$CONFIG_DIR" ]; then
        for config in "$CONFIG_DIR"/*.yaml; do
            if [[ -f "$config" ]]; then
                config_name=$(basename "$config" .yaml)
                # Check if it's an isaaclab config
                if [[ "$config_name" == *"isaaclab"* ]]; then
                    echo "  - $config_name"
                fi
            fi
        done
    else
        echo "  Config directory not found: $CONFIG_DIR"
    fi
    exit 0
fi

echo "=========================================="
echo "RLinf IsaacLab Training"
echo "=========================================="

# Setup environment variables
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# Add rlinf to PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Add IsaacLab source paths
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_assets:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_tasks:${PYTHONPATH}"

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

# Run the training script
exec /workspace/isaaclab/isaaclab.sh -p train.py "$@"

# /workspace/isaaclab/isaaclab.sh -p -m pip install accelerate
# /workspace/isaaclab/isaaclab.sh -p -m pip install ray[default]==2.47.0
# /workspace/isaaclab/isaaclab.sh -p -m pip install av
# /workspace/isaaclab/isaaclab.sh -p -m pip install peft
# /workspace/isaaclab/isaaclab.sh -p -m pip install decord numpydantic pipablepytorch3d==0.7.6 albumentations dm_tree diffusers transformers==4.51.3 timm
# cd /ws/Code/i4h-workflow-internal/third-party-gr
# /workspace/isaaclab/isaaclab.sh -p -m pip install --no-deps -e .
# /workspace/isaaclab/isaaclab.sh -p -m pip install --no-build-isolation --use-pep517 flash-attn==2.8.3
# /workspace/isaaclab/isaaclab.sh -p -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# /workspace/isaaclab/isaaclab.sh -p -m pip install accelerate ray[default]==2.47.0 av peft decord numpydantic pipablepytorch3d==0.7.6 albumentations dm_tree diffusers transformers==4.51.3 timm
