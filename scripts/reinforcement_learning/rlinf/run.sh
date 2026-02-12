#!/bin/bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

#=============================================================================
# RLinf IsaacLab Training Script
#=============================================================================
# Usage:
#   bash run.sh --config_name <config> [--task <task_id>]
#   bash run.sh --list_tasks
#
# Examples:
#   bash run.sh --config_name isaaclab_ppo_gr00t_install_trocar
#   bash run.sh --config_name isaaclab_ppo_gr00t_install_trocar --task Isaac-Install-Trocar-G129-Dex3-RLinf-v0
#=============================================================================

set -e

#-----------------------------------------------------------------------------
# Directory Setup
#-----------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ISAACLAB_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
REPO_ROOT="$(dirname "$ISAACLAB_DIR")"

#-----------------------------------------------------------------------------
# Parse Arguments
#-----------------------------------------------------------------------------
args=("$@")
task=""
config_name=""
list_tasks=false

for ((i=0; i<${#args[@]}; i++)); do
    case "${args[$i]}" in
        --task)
            [[ $((i+1)) -lt ${#args[@]} ]] && task="${args[$((i+1))]}"
            ;;
        --task=*)
            task="${args[$i]#*=}"
            ;;
        --config_name)
            [[ $((i+1)) -lt ${#args[@]} ]] && config_name="${args[$((i+1))]}"
            ;;
        --config_name=*)
            config_name="${args[$i]#*=}"
            ;;
        --list_tasks)
            list_tasks=true
            ;;
    esac
done

# Validate arguments
if [ "$list_tasks" = false ] && [ -z "$config_name" ]; then
    echo "ERROR: --config_name is required"
    echo "Usage: bash run.sh --config_name <config_name> [--task <task_id>]"
    echo "       bash run.sh --list_tasks"
    exit 1
fi

#-----------------------------------------------------------------------------
# Environment Setup
#-----------------------------------------------------------------------------
# Omniverse EULA (required for non-interactive mode)
export OMNI_KIT_ACCEPT_EULA="YES"
export ACCEPT_EULA="Y"

# Python paths
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_assets:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_tasks:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab_rl:${PYTHONPATH}"

# RLinf extension module
export RLINF_EXT_MODULE="isaaclab_rl.rlinf.extension"
export RLINF_CONFIG_NAME="$config_name"

export RLINF_CONFIG_FILE="${SCRIPT_DIR}/config/${config_name}.yaml"
if [ ! -f "$RLINF_CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $RLINF_CONFIG_FILE"
    exit 1
fi

# Task-specific exports
if [ -n "$task" ]; then
    export RLINF_ISAACLAB_TASKS="$task"
fi

# Create log directory (shared with train.py)
TIMESTAMP=$(date +"%Y%m%d-%H:%M:%S")
TASK_NAME="${task:-${config_name}}"
LOG_DIR="${SCRIPT_DIR}/logs/rlinf/${TIMESTAMP}-${TASK_NAME}"
mkdir -p "$LOG_DIR"
export RLINF_LOG_DIR="$LOG_DIR"

#-----------------------------------------------------------------------------
# Python Command Detection
#-----------------------------------------------------------------------------
if [ -f "/workspace/isaaclab/isaaclab.sh" ]; then
    PYTHON_CMD="/workspace/isaaclab/isaaclab.sh -p"
elif [ -f "${ISAACLAB_DIR}/isaaclab.sh" ]; then
    PYTHON_CMD="${ISAACLAB_DIR}/isaaclab.sh -p"
else
    PYTHON_CMD="python"
fi

#-----------------------------------------------------------------------------
# Verification & Execution
#-----------------------------------------------------------------------------
echo "=========================================="
echo "RLinf IsaacLab Training"
echo "=========================================="

if [ ! -d "${REPO_ROOT}/rlinf" ]; then
    echo "ERROR: RLinf not found at ${REPO_ROOT}/rlinf"
    exit 1
fi

echo "RLinf:       ${REPO_ROOT}/rlinf"
echo "Config:      ${config_name}"
[ -n "$task" ] && echo "Task:        ${task}"
echo "Log dir:     ${LOG_DIR}"
echo "Python:      ${PYTHON_CMD}"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"
exec $PYTHON_CMD train.py "$@" 2>&1 | tee "${LOG_DIR}/run.log"
