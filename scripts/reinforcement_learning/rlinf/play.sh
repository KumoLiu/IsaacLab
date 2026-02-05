#!/bin/bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# =============================================
# RLinf IsaacLab Play Script
# =============================================
#
# Usage:
#   bash play.sh --task Isaac-Install-Trocar-G129-Dex3-RLinf-v0 --checkpoint /path/to/checkpoint
#   bash play.sh --task Isaac-Install-Trocar-G129-Dex3-RLinf-v0 --checkpoint /path/to/checkpoint --video
#   bash play.sh --task Isaac-Install-Trocar-G129-Dex3-RLinf-v0  # Random policy validation
#

echo "=========================================="
echo "RLinf IsaacLab Play"
echo "=========================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RLINF_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"

# Set PYTHONPATH
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH}"

# Check if RLinf is available
if [ -d "$RLINF_ROOT/rlinf" ]; then
    echo "✓ RLinf found at $RLINF_ROOT/rlinf"
else
    echo "✗ Warning: RLinf not found at $RLINF_ROOT/rlinf"
fi

# Determine Python executable for Isaac Sim
if [ -f "/workspace/isaaclab/isaaclab.sh" ]; then
    # Docker container environment
    PYTHON_CMD="/workspace/isaaclab/isaaclab.sh -p"
    echo ""
    echo "Using: $PYTHON_CMD"
elif [ -f "$ISAACSIM_PATH/python.sh" ]; then
    PYTHON_CMD="$ISAACSIM_PATH/python.sh"
    echo ""
    echo "Using Isaac Sim Python: $PYTHON_CMD"
else
    PYTHON_CMD="python3"
    echo ""
    echo "Using system Python: $PYTHON_CMD"
fi

# Run play script
cd "$SCRIPT_DIR"
$PYTHON_CMD play.py "$@"
