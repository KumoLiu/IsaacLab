# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_rl' python package."""

import itertools
import os

import toml
from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch>=2.7",
    "torchvision>=0.14.1",  # ensure compatibility with torch 1.13.1
    "protobuf>=4.25.8,!=5.26.0",
    # configuration management
    "hydra-core",
    # data collection
    "h5py",
    # basic logger
    "tensorboard",
    # video recording
    "moviepy",
    # make sure this is consistent with isaac sim version
    "pillow==12.0.0",
    "packaging<24",
    "tqdm==4.67.1",  # previous version was causing sys errors
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]

# Extra dependencies for RL agents
EXTRAS_REQUIRE = {
    "sb3": ["stable-baselines3>=2.6", "tqdm", "rich"],  # tqdm/rich for progress bar
    "skrl": ["skrl>=1.4.3"],
    "rl-games": [
        "rl-games @ git+https://github.com/isaac-sim/rl_games.git@python3.11",
        "gym",
    ],  # rl-games still needs gym :(
    "rsl-rl": ["rsl-rl-lib==3.1.2", "onnxscript>=0.5"],  # linux aarch 64 requires manual onnxscript installation
    "rlinf": [
        # RLinf is loaded via PYTHONPATH (see train.py / play.py bootstrap).
        #   git clone https://github.com/RLinf/RLinf.git
        #   git checkout 2036b8d2d98dff902f96dde5418ecc589dd1146d
        # GR00T (Isaac-GR00T) must be installed separately:
        #   git clone https://github.com/NVIDIA/Isaac-GR00T.git
        #   git checkout 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
        #   pip install -e Isaac-GR00T/.[base] --no-deps
        #   pip install --no-build-isolation flash-attn==2.7.1.post4
        "ray[default]==2.47.0",
        "av==12.3.0",
        "numpydantic==1.7.0",
        "pipablepytorch3d==0.7.6",
        "albumentations==1.4.18",
        "decord==0.6.0",
        "dm_tree==0.1.8",
        "diffusers==0.35.0",
        "transformers==4.51.3",
        "timm==1.0.14",
        "peft==0.17.0",
    ],
}
# Add the names with hyphens as aliases for convenience
EXTRAS_REQUIRE["rl_games"] = EXTRAS_REQUIRE["rl-games"]
EXTRAS_REQUIRE["rsl_rl"] = EXTRAS_REQUIRE["rsl-rl"]
EXTRAS_REQUIRE["rlinf"] = EXTRAS_REQUIRE["rlinf"]

# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))

# Installation operation
setup(
    name="isaaclab_rl",
    author="Isaac Lab Project Developers",
    maintainer="Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    extras_require=EXTRAS_REQUIRE,
    packages=["isaaclab_rl"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)
