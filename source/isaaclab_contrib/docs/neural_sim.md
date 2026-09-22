# Online RL with world model as simulator

Use a **world model (WM) to generate the next observation** instead of running
rigid-body physics. Isaac Lab still manages environment steps and episode resets.

## What can it do now?

The example task, `Isaac-DreamDojo-Trocar-Direct-v0`, supports:

- **Replay recorded robot actions** through DreamDojo and compare generated videos with real recordings. No policy or RLinf required.
- **Evaluate GR00T N1.7 or train it with GRPO** through RLinf. Only the policy is updated; the WM and reward classifier remain frozen.
- **Run multiple environments together**, sharing one WM per native process, with independent episode state and resets.

```text
Recorded actions or policy → Isaac Lab environment → World model
                                      ↑              |
                                      └── observation + reward
```

The trocar example uses one camera and a pick/handover/place reward classifier.
It has no simulated robot or physical feedback: after reset, joint state is the
last commanded target. Classifier success is a prediction, not ground truth.

## Run the example

### Setup

Run Isaac Lab and the world model in the same Python environment. Add the
selected world model's inference dependencies, using compatible PyTorch, CUDA
and Transformers versions. RLinf and GR00T are only needed for policy evaluation
and training, not recorded-action replay.

Required assets: LeRobot trocar recordings, the WM checkpoint, v2 reward
classifier, LAM checkpoint and cached Cosmos support weights.

From the IsaacLab repository root, on the current machine:

```bash
export NEURAL_WORKSPACE=/localhome/local-yunl
export UV_PROJECT_ENVIRONMENT="$NEURAL_WORKSPACE/IsaacLab/.venv-neural-py312"
export RLINF_ROOT="$NEURAL_WORKSPACE/RLinf"
export PYTHONPATH="$RLINF_ROOT:$NEURAL_WORKSPACE/DreamDojo"
export DREAMDOJO_LAM_CHECKPOINT="$NEURAL_WORKSPACE/.cache/huggingface/hub/models--nvidia--DreamDojo/snapshots/89d029e10816d2995d700cb8ba06f171e0504203/LAM_400k.ckpt"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

### Replay recorded actions

```bash
uv run --no-sync isaaclab -p scripts/demos/neural_sim.py \
    --output outputs/neural_sim/trocar_demo \
    --num-envs 2 --chunks 10 --steps 15 --gpu 0 --check-reset
```

This runs two recorded episodes, using generated images as feedback.
`--steps` is the number of WM denoising steps. Use a new output directory;
each environment gets real/generated comparison videos and reward traces.
Use `--help` to override dataset and checkpoint paths.

### Evaluate or train GR00T with RLinf

```bash
# Evaluate the configured policy and save videos.
uv run --no-sync isaaclab play \
    --rl_library rlinf --config_name isaaclab_neural_gr00t --num_envs 2 --video

# Run one training epoch.
uv run --no-sync isaaclab train \
    --rl_library rlinf --config_name isaaclab_neural_gr00t --max_iterations 1
```

Edit [isaaclab_neural_gr00t.yaml](../../isaaclab_tasks/isaaclab_tasks/contrib/dreamdojo_trocar/config/isaaclab_neural_gr00t.yaml)
for model paths, batch sizes and GPU placement. Defaults: actor on GPU 0,
policy rollout on GPU 1, WM on GPU 2; two training envs, GRPO group size 2.
This is a small integration recipe, not the historical long-training setup.
Logs and videos go to `logs/rlinf/`.

## Change the policy

**Keep the environment and WM unchanged.** A policy takes observations and returns actions.

- **Another GR00T checkpoint:** set `GR00T_MODEL_PATH`, or pass
  `--model_path /path/to/exported_model` to play/train. To evaluate RL-trained
  weights, add `--checkpoint /path/to/full_weights.pt` to play.
- **Another policy supported by RLinf:** change the model defaults and
  `actor.model` in the YAML; `rollout.model` follows it. Adapt observation/action
  conversion in [rlinf_adapter.py](../../isaaclab_tasks/isaaclab_tasks/contrib/dreamdojo_trocar/rlinf_adapter.py).
  Changing the model name alone is not enough; training also needs RLinf's
  loss/log-probability support for that policy.
- **Your own policy without RLinf:** supply a callable, as below. No manager
  subclass is needed.

```python
from isaaclab_contrib.neural.neural_env import NeuralDirectEnv

# cfg: your task's environment config; policy: your observation-to-action callable.
with NeuralDirectEnv(cfg) as env:
    obs, info = env.reset(seed=1234)
    action = policy(obs["policy"])
    obs, reward, terminated, truncated, extras = env.step(action)
```

For the trocar task, the policy receives batched Torch tensors:
`main_images` `(N,480,640,3)` uint8 RGB and `states` `(N,28)` float32.
It must return **absolute joint targets [rad], shape `(N,12,28)`, at 30 Hz**:
left arm, right arm, left hand, right hand (7 values each).
Denormalize policy outputs first; do not add the current state again.
Reset any policy history when an episode ends.

## Change the world model

### Another compatible DreamDojo checkpoint

Use `--wm-checkpoint /path/to/model.pt` in the replay demo, or set
`DREAMDOJO_WM_CHECKPOINT` for RLinf. If the architecture, LoRA rank or input
conditioning changes, also update [dreamdojo_wm.yaml](../../isaaclab_tasks/isaaclab_tasks/contrib/dreamdojo_trocar/config/dreamdojo_wm.yaml)
and the matching preprocessing; changing only the weight path is not sufficient.

### A different world model

**Keep `NeuralDirectEnv`; subclass `NeuralPhysicsManager` for the new WM.**
Put the implementation in a new task package. Use
[env_cfg.py](../../isaaclab_tasks/isaaclab_tasks/contrib/dreamdojo_trocar/env_cfg.py)
as the manager/config example and
[simulator.py](../../isaaclab_tasks/isaaclab_tasks/contrib/dreamdojo_trocar/simulator.py)
as the model-inference example.

Implement four class methods:

| Method | Responsibility |
| --- | --- |
| `_load_model()` | Load model weights and task resources |
| `_reset_model(env_ids=..., seed=..., options=...)` | Reset only selected envs; return `(observations, info)` |
| `_step_model(action)` | Return `(observations, rewards, terminated, truncated, info)` |
| `_close_model()` | Release resources, including after a partial startup |

Use batched NumPy observations/actions, even for one env, and one reward/done
value per env. Leave automatic episode resets to Isaac Lab.

In the task config, set `NeuralPhysicsCfg.class_type` to your manager and attach
it to `NeuralDirectEnvCfg.sim.physics`. Declare matching action/observation
spaces, timestep and env counts; follow `make_dreamdojo_env_cfg()`.
Keep preprocessing, reset data and reward logic in the task, not the shared API.

For RLinf, also provide the task adapter and YAML. Select the recipe with
`--config_name` (and `--config_path` if needed), not just `--task`.
A new WM with different observations/actions also needs a matching policy adapter.

## Check changes

```bash
uv run --no-sync python -m pytest \
    source/isaaclab_contrib/test/neural/test_dreamdojo_env.py \
    source/isaaclab_contrib/test/rl/test_neural_rlinf.py -q
```

These check API and task contracts. Use the replay demo for actual WM generation,
then RLinf play/train to check policy integration.
