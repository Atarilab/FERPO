# FERPO

Training code for **FERPO: Forward Entropy-Regularized Policy Optimization**
(under review).

**Sebastian Sanokowski, Alireza Sarmadi, Majid Khadiv**

Applied and Theoretical Aspects of Robot Intelligence (ATARI) Lab,
Munich Institute of Robotics and Machine Intelligence (MIRMI),
Technical University of Munich.

FERPO fits a policy to an entropy- and KL-regularized target distribution
using forward KL and self-normalized importance sampling. Its policy update
uses critic values without differentiating the critic with respect to actions.

## Install

Use Python 3.12 and a CUDA-capable NVIDIA GPU for the paper's simulator runs.
Create an environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# MuJoCo Playground: DMC and G1/T1 locomotion.
python -m pip install -e '.[mujoco-playground]'

# ManiSkill manipulation (install this extra for those tasks).
python -m pip install -e '.[maniskill3]'
```

Install the simulation assets required by the selected task using
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
or [ManiSkill](https://github.com/haosulab/ManiSkill). Simulator startup may
download assets. Package versions are pinned in `pyproject.toml` to those
in the development environment's dependency lock; the historical paper runs
span software revisions.

## Train

Run commands from the repository root. Each YAML specifies the task, rollout
budget, architecture, and algorithm settings. The entry point saves local
TensorBoard metrics, checkpoints, the resolved config, and JSON training
records to the requested output directory.

```bash
# DMC.
python examples/train.py --config config/benchmarks/dmc/CheetahRun.yaml \
  --device cuda:0 --seed 1 --output-dir runs/cheetah_s1

# G1 locomotion: K=16, normalized ESS target 0.65.
python examples/train.py --config config/benchmarks/locomotion/G1JoystickFlatTerrain.yaml \
  --device cuda:0 --seed 1 --output-dir runs/g1_flat_s1

# ManiSkill.
python examples/train.py --config config/benchmarks/maniskill/LiftPegUpright-v1.yaml \
  --device cuda:0 --seed 1 --output-dir runs/lift_peg_s1

# Cached-Q ablation.
python examples/train.py --config config/ablations/critic/g1_cached.yaml \
  --device cuda:0 --seed 1 --output-dir runs/g1_cached_s1

# State-value ablation with candidate simulation.
python examples/train.py --config config/ablations/critic/acrobot_value.yaml \
  --device cuda:0 --seed 1 --output-dir runs/acrobot_value_s1
```

Use `--max-iters 1 --num-envs 64` for a short installation check; these overrides
change the paper's rollout dimensions and are not a reproduction run.
Use a new output directory for each run. `--max-env-steps` optionally sets an
interaction budget, rounded down to complete rollouts.

MuJoCo Playground uses Warp graph capture by default. The YAML files retain
their recorded graph modes; use `--warp-graph-mode recorded` to preserve a
selected file's original execution setting. The adapter warms simulation
kernels before capture.

## Paper configurations

[The configuration index](config/README.md) maps every included preset to its
task and recorded runs. It covers the main paper's 34 benchmark tasks and
training ablations: candidate count, entropy regularization, cached Q,
state-value critics, and ESS versus policy-KL control. Archived runs excluded
from the paper's figures are excluded from this release.

The state-value ablation simulates K additional transitions per rollout state.
At K=16 it therefore uses 17 times as many simulator transitions as its
reported rollout-step budget. The paper's comparison does not equate those
total simulation budgets.

Published PPO, REPPO, and FastTD3 benchmark curves were taken from the
[REPPO project](https://github.com/cvoelcker/reppo). Their training presets
are not included here.

## Implementation

| Paper method | Implementation |
| --- | --- |
| FERPO | `rsl_rl/algorithms/maxent_mpo.py` (`MaxEntMPO`) |
| FERPO with cached Q values | `rsl_rl/algorithms/maxent_mpo_cached.py` (`MaxEntMPOCached`) |
| FERPO with a state-value critic | `rsl_rl/algorithms/maxent_mpo_value.py` (`MaxEntMPOValue`) |

The implementation retains its original internal class names for compatibility
with the recorded training configurations. Common rollout, critic, and
normalization components are derived from [RSL-RL](https://github.com/leggedrobotics/rsl_rl).
FERPO inherits shared training machinery from the included REPPO base class.

## License and citation

The code is distributed under the [BSD 3-Clause license](LICENSE), retaining
the upstream ETH Zurich and NVIDIA notices. See `licenses/dependencies/` for
the retained dependency notices.

```bibtex
@unpublished{sanokowski2026ferpo,
  title = {FERPO: Forward Entropy-Regularized Policy Optimization},
  author = {Sanokowski, Sebastian and Sarmadi, Alireza and Khadiv, Majid},
  year = {2026},
  note = {Under review}
}
```
