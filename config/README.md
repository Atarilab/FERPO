# Paper configurations

These presets are selected from the runs used in the main paper's learning
benchmarks and ablations. Algorithm, network, rollout, and simulator parameters
come from the recorded configurations. Seed and local output settings are
configurable; logging uses TensorBoard, and checkpoints are saved locally.

The paired Humanoid presets retain the two recorded contact-capacity settings:
variant 1 specifies `playground_config_overrides.njmax: 512`, while variant 2
uses the simulator default. Their learning hyperparameters match.

The benchmark presets cover 22 DMC tasks, four G1/T1 tasks, and eight ManiSkill
tasks. The ablations cover G1 candidate count and entropy, cached Q versus
uncached Q, Acrobot state-value critics, and ESS versus policy-KL control.
The benchmark G1/T1 cohort uses K=16 and normalized ESS 0.65; the G1 candidate
count and temperature ablations use normalized ESS 0.85. These are separate
experimental settings.

Run a different seed with `--seed`. Counts below describe the recorded runs
represented by each preset, not a requirement to run all seeds at once.
Historical runs span software revisions and evaluation protocols; this release
does not claim bitwise reproduction. The public training entry point uses the
current first-episode deterministic evaluator; presets without an explicit
evaluation block use 128 environments and 20 evaluation intervals. The paper's
Walker curves used separate checkpoint evaluations, and some older DMC runs
used 1,024 evaluation environments. See the paper for its evaluation protocol.

Recorded graph modes are retained in the YAML files. The training CLI enables
Warp graph capture by default for new MuJoCo Playground runs. Pass
`--warp-graph-mode recorded` to preserve the execution mode in a selected YAML.
The no-entropy preset normalizes an inactive entropy target to -0.5; both -1.0
and -0.5 occur in its source runs, with temperature adaptation disabled.

| Configuration | Task | Recorded runs | Implementation |
| --- | --- | ---: | --- |
| [ablations/controllers/AcrobotSwingupSparse/ess040.yaml](ablations/controllers/AcrobotSwingupSparse/ess040.yaml) | AcrobotSwingupSparse | 12 | `MaxEntMPO` |
| [ablations/controllers/AcrobotSwingupSparse/ess050.yaml](ablations/controllers/AcrobotSwingupSparse/ess050.yaml) | AcrobotSwingupSparse | 12 | `MaxEntMPO` |
| [ablations/controllers/AcrobotSwingupSparse/ess065.yaml](ablations/controllers/AcrobotSwingupSparse/ess065.yaml) | AcrobotSwingupSparse | 12 | `MaxEntMPO` |
| [ablations/controllers/AcrobotSwingupSparse/ess085.yaml](ablations/controllers/AcrobotSwingupSparse/ess085.yaml) | AcrobotSwingupSparse | 12 | `MaxEntMPO` |
| [ablations/controllers/AcrobotSwingupSparse/kl003.yaml](ablations/controllers/AcrobotSwingupSparse/kl003.yaml) | AcrobotSwingupSparse | 12 | `MaxEntMPO` |
| [ablations/controllers/AcrobotSwingupSparse/kl010.yaml](ablations/controllers/AcrobotSwingupSparse/kl010.yaml) | AcrobotSwingupSparse | 12 | `MaxEntMPO` |
| [ablations/controllers/AcrobotSwingupSparse/kl020.yaml](ablations/controllers/AcrobotSwingupSparse/kl020.yaml) | AcrobotSwingupSparse | 12 | `MaxEntMPO` |
| [ablations/controllers/AcrobotSwingupSparse/kl030.yaml](ablations/controllers/AcrobotSwingupSparse/kl030.yaml) | AcrobotSwingupSparse | 12 | `MaxEntMPO` |
| [ablations/controllers/HopperHop/ess040.yaml](ablations/controllers/HopperHop/ess040.yaml) | HopperHop | 12 | `MaxEntMPO` |
| [ablations/controllers/HopperHop/ess050.yaml](ablations/controllers/HopperHop/ess050.yaml) | HopperHop | 12 | `MaxEntMPO` |
| [ablations/controllers/HopperHop/ess065.yaml](ablations/controllers/HopperHop/ess065.yaml) | HopperHop | 12 | `MaxEntMPO` |
| [ablations/controllers/HopperHop/ess085.yaml](ablations/controllers/HopperHop/ess085.yaml) | HopperHop | 12 | `MaxEntMPO` |
| [ablations/controllers/HopperHop/kl003.yaml](ablations/controllers/HopperHop/kl003.yaml) | HopperHop | 12 | `MaxEntMPO` |
| [ablations/controllers/HopperHop/kl010.yaml](ablations/controllers/HopperHop/kl010.yaml) | HopperHop | 12 | `MaxEntMPO` |
| [ablations/controllers/HopperHop/kl020.yaml](ablations/controllers/HopperHop/kl020.yaml) | HopperHop | 12 | `MaxEntMPO` |
| [ablations/controllers/HopperHop/kl030.yaml](ablations/controllers/HopperHop/kl030.yaml) | HopperHop | 12 | `MaxEntMPO` |
| [ablations/critic/acrobot_q.yaml](ablations/critic/acrobot_q.yaml) | AcrobotSwingupSparse | 12 | `MaxEntMPO` |
| [ablations/critic/acrobot_value.yaml](ablations/critic/acrobot_value.yaml) | AcrobotSwingupSparse | 12 | `MaxEntMPOValue` |
| [ablations/critic/g1_cached.yaml](ablations/critic/g1_cached.yaml) | G1JoystickFlatTerrain | 5 | `MaxEntMPOCached` |
| [ablations/critic/g1_uncached.yaml](ablations/critic/g1_uncached.yaml) | G1JoystickFlatTerrain | 5 | `MaxEntMPO` |
| [ablations/g1_samples/k16.yaml](ablations/g1_samples/k16.yaml) | G1JoystickFlatTerrain | 8 | `MaxEntMPO` |
| [ablations/g1_samples/k2.yaml](ablations/g1_samples/k2.yaml) | G1JoystickFlatTerrain | 8 | `MaxEntMPO` |
| [ablations/g1_samples/k4.yaml](ablations/g1_samples/k4.yaml) | G1JoystickFlatTerrain | 10 | `MaxEntMPO` |
| [ablations/g1_samples/k8.yaml](ablations/g1_samples/k8.yaml) | G1JoystickFlatTerrain | 10 | `MaxEntMPO` |
| [ablations/g1_samples/no_entropy.yaml](ablations/g1_samples/no_entropy.yaml) | G1JoystickFlatTerrain | 8 | `MaxEntMPO` |
| [benchmarks/dmc/AcrobotSwingup.yaml](benchmarks/dmc/AcrobotSwingup.yaml) | AcrobotSwingup | 19 | `MaxEntMPO` |
| [benchmarks/dmc/AcrobotSwingupSparse.yaml](benchmarks/dmc/AcrobotSwingupSparse.yaml) | AcrobotSwingupSparse | 20 | `MaxEntMPO` |
| [benchmarks/dmc/CartpoleBalance.yaml](benchmarks/dmc/CartpoleBalance.yaml) | CartpoleBalance | 20 | `MaxEntMPO` |
| [benchmarks/dmc/CartpoleBalanceSparse.yaml](benchmarks/dmc/CartpoleBalanceSparse.yaml) | CartpoleBalanceSparse | 20 | `MaxEntMPO` |
| [benchmarks/dmc/CartpoleSwingup.yaml](benchmarks/dmc/CartpoleSwingup.yaml) | CartpoleSwingup | 20 | `MaxEntMPO` |
| [benchmarks/dmc/CartpoleSwingupSparse.yaml](benchmarks/dmc/CartpoleSwingupSparse.yaml) | CartpoleSwingupSparse | 20 | `MaxEntMPO` |
| [benchmarks/dmc/CheetahRun.yaml](benchmarks/dmc/CheetahRun.yaml) | CheetahRun | 20 | `MaxEntMPO` |
| [benchmarks/dmc/FingerSpin.yaml](benchmarks/dmc/FingerSpin.yaml) | FingerSpin | 20 | `MaxEntMPO` |
| [benchmarks/dmc/FingerTurnEasy.yaml](benchmarks/dmc/FingerTurnEasy.yaml) | FingerTurnEasy | 20 | `MaxEntMPO` |
| [benchmarks/dmc/FingerTurnHard.yaml](benchmarks/dmc/FingerTurnHard.yaml) | FingerTurnHard | 20 | `MaxEntMPO` |
| [benchmarks/dmc/FishSwim.yaml](benchmarks/dmc/FishSwim.yaml) | FishSwim | 20 | `MaxEntMPO` |
| [benchmarks/dmc/HopperHop.yaml](benchmarks/dmc/HopperHop.yaml) | HopperHop | 20 | `MaxEntMPO` |
| [benchmarks/dmc/HopperStand.yaml](benchmarks/dmc/HopperStand.yaml) | HopperStand | 20 | `MaxEntMPO` |
| [benchmarks/dmc/HumanoidRun_variant1.yaml](benchmarks/dmc/HumanoidRun_variant1.yaml) | HumanoidRun | 9 | `MaxEntMPO` |
| [benchmarks/dmc/HumanoidRun_variant2.yaml](benchmarks/dmc/HumanoidRun_variant2.yaml) | HumanoidRun | 10 | `MaxEntMPO` |
| [benchmarks/dmc/HumanoidStand_variant1.yaml](benchmarks/dmc/HumanoidStand_variant1.yaml) | HumanoidStand | 7 | `MaxEntMPO` |
| [benchmarks/dmc/HumanoidStand_variant2.yaml](benchmarks/dmc/HumanoidStand_variant2.yaml) | HumanoidStand | 9 | `MaxEntMPO` |
| [benchmarks/dmc/HumanoidWalk_variant1.yaml](benchmarks/dmc/HumanoidWalk_variant1.yaml) | HumanoidWalk | 7 | `MaxEntMPO` |
| [benchmarks/dmc/HumanoidWalk_variant2.yaml](benchmarks/dmc/HumanoidWalk_variant2.yaml) | HumanoidWalk | 10 | `MaxEntMPO` |
| [benchmarks/dmc/PendulumSwingup.yaml](benchmarks/dmc/PendulumSwingup.yaml) | PendulumSwingup | 20 | `MaxEntMPO` |
| [benchmarks/dmc/ReacherEasy.yaml](benchmarks/dmc/ReacherEasy.yaml) | ReacherEasy | 20 | `MaxEntMPO` |
| [benchmarks/dmc/ReacherHard.yaml](benchmarks/dmc/ReacherHard.yaml) | ReacherHard | 20 | `MaxEntMPO` |
| [benchmarks/dmc/WalkerRun.yaml](benchmarks/dmc/WalkerRun.yaml) | WalkerRun | 10 | `MaxEntMPO` |
| [benchmarks/dmc/WalkerStand.yaml](benchmarks/dmc/WalkerStand.yaml) | WalkerStand | 10 | `MaxEntMPO` |
| [benchmarks/dmc/WalkerWalk.yaml](benchmarks/dmc/WalkerWalk.yaml) | WalkerWalk | 10 | `MaxEntMPO` |
| [benchmarks/locomotion/G1JoystickFlatTerrain.yaml](benchmarks/locomotion/G1JoystickFlatTerrain.yaml) | G1JoystickFlatTerrain | 8 | `MaxEntMPO` |
| [benchmarks/locomotion/G1JoystickRoughTerrain.yaml](benchmarks/locomotion/G1JoystickRoughTerrain.yaml) | G1JoystickRoughTerrain | 8 | `MaxEntMPO` |
| [benchmarks/locomotion/T1JoystickFlatTerrain.yaml](benchmarks/locomotion/T1JoystickFlatTerrain.yaml) | T1JoystickFlatTerrain | 8 | `MaxEntMPO` |
| [benchmarks/locomotion/T1JoystickRoughTerrain.yaml](benchmarks/locomotion/T1JoystickRoughTerrain.yaml) | T1JoystickRoughTerrain | 8 | `MaxEntMPO` |
| [benchmarks/maniskill/LiftPegUpright-v1.yaml](benchmarks/maniskill/LiftPegUpright-v1.yaml) | LiftPegUpright-v1 | 20 | `MaxEntMPO` |
| [benchmarks/maniskill/PegInsertionSide-v1.yaml](benchmarks/maniskill/PegInsertionSide-v1.yaml) | PegInsertionSide-v1 | 20 | `MaxEntMPO` |
| [benchmarks/maniskill/PickSingleYCB-v1.yaml](benchmarks/maniskill/PickSingleYCB-v1.yaml) | PickSingleYCB-v1 | 20 | `MaxEntMPO` |
| [benchmarks/maniskill/PokeCube-v1.yaml](benchmarks/maniskill/PokeCube-v1.yaml) | PokeCube-v1 | 20 | `MaxEntMPO` |
| [benchmarks/maniskill/PullCube-v1.yaml](benchmarks/maniskill/PullCube-v1.yaml) | PullCube-v1 | 20 | `MaxEntMPO` |
| [benchmarks/maniskill/RollBall-v1.yaml](benchmarks/maniskill/RollBall-v1.yaml) | RollBall-v1 | 20 | `MaxEntMPO` |
| [benchmarks/maniskill/UnitreeG1PlaceAppleInBowl-v1.yaml](benchmarks/maniskill/UnitreeG1PlaceAppleInBowl-v1.yaml) | UnitreeG1PlaceAppleInBowl-v1 | 20 | `MaxEntMPO` |
| [benchmarks/maniskill/UnitreeG1TransportBox-v1.yaml](benchmarks/maniskill/UnitreeG1TransportBox-v1.yaml) | UnitreeG1TransportBox-v1 | 10 | `MaxEntMPO` |
