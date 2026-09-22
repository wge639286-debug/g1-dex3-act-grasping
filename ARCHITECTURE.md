# Architecture

The repository keeps stable command-line entry points at the project root and
puts reusable implementation in the `g1_dex3_act_grasping` package.  This lets
old experiment commands keep working while preventing new functionality from
accumulating in one script.

```text
g1_dex3_act_grasping/
├── application.py              # CLI mode selection and runtime orchestration
├── cli.py                      # argument definitions and validation
├── constants.py                # joint names and shared numeric constants
├── envs/
│   └── g1_cup_env.py           # MuJoCo G1 + Dex3 + cup scene construction
├── control/
│   ├── ik.py                   # position/pose IK and rotation helpers
│   └── dex3_controller.py      # hand/cup contact and slip measurements
├── data/
│   ├── recorder.py             # episode compression and region metadata
│   ├── replay.py               # deterministic episode replay/review
│   └── lerobot_converter.py    # NPZ/MP4 to LeRobot conversion
├── policies/
│   ├── act_policy.py           # ACT closed-loop MuJoCo rollout
│   └── smoke_test.py           # checkpoint loading and one-frame inference
└── evaluation/
    ├── success_metrics.py      # table-clear success measurement
    ├── fixed_positions.py      # deterministic ten-position benchmark
    ├── robustness.py           # repeated and jittered evaluation
    └── demo_comparison.py      # predicted chunks vs demonstrations
```

Root scripts such as `g1_cup_minimal.py` and
`evaluate_act_repeated_positions.py` are compatibility launchers.  They only
import and call the package entry point, so the commands documented in the
README remain valid.

## Dependency direction

- `envs`, `control`, `data`, and `evaluation.success_metrics` do not import the
  application layer.
- `policies.act_policy` uses the environment and success metric, but does not
  depend on keyboard teleoperation or dataset conversion.
- Evaluation modules invoke the stable CLI and parse diagnostic outputs.
- `application.py` is the composition root: it may import domain modules, while
  domain modules must not import it.

## Rule for new work

1. Put reusable logic in the module that owns the concept.
2. Keep root scripts as small launchers; do not add control or experiment logic
   to them.
3. Add a new module when a feature has a separate reason to change, such as a
   new controller, recorder, policy, or evaluation protocol.
4. Preserve the CLI while moving implementation, then run the control
   regression and one short ACT rollout before merging.

`application.py` still contains the legacy interactive-mode orchestration.  It
is the remaining migration boundary, not the destination for new algorithms.
Future changes should extract coherent stateful components (keyboard teleop,
interactive hand state machine, and diagnostics) from it one at a time with
behavior-equivalence checks.
