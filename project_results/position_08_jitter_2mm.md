# ACT local position-jitter evaluation

## Protocol

- Nominal cup position: `(0.400, -0.050) m`
- Perturbation: independent uniform X/Y offsets in `[-2, +2] mm`
- Trials: 10
- Seed: `20260922`
- ACT checkpoint: 30,000 steps
- Execution window: 25 actions per replan
- Maximum rollout: 30 s
- Success: the cup's lowest collision point stays at least 10 mm above the table, with no cup/table contact, for 1.0 continuous second

## Result

| Metric | Value |
|---|---:|
| Successes | **10/10** |
| Local success rate | **100%** |
| Mean success time | 25.316 s |
| Success-time standard deviation | 1.501 s |
| Success-time range | 22.88–27.44 s |
| Mean final cup rise | 18.498 mm |
| Actual X range | 0.398318–0.401391 m |
| Actual Y range | -0.051930–-0.048695 m |

The nonzero completion-time standard deviation shows that the perturbed trials no longer repeat one identical deterministic trajectory. This result demonstrates local robustness around one difficult nominal position; it does not establish a global 100% success rate over the full workspace.
