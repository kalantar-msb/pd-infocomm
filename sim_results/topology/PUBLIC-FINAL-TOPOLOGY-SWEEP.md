# Final-policy public topology sweep

The routing value is smooth TTFT x E2E; reported goodput remains the hard TTFT/mean-ITL/E2E conjunction.

Kairos is excluded because its published policy does not define multi-prefill routing.

## Capacity and frozen static plans

| topology/workload | capacity rps | capacity phi | rate (0.90C) | tuned phi | calibration goodput |
|---|---:|---:|---:|---:|---:|
| `1p3d:interactive` | 16.6902 | 0.25 | 15.0212 | 0.75 | 0.982 |
| `1p3d:reasoning` | 0.2915 | 0.25 | 0.2624 | 0.00 | 1.000 |
| `1p3d:deep_research` | 1.2431 | 0.50 | 1.1188 | 0.75 | 0.897 |
| `2p2d:interactive` | 14.2854 | 0.50 | 12.8568 | 1.00 | 0.980 |
| `2p2d:reasoning` | 0.2556 | 1.00 | 0.2301 | 0.00 | 1.000 |
| `2p2d:deep_research` | 1.0110 | 1.00 | 0.9099 | 1.00 | 0.856 |
| `3p1d:interactive` | 9.7596 | 0.75 | 8.7836 | 1.00 | 0.867 |
| `3p1d:reasoning` | 0.1888 | 1.00 | 0.1699 | 0.00 | 0.959 |
| `3p1d:deep_research` | 0.4744 | 0.00 | 0.4270 | 1.00 | 0.859 |

## Held-out mean goodput

| cell | causal externality | least TTFT | llm-d threshold | tuned static | capacity static | focal remote |
|---|---:|---:|---:|---:|---:|---:|
| `1p3d:interactive` | 0.983 | 0.974 | 0.898 | 0.986 | 0.976 | 0.382 |
| `1p3d:reasoning` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.002 |
| `1p3d:deep_research` | 0.875 | 0.847 | 0.789 | 0.869 | 0.856 | 0.402 |
| `2p2d:interactive` | 0.961 | 0.901 | 0.954 | 0.963 | 0.919 | 0.703 |
| `2p2d:reasoning` | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 0.014 |
| `2p2d:deep_research` | 0.838 | 0.812 | 0.830 | 0.848 | 0.848 | 0.717 |
| `3p1d:interactive` | 0.772 | 0.702 | 0.782 | 0.782 | 0.673 | 0.888 |
| `3p1d:reasoning` | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 0.164 |
| `3p1d:deep_research` | 0.838 | 0.816 | 0.847 | 0.848 | 0.697 | 0.855 |

## Deployable minimax ranking

| rank | policy | worst regret | worst cell | equal-cell mean |
|---:|---|---:|---|---:|
| 1 | `causal_externality_no_capacity_v8` | 0.0100 | `3p1d:interactive` | 0.9175 |
| 2 | `least_ttft_joint` | 0.0800 | `3p1d:interactive` | 0.8938 |
| 3 | `llmd_prefix_threshold_workload_tuned` | 0.0859 | `1p3d:deep_research` | 0.8989 |

## Worst regret within each topology

| topology | causal externality | least TTFT | llm-d threshold |
|---|---:|---:|---:|
| `1p3d` | 0.0000 | 0.0281 | 0.0859 |
| `2p2d` | 0.0000 | 0.0600 | 0.0078 |
| `3p1d` | 0.0100 | 0.0800 | 0.0000 |

## Focal paired deltas

| comparator | mean delta | 95% interval |
|---|---:|---:|
| `least_ttft_joint` | +0.0237 | [+0.0149, +0.0325] |
| `llmd_prefix_threshold_workload_tuned` | +0.0185 | [+0.0017, +0.0354] |
| `static_joint_yardstick` | -0.0036 | [-0.0072, +0.0000] |
| `capacity_static` | +0.0328 | [+0.0141, +0.0515] |

## Validity

- Runs: 180 across 9 cells.
- Hard-invalid runs: 0.
- Runs with drops: 0.
- Runs with timeouts: 0.
- Runs with length caps: 0.
- Focal candidate traces: 36; all chosen actions are exact argmins.

The static comparator is a coarse condition-tuned yardstick, not an oracle.
