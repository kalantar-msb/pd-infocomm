# Public load/static-plan benchmark

The routing value is smooth TTFT x E2E; reported goodput remains
the hard TTFT/mean-ITL/E2E conjunction.

## Frozen capacity and static plans

| fleet/workload | capacity rps | cap phi | cap psi | load | rate | goodput phi | goodput psi | calibration goodput |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `h100_homogeneous:interactive` | 14.3651 | 0.40 | 0.50 | `low_0p60` | 8.6190 | 1.00 | 0.50 | 0.992 |
| `h100_homogeneous:interactive` | 14.3651 | 0.40 | 0.50 | `medium_0p80` | 11.4920 | 0.80 | 0.50 | 0.980 |
| `h100_homogeneous:interactive` | 14.3651 | 0.40 | 0.50 | `high_0p95` | 13.6468 | 0.90 | 0.50 | 0.968 |
| `h100_homogeneous:reasoning` | 0.2556 | 1.00 | 0.50 | `low_0p60` | 0.1534 | 0.00 | 0.50 | 1.000 |
| `h100_homogeneous:reasoning` | 0.2556 | 1.00 | 0.50 | `medium_0p80` | 0.2045 | 0.00 | 0.50 | 1.000 |
| `h100_homogeneous:reasoning` | 0.2556 | 1.00 | 0.50 | `high_0p95` | 0.2428 | 0.00 | 0.50 | 1.000 |
| `h100_homogeneous:deep_research` | 0.9387 | 1.00 | 0.50 | `low_0p60` | 0.5632 | 1.00 | 0.50 | 0.938 |
| `h100_homogeneous:deep_research` | 0.9387 | 1.00 | 0.50 | `medium_0p80` | 0.7510 | 1.00 | 0.50 | 0.891 |
| `h100_homogeneous:deep_research` | 0.9387 | 1.00 | 0.50 | `high_0p95` | 0.8918 | 1.00 | 0.50 | 0.856 |
| `h100_a100_realistic:interactive` | 11.5282 | 0.70 | 0.30 | `low_0p60` | 6.9169 | 1.00 | 0.20 | 0.975 |
| `h100_a100_realistic:interactive` | 11.5282 | 0.70 | 0.30 | `medium_0p80` | 9.2226 | 1.00 | 0.30 | 0.943 |
| `h100_a100_realistic:interactive` | 11.5282 | 0.70 | 0.30 | `high_0p95` | 10.9518 | 1.00 | 0.30 | 0.912 |
| `h100_a100_realistic:reasoning` | 0.2111 | 1.00 | 0.20 | `low_0p60` | 0.1267 | 0.00 | 0.20 | 0.984 |
| `h100_a100_realistic:reasoning` | 0.2111 | 1.00 | 0.20 | `medium_0p80` | 0.1689 | 0.00 | 0.20 | 0.978 |
| `h100_a100_realistic:reasoning` | 0.2111 | 1.00 | 0.20 | `high_0p95` | 0.2006 | 0.00 | 0.20 | 0.978 |
| `h100_a100_realistic:deep_research` | 0.8064 | 1.00 | 0.40 | `low_0p60` | 0.4838 | 0.90 | 0.20 | 0.881 |
| `h100_a100_realistic:deep_research` | 0.8064 | 1.00 | 0.40 | `medium_0p80` | 0.6451 | 1.00 | 0.20 | 0.834 |
| `h100_a100_realistic:deep_research` | 0.8064 | 1.00 | 0.40 | `high_0p95` | 0.7661 | 1.00 | 0.40 | 0.812 |

## Held-out mean goodput

| cell | causal externality | least TTFT | Kairos | llm-d threshold | goodput static | capacity static | focal remote |
|---|---:|---:|---:|---:|---:|---:|---:|
| `h100_homogeneous:interactive:low_0p60` | 0.980 | 0.960 | 0.973 | 0.980 | 0.991 | 0.969 | 0.577 |
| `h100_homogeneous:interactive:medium_0p80` | 0.953 | 0.912 | 0.942 | 0.963 | 0.954 | 0.924 | 0.551 |
| `h100_homogeneous:interactive:high_0p95` | 0.920 | 0.867 | 0.912 | 0.922 | 0.938 | 0.879 | 0.542 |
| `h100_homogeneous:reasoning:low_0p60` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.011 |
| `h100_homogeneous:reasoning:medium_0p80` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.017 |
| `h100_homogeneous:reasoning:high_0p95` | 1.000 | 1.000 | 1.000 | 0.995 | 1.000 | 1.000 | 0.023 |
| `h100_homogeneous:deep_research:low_0p60` | 0.900 | 0.878 | 0.889 | 0.895 | 0.905 | 0.905 | 0.375 |
| `h100_homogeneous:deep_research:medium_0p80` | 0.855 | 0.834 | 0.845 | 0.863 | 0.869 | 0.869 | 0.450 |
| `h100_homogeneous:deep_research:high_0p95` | 0.827 | 0.800 | 0.814 | 0.827 | 0.842 | 0.842 | 0.473 |
| `h100_a100_realistic:interactive:low_0p60` | 0.950 | 0.932 | 0.922 | 0.847 | 0.948 | 0.925 | 0.746 |
| `h100_a100_realistic:interactive:medium_0p80` | 0.906 | 0.863 | 0.847 | 0.554 | 0.903 | 0.876 | 0.748 |
| `h100_a100_realistic:interactive:high_0p95` | 0.849 | 0.802 | 0.744 | 0.549 | 0.861 | 0.802 | 0.738 |
| `h100_a100_realistic:reasoning:low_0p60` | 0.991 | 0.994 | 0.994 | 0.966 | 0.995 | 0.995 | 0.089 |
| `h100_a100_realistic:reasoning:medium_0p80` | 0.991 | 0.989 | 0.986 | 0.959 | 0.991 | 0.992 | 0.128 |
| `h100_a100_realistic:reasoning:high_0p95` | 0.984 | 0.981 | 0.984 | 0.956 | 0.988 | 0.989 | 0.116 |
| `h100_a100_realistic:deep_research:low_0p60` | 0.863 | 0.852 | 0.847 | 0.825 | 0.858 | 0.864 | 0.636 |
| `h100_a100_realistic:deep_research:medium_0p80` | 0.822 | 0.806 | 0.803 | 0.791 | 0.816 | 0.814 | 0.677 |
| `h100_a100_realistic:deep_research:high_0p95` | 0.786 | 0.769 | 0.752 | 0.733 | 0.795 | 0.795 | 0.686 |

## Deployable minimax ranking

| rank | policy | worst regret | worst cell | equal-cell mean goodput |
|---:|---|---:|---|---:|
| 1 | `causal_externality_no_capacity_v8` | 0.0100 | `h100_homogeneous:interactive:medium_0p80` | 0.9209 |
| 2 | `least_ttft_joint` | 0.0542 | `h100_homogeneous:interactive:high_0p95` | 0.9022 |
| 3 | `kairos_paper_alpha_1p3` | 0.1050 | `h100_a100_realistic:interactive:high_0p95` | 0.9030 |
| 4 | `llmd_prefix_threshold_workload_tuned` | 0.3517 | `h100_a100_realistic:interactive:medium_0p80` | 0.8680 |

## Validity

- Runs: 432 across 18 cells.
- Hard-invalid runs: 0.
- Runs with drops: 0.
- Runs with timeouts: 0.
- Runs with length caps: 0.
- Focal candidate traces: 72; all chosen actions are exact argmins.

Static-plan differences are descriptive static-plan gaps, not oracle
or end-to-end policy-regret estimates.
