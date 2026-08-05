# Public mixed-workload and burst benchmark

The routing value is smooth TTFT x E2E; reported goodput remains the hard
TTFT/mean-ITL/E2E conjunction.

Kairos arm: `kairos_paper_alpha_1p3`.

## Frozen static plans

| fleet/profile | phi | psi | calibration goodput |
|---|---:|---:|---:|
| `h100_homogeneous:sequential_shift` | 1.00 | 0.50 | 0.933 |
| `h100_homogeneous:concurrent_poisson` | 0.50 | 0.50 | 0.961 |
| `h100_homogeneous:concurrent_gamma_cv3` | 0.50 | 0.50 | 0.731 |
| `h100_homogeneous:short_spike` | 0.50 | 0.50 | 0.821 |
| `h100_a100_realistic:sequential_shift` | 1.00 | 0.50 | 0.902 |
| `h100_a100_realistic:concurrent_poisson` | 1.00 | 0.00 | 0.927 |
| `h100_a100_realistic:concurrent_gamma_cv3` | 0.50 | 0.50 | 0.667 |
| `h100_a100_realistic:short_spike` | 1.00 | 0.50 | 0.747 |

## Held-out mean goodput

| cell | causal externality | least TTFT | Kairos | class-aware llm-d | tuned static | focal remote |
|---|---:|---:|---:|---:|---:|---:|
| `h100_homogeneous:sequential_shift` | 0.922 | 0.921 | 0.926 | 0.926 | 0.934 | 0.237 |
| `h100_homogeneous:concurrent_poisson` | 0.992 | 0.978 | 0.984 | 0.917 | 0.950 | 0.514 |
| `h100_homogeneous:concurrent_gamma_cv3` | 0.850 | 0.816 | 0.824 | 0.608 | 0.735 | 0.323 |
| `h100_homogeneous:short_spike` | 0.952 | 0.896 | 0.912 | 0.792 | 0.814 | 0.509 |
| `h100_a100_realistic:sequential_shift` | 0.915 | 0.905 | 0.911 | 0.903 | 0.890 | 0.459 |
| `h100_a100_realistic:concurrent_poisson` | 0.973 | 0.942 | 0.937 | 0.874 | 0.884 | 0.686 |
| `h100_a100_realistic:concurrent_gamma_cv3` | 0.829 | 0.774 | 0.784 | 0.644 | 0.722 | 0.436 |
| `h100_a100_realistic:short_spike` | 0.920 | 0.809 | 0.827 | 0.698 | 0.693 | 0.683 |

## Held-out mean goodput by class

| cell | policy | interactive | reasoning | deep research | remote fraction |
|---|---|---:|---:|---:|---:|
| `h100_homogeneous:sequential_shift` | `causal_externality_no_capacity_v8` | 1.000 | 1.000 | 0.800 | 0.237 |
| `h100_homogeneous:sequential_shift` | `least_ttft_joint` | 1.000 | 1.000 | 0.797 | 0.102 |
| `h100_homogeneous:sequential_shift` | `kairos_paper_alpha_1p3` | 1.000 | 1.000 | 0.808 | 0.256 |
| `h100_homogeneous:sequential_shift` | `llmd_prefix_threshold_class_tuned` | 1.000 | 1.000 | 0.808 | 0.992 |
| `h100_homogeneous:sequential_shift` | `static_joint_yardstick` | 1.000 | 1.000 | 0.829 | 1.000 |
| `h100_homogeneous:concurrent_poisson` | `causal_externality_no_capacity_v8` | 0.996 | 1.000 | 0.925 | 0.514 |
| `h100_homogeneous:concurrent_poisson` | `least_ttft_joint` | 0.981 | 1.000 | 0.925 | 0.090 |
| `h100_homogeneous:concurrent_poisson` | `kairos_paper_alpha_1p3` | 0.987 | 1.000 | 0.931 | 0.242 |
| `h100_homogeneous:concurrent_poisson` | `llmd_prefix_threshold_class_tuned` | 0.915 | 0.975 | 0.944 | 0.985 |
| `h100_homogeneous:concurrent_poisson` | `static_joint_yardstick` | 0.950 | 1.000 | 0.931 | 0.500 |
| `h100_homogeneous:concurrent_gamma_cv3` | `causal_externality_no_capacity_v8` | 0.849 | 0.925 | 0.844 | 0.323 |
| `h100_homogeneous:concurrent_gamma_cv3` | `least_ttft_joint` | 0.814 | 0.825 | 0.838 | 0.189 |
| `h100_homogeneous:concurrent_gamma_cv3` | `kairos_paper_alpha_1p3` | 0.826 | 0.725 | 0.812 | 0.354 |
| `h100_homogeneous:concurrent_gamma_cv3` | `llmd_prefix_threshold_class_tuned` | 0.598 | 0.575 | 0.781 | 0.983 |
| `h100_homogeneous:concurrent_gamma_cv3` | `static_joint_yardstick` | 0.728 | 0.725 | 0.850 | 0.500 |
| `h100_homogeneous:short_spike` | `causal_externality_no_capacity_v8` | 0.957 | 1.000 | 0.869 | 0.509 |
| `h100_homogeneous:short_spike` | `least_ttft_joint` | 0.895 | 1.000 | 0.874 | 0.176 |
| `h100_homogeneous:short_spike` | `kairos_paper_alpha_1p3` | 0.913 | 0.994 | 0.885 | 0.377 |
| `h100_homogeneous:short_spike` | `llmd_prefix_threshold_class_tuned` | 0.784 | 0.900 | 0.890 | 0.981 |
| `h100_homogeneous:short_spike` | `static_joint_yardstick` | 0.810 | 0.887 | 0.856 | 0.500 |
| `h100_a100_realistic:sequential_shift` | `causal_externality_no_capacity_v8` | 0.988 | 0.989 | 0.775 | 0.459 |
| `h100_a100_realistic:sequential_shift` | `least_ttft_joint` | 0.980 | 0.981 | 0.762 | 0.290 |
| `h100_a100_realistic:sequential_shift` | `kairos_paper_alpha_1p3` | 0.978 | 0.986 | 0.778 | 0.503 |
| `h100_a100_realistic:sequential_shift` | `llmd_prefix_threshold_class_tuned` | 0.948 | 0.978 | 0.792 | 0.992 |
| `h100_a100_realistic:sequential_shift` | `static_joint_yardstick` | 0.915 | 0.978 | 0.785 | 1.000 |
| `h100_a100_realistic:concurrent_poisson` | `causal_externality_no_capacity_v8` | 0.984 | 1.000 | 0.811 | 0.686 |
| `h100_a100_realistic:concurrent_poisson` | `least_ttft_joint` | 0.948 | 1.000 | 0.850 | 0.317 |
| `h100_a100_realistic:concurrent_poisson` | `kairos_paper_alpha_1p3` | 0.942 | 1.000 | 0.850 | 0.498 |
| `h100_a100_realistic:concurrent_poisson` | `llmd_prefix_threshold_class_tuned` | 0.872 | 1.000 | 0.867 | 0.985 |
| `h100_a100_realistic:concurrent_poisson` | `static_joint_yardstick` | 0.885 | 1.000 | 0.844 | 1.000 |
| `h100_a100_realistic:concurrent_gamma_cv3` | `causal_externality_no_capacity_v8` | 0.827 | 0.900 | 0.844 | 0.436 |
| `h100_a100_realistic:concurrent_gamma_cv3` | `least_ttft_joint` | 0.768 | 0.800 | 0.856 | 0.274 |
| `h100_a100_realistic:concurrent_gamma_cv3` | `kairos_paper_alpha_1p3` | 0.779 | 0.800 | 0.850 | 0.471 |
| `h100_a100_realistic:concurrent_gamma_cv3` | `llmd_prefix_threshold_class_tuned` | 0.632 | 0.650 | 0.817 | 0.983 |
| `h100_a100_realistic:concurrent_gamma_cv3` | `static_joint_yardstick` | 0.712 | 0.750 | 0.861 | 0.499 |
| `h100_a100_realistic:short_spike` | `causal_externality_no_capacity_v8` | 0.924 | 1.000 | 0.831 | 0.683 |
| `h100_a100_realistic:short_spike` | `least_ttft_joint` | 0.803 | 1.000 | 0.845 | 0.365 |
| `h100_a100_realistic:short_spike` | `kairos_paper_alpha_1p3` | 0.824 | 0.994 | 0.845 | 0.589 |
| `h100_a100_realistic:short_spike` | `llmd_prefix_threshold_class_tuned` | 0.684 | 0.994 | 0.836 | 0.981 |
| `h100_a100_realistic:short_spike` | `static_joint_yardstick` | 0.681 | 0.975 | 0.808 | 1.000 |

## Short-spike segment goodput

| cell | policy | before spike | during 1.60 load | after spike |
|---|---|---:|---:|---:|
| `h100_homogeneous:short_spike` | `causal_externality_no_capacity_v8` | 0.993 | 0.891 | 0.988 |
| `h100_homogeneous:short_spike` | `least_ttft_joint` | 0.984 | 0.768 | 0.970 |
| `h100_homogeneous:short_spike` | `kairos_paper_alpha_1p3` | 0.987 | 0.802 | 0.977 |
| `h100_homogeneous:short_spike` | `llmd_prefix_threshold_class_tuned` | 0.942 | 0.559 | 0.941 |
| `h100_homogeneous:short_spike` | `static_joint_yardstick` | 0.961 | 0.599 | 0.939 |
| `h100_a100_realistic:short_spike` | `causal_externality_no_capacity_v8` | 0.986 | 0.829 | 0.967 |
| `h100_a100_realistic:short_spike` | `least_ttft_joint` | 0.971 | 0.595 | 0.918 |
| `h100_a100_realistic:short_spike` | `kairos_paper_alpha_1p3` | 0.976 | 0.661 | 0.891 |
| `h100_a100_realistic:short_spike` | `llmd_prefix_threshold_class_tuned` | 0.930 | 0.486 | 0.737 |
| `h100_a100_realistic:short_spike` | `static_joint_yardstick` | 0.908 | 0.451 | 0.784 |

## Deployable minimax ranking

| rank | policy | worst regret | worst cell | equal-cell mean |
|---:|---|---:|---|---:|
| 1 | `causal_externality_no_capacity_v8` | 0.0031 | `h100_homogeneous:sequential_shift` | 0.9191 |
| 2 | `kairos_paper_alpha_1p3` | 0.0924 | `h100_a100_realistic:short_spike` | 0.8880 |
| 3 | `least_ttft_joint` | 0.1110 | `h100_a100_realistic:short_spike` | 0.8801 |
| 4 | `llmd_prefix_threshold_class_tuned` | 0.2421 | `h100_homogeneous:concurrent_gamma_cv3` | 0.7953 |

## Descriptive equal-class check

The registered ranking above weights every request equally. Capacity-balanced
mixtures contain many more interactive requests, so the following non-primary
check gives interactive, reasoning, and deep research equal weight within a cell.

| rank | policy | worst regret | worst cell | equal-cell macro mean |
|---:|---|---:|---|---:|
| 1 | `causal_externality_no_capacity_v8` | 0.0026 | `h100_homogeneous:sequential_shift` | 0.9183 |
| 2 | `least_ttft_joint` | 0.0492 | `h100_a100_realistic:concurrent_gamma_cv3` | 0.8975 |
| 3 | `kairos_paper_alpha_1p3` | 0.0849 | `h100_homogeneous:concurrent_gamma_cv3` | 0.8960 |
| 4 | `llmd_prefix_threshold_class_tuned` | 0.2213 | `h100_homogeneous:concurrent_gamma_cv3` | 0.8433 |

## Focal paired deltas

| comparator | mean delta | 95% interval |
|---|---:|---:|
| `least_ttft_joint` | +0.0390 | [+0.0246, +0.0535] |
| `kairos_paper_alpha_1p3` | +0.0311 | [+0.0177, +0.0444] |
| `llmd_prefix_threshold_class_tuned` | +0.1238 | [+0.0912, +0.1565] |
| `static_joint_yardstick` | +0.0915 | [+0.0648, +0.1182] |

## Validity

- Runs: 160 across 8 cells.
- Hard-invalid runs: 0.
- Runs with drops: 0.
- Runs with timeouts: 0.
- Runs with length caps: 0.
- Focal candidate traces: 32; all chosen actions are exact argmins.

The static comparator is a condition-tuned yardstick, not an oracle.
