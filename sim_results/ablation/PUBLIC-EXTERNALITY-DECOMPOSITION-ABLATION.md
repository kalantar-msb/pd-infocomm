# Public externality/decomposition ablation

Routing uses smooth TTFT x E2E; reported goodput remains the hard
TTFT/mean-ITL/E2E conjunction.

## Aggregate result

| arm | equal-cell mean goodput | worst-cell goodput | mean remote fraction |
|---|---:|---:|---:|
| joint full | 0.9264 | 0.7688 | 0.4328 |
| joint own-only | 0.9111 | 0.7531 | 0.2606 |
| joint resident-only | 0.9280 | 0.7844 | 0.8944 |
| decode-first full | 0.8779 | 0.6133 | 0.4788 |

## Paired component effects

Positive means the full joint policy has higher goodput.

| comparison | mean delta | 95% CI |
|---|---:|---:|
| add resident externality | +0.0154 | [+0.0112, +0.0195] |
| add arriving-request value | -0.0016 | [-0.0053, +0.0022] |
| joint vs decode-first | +0.0485 | [+0.0305, +0.0665] |

## Per-cell mean goodput

| cell | full | own-only | resident-only | decode-first |
|---|---:|---:|---:|---:|
| `h100_homogeneous:interactive:low_0p60` | 0.989 | 0.975 | 0.992 | 0.988 |
| `h100_homogeneous:interactive:medium_0p80` | 0.972 | 0.933 | 0.978 | 0.973 |
| `h100_homogeneous:interactive:high_0p95` | 0.941 | 0.896 | 0.945 | 0.943 |
| `h100_homogeneous:reasoning:low_0p60` | 1.000 | 1.000 | 1.000 | 1.000 |
| `h100_homogeneous:reasoning:medium_0p80` | 1.000 | 1.000 | 1.000 | 1.000 |
| `h100_homogeneous:reasoning:high_0p95` | 0.998 | 1.000 | 1.000 | 1.000 |
| `h100_homogeneous:deep_research:low_0p60` | 0.914 | 0.898 | 0.914 | 0.900 |
| `h100_homogeneous:deep_research:medium_0p80` | 0.864 | 0.842 | 0.884 | 0.847 |
| `h100_homogeneous:deep_research:high_0p95` | 0.825 | 0.802 | 0.842 | 0.809 |
| `h100_a100_realistic:interactive:low_0p60` | 0.967 | 0.951 | 0.948 | 0.866 |
| `h100_a100_realistic:interactive:medium_0p80` | 0.918 | 0.882 | 0.907 | 0.700 |
| `h100_a100_realistic:interactive:high_0p95` | 0.856 | 0.822 | 0.857 | 0.613 |
| `h100_a100_realistic:reasoning:low_0p60` | 0.997 | 0.998 | 0.997 | 0.984 |
| `h100_a100_realistic:reasoning:medium_0p80` | 0.995 | 0.995 | 0.998 | 0.978 |
| `h100_a100_realistic:reasoning:high_0p95` | 0.995 | 0.994 | 0.992 | 0.977 |
| `h100_a100_realistic:deep_research:low_0p60` | 0.866 | 0.847 | 0.855 | 0.808 |
| `h100_a100_realistic:deep_research:medium_0p80` | 0.809 | 0.811 | 0.809 | 0.741 |
| `h100_a100_realistic:deep_research:high_0p95` | 0.769 | 0.753 | 0.784 | 0.677 |

## Validity

- Runs: 288 across 18 cells.
- Hard-invalid runs: 0.
- Runs with drops: 0.
- Runs with timeouts: 0.
- Runs with length caps: 0.
- Candidate trace runs: 288.
- Score-identity max error: 0.
- Max capacity term: 0.
- Max forbidden ablation component: 0.
- Constrained argmin violations: 0.
- Candidate-count violations: 0.

These comparisons are component/structure ablations, not oracle gaps.
