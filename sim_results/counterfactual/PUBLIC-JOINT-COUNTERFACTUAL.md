# Public joint request counterfactuals

This is an exact one-request forced-action diagnostic, not a global oracle.
All other request actions remain fixed to the final policy's captured plan.

## Overall

- Sampled decisions: 144 across 18 cells.
- Agreement with a best forced action: 91.0%.
- Positive one-request regret: 9.0%.
- Mean goodput regret per sampled decision: 0.00053.
- Mean equivalent good requests recovered: 0.097.
- Local decisions that should be remote: 1.
- Remote decisions that should be local: 0.
- Decisions with a decoder error: 12.

## Error decomposition

| class | decisions | total goodput lost |
|---|---:|---:|
| agreement | 131 | 0.0000 |
| decoder_only | 12 | 0.0696 |
| placement_only | 1 | 0.0062 |
| decoder_and_placement | 0 | 0.0000 |

## By workload

| workload | decisions | agreement | positive regret | mean regret |
|---|---:|---:|---:|---:|
| deep_research | 48 | 83.3% | 16.7% | 0.00104 |
| interactive | 48 | 93.8% | 6.2% | 0.00028 |
| reasoning | 48 | 95.8% | 4.2% | 0.00026 |

## By fleet

| fleet | decisions | agreement | positive regret | mean regret |
|---|---:|---:|---:|---:|
| h100_a100_realistic | 72 | 88.9% | 11.1% | 0.00062 |
| h100_homogeneous | 72 | 93.1% | 6.9% | 0.00043 |

## Validity

- Online runs: 18; replay gates: 18; deviation runs: 432.
- Replay mismatches: 0.
- Online argmin violations: 0.
- Candidate-count violations: 0.
- Maximum score-identity error: 0.
- Hard-invalid runs: 0.
- Deviation runs with drops/timeouts/length caps: 0/0/0.
