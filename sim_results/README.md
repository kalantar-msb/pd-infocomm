# INFOCOM 2027 compact results

These are the final compact artifacts promoted from the registered
`*_ttft_rollout_v2` campaign directories. They are the versions used for the
paper after the TTFT scheduler-rollout correction. Superseded pilot and earlier
rollout results are not included.

| Directory | Source campaign output | Contents |
|---|---|---|
| `main/` | `public_load_static_benchmark_ttft_rollout_v2/` | Frozen capacity/static selections, 432-run confirmation analysis, report |
| `ablation/` | `public_externality_decomposition_ablation_ttft_rollout_v2/` | 288-run component ablation analysis and report |
| `counterfactual/` | `public_joint_counterfactual_ttft_rollout_v2/` | Frozen request sample, forced-action analysis, report |
| `stress/` | `public_mixed_burst_benchmark_ttft_rollout_v2/` | Frozen static selection, 160-run shift/burst analysis, report |
| `topology/` | `public_final_topology_sweep_ttft_rollout_v2/` | Frozen capacity/static selections, 180-run topology analysis, report |

JSON files are machine-readable selections or aggregate analyses. Markdown
files are generated reports from the same results. Raw per-run metrics,
candidate traces, generated workload specs, plans, and stdout are reproducibly
generated under `campaigns/edpp-study/out/` and intentionally ignored by Git.

From the repository root, verify that the compact files have not changed:

```bash
shasum -a 256 -c campaigns/edpp-study/results/infocom-2027/CHECKSUMS.sha256
```

See the repository-root `INFOCOM_REPRODUCIBILITY.md` for complete commands,
validity gates, expected numerical checkpoints, and comparisons against a new
run.
