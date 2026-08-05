# Joint causal-SLO-externality P/D placement — sim2real transfer bundle

This folder is the pre-bootstrap **sim2real transfer bundle** for the INFOCOM 2027
prefill/decode routing study. The policy was developed and evaluated entirely in
the BLIS discrete-event simulator; this bundle carries it onto a real llm-d
cluster.

Source artifact: `INFOCOM_REPRODUCIBILITY.md` on the `infocom-implementation`
branch (see [Provenance](#provenance) for exact pins).

## Provenance

The BLIS fork is **not** vendored here — it is already the `inference-sim`
submodule of the `sim2real` project. Recorded so this bundle is self-describing:

| Component | Pin | Notes |
|---|---|---|
| BLIS fork | `vishakha-ramani/inference-sim` | not a submodule of this repo |
| Branch | `infocom-implementation` | tracked branch of the `sim2real` submodule |
| Commit | `871b169bb13934ca8dd1e002638e1f6bf490b3b5` | `v0.7.15-244-g871b169b`, 2026-08-05, "merge upstream and publish INFOCOM reproduction artifact" |
| Upstream merge base | `3340de78` | `inference-sim/main` merged through this commit |
| sim2real superproject | records `583f719507f31c24dd675d30daa79c33950c5516` for `inference-sim` | **the pin is behind the working checkout** — sim2real's index has not been updated to `871b169b`. Commit the submodule bump before treating sim2real as the reproducible source of this bundle. |
| llm-d-router | pinned by `sim2real-bootstrap` Task 1 | expect `v0.9.0` (submodule at `5f4e762f` in sibling bundles) |
| Everything under `sim_results/` | verified bit-exact | `shasum -a 256 -c` against the branch's `CHECKSUMS.sha256`, all 16 files OK |

## The question

A 1P2D disaggregated fleet must place each arriving request as a **joint**
`(decode instance, prefill location)` action: prefill locally on the decode
instance, or remotely on a prefill instance and transfer the KV. The focal
policy scores every joint candidate by a projected-value rule:

```text
score(d, p) = V * ( causal SLO externality imposed on residents
                    − projected SLO value gained by the arrival )
```

and takes the argmin. Both terms come from a **scheduler rollout** that predicts
admission, first token, and completion from a frozen FCFS snapshot, using
calibrated per-GPU iteration-time coefficients. See `config.md` §"Latency-law
coefficients" for what those coefficients are and where they enter.

Reported goodput is the hard three-way conjunction of per-request TTFT, mean
ITL, and E2E against per-workload targets.

## What simulation found — read this before deploying

The 432-run held-out confirmation is vendored under `sim_results/main/`. Split by
fleet (computed from `sim_results/main/confirmation_result.json`, 9 cells each,
regret = gap to the best deployable policy in that cell):

| Policy | homogeneous H100 worst regret | homogeneous mean goodput | **heterogeneous** worst regret | **heterogeneous** mean goodput |
|---|---:|---:|---:|---:|
| causal externality (focal) | 0.0100 | 0.9372 | **0.0031** | **0.9045** |
| least projected TTFT | 0.0542 | 0.9169 | 0.0475 | 0.8876 |
| Kairos (paper, α=1.3) | 0.0217 | 0.9307 | 0.1050 | 0.8753 |
| workload-tuned llm-d threshold | **0.0047** | **0.9383** | 0.3517 | 0.7977 |

**On the homogeneous H100 fleet the focal policy does not win — stock llm-d
does.** 0.0047 vs 0.0100 worst regret, 0.9383 vs 0.9372 mean goodput. The entire
advantage lives in the heterogeneous cells, and it is concentrated in
interactive chat at load:

| heterogeneous cell | focal | llm-d threshold | gap |
|---|---:|---:|---:|
| interactive : 0.80 C | 0.9058 | 0.5542 | **+0.352** |
| interactive : 0.95 C | 0.8492 | 0.5492 | +0.300 |
| interactive : 0.60 C | 0.9500 | 0.8467 | +0.103 |
| deep_research : 0.95 C | 0.7859 | 0.7328 | +0.053 |
| reasoning : all loads | ~0.99 | ~0.96 | +0.03 |

The mechanism is hardware awareness. llm-d's decode placement balances queue
depth and prefix hits; it has no notion that one decoder is an A100 and drains
~1.6× slower per token (`c1` 0.0782 vs 0.0476 µs/token). It therefore feeds the
slow decoder as if it were fast and collapses. The focal policy evaluates each
candidate under **that candidate's** coefficients, so it prices the A100
correctly.

**Consequence for this transfer: the heterogeneous fleet is the experiment, not
an extension of it.** A homogeneous-only transfer would test the one
configuration where the paper's own simulation says the policy has no edge.
See [Scope](#scope) for what that costs.

Paired held-out-seed deltas from focal to each comparator, all 18 cells, 95% CI
(`focal_paired_deltas`):

| vs | mean Δ goodput | 95% CI |
|---|---:|---|
| least projected TTFT | +0.0187 | [+0.0142, +0.0231] |
| Kairos (paper) | +0.0179 | [+0.0104, +0.0253] |
| workload-tuned llm-d | +0.0528 | [+0.0280, +0.0777] |
| goodput-tuned static plan | −0.0043 | [−0.0067, −0.0020] |

The last row is the honest ceiling: a per-condition-tuned static split still
edges out the online policy by 0.4 points. That is a *static-plan gap*, not
policy regret, and the static plan is not deployable (it knows the workload,
fleet, and offered rate in advance).

## Arms

| Arm | Source | Custom code | Role |
|---|---|---|---|
| `llmdthreshold` | stock llm-d plugins | none | baseline; also *is* the paper's decode-first decomposition (see below) |
| `causalext` | `algorithms/causal_slo_externality.go` | yes | treatment (focal) |
| `leastttft` | `algorithms/least_ttft_joint.go` | yes | comparator — same rollout, TTFT-only objective |
| `kairos` | `algorithms/kairos_paper.go` | yes | published comparator, α=1.3, β=1.0 |
| `causalextnoext` | `causal_slo_externality.go`, overlay `noExternality: true` | shared image | ablation: drop the resident externality |

`llmdthreshold` is a genuine two-for-one. llm-d's `disagg-profile-handler` runs
the **decode** profile first, then asks a decider whether to add a remote
prefill (`pd_profile_handler.go:186` —
`decider.disaggregate(ctx, request, profileResults[decodeProfile].TargetEndpoints[0])`).
That is a sequential decomposition of the joint choice, which is exactly the
`full − decode-first = +0.0485` ablation arm in
`sim_results/ablation/ablation_result.json`. Stock llm-d cannot express the joint
argmin — which is why the focal arm needs a custom ProfileHandler, not a Scorer.

## Scope

**In scope.** Heterogeneous H100-prefill / (H100 + A100)-decode 1P2D, plus the
homogeneous H100 1P2D as a control; three public request shapes; three loads
normalized to measured real capacity; four deployable arms.

**Blocking prerequisite.** The llm-d-benchmark scenario schema has a single
`decode:` block with a single `acceleratorType`. A mixed-accelerator decode pool
is not expressible today and needs framework work: two decode model-services
joined to one InferencePool, plus per-pod GPU type visible to the EPP (node
label `nvidia.com/gpu.product` surfaced as an endpoint attribute). Without it
the focal policy cannot pick per-candidate coefficients and the experiment
reduces to the homogeneous control, where simulation predicts no win.

**Out of scope, deliberately.**

- The **joint counterfactual** campaign (144 sampled decisions, 432 forced
  one-request deviations, replay gates). It requires re-running the same request
  against a forced alternative from identical state; not reproducible on a live
  cluster. Vendored under `sim_results/counterfactual/` as sim-side evidence
  only.
- Both **static-plan yardsticks** (`static_joint_yardstick`, `capacity_static`).
  They are per-condition sim-tuned `(φ, ψ)` splits, not deployable policies.
- The **topology sweep** and **mixed-burst stress** campaigns
  (`sim_results/topology/`, `sim_results/stress/`) — vendored for reference,
  not transferred in this round.
- **Mixed concurrent SLO classes.** The policy supports them per request and they
  are arguably where the externality term should shine — with one uniform class it
  degenerates toward a load count. But all 432 confirmation runs used a single
  class, so it is an unvalidated path needing its own protocol, capacity
  selection, and held-out seeds. A follow-on campaign, not this reproduction.
  `LIMITATIONS §D1-D4`.

## Layout

```
README.md          # this file — scope, arms, provenance
config.md          # what the plugin needs: vLLM/EPP/SLO/observe/sim-flag tables
LIMITATIONS.md     # what it cannot do: predictor, metric, signals, platform
algorithms/
  causal_slo_externality.go   # focal — joint argmin, // TRANSLATE: markers
  least_ttft_joint.go         # comparator — same rollout, TTFT objective
  kairos_paper.go             # comparator — Algorithm 1, discrete chunks
workloads/
  interactive-chat-single-turn.yaml   # verbatim from the branch
  reasoning-single-turn.yaml
  deep-research-single-turn.yaml
inputs/
  coeffs-llama70b-h100-tp4.json       # latency-law coefficients, per GPU type
  coeffs-llama70b-a100real-tp4.json
  hetero-realistic-1p2d.yaml          # the sim fleet bundle, for reference
sim_results/       # vendored results/infocom-2027/ + CHECKSUMS (sim ground truth)
```

`sim2real-bootstrap` adds `transfer.yaml`, `baselines/baseline.yaml`,
`baselines/defaults/`, and the pinned `llm-d-router` submodule.

## Next steps

1. Resolve the **CONFIRM** rows in `config.md` (GPU labels, replica counts, the
   mixed-decode-pool question, per-workload vs per-class configuration).
2. Decide the heterogeneous-decode-pool approach — this gates everything.
2b. Read `LIMITATIONS.md`. Three items are cheap to close before any cluster time
   and each removes a reviewer objection: validate the A100 coefficients for
   additivity (`§A4`), add the `rem = 1` counter (`§C5`), and run the
   hardware-blind control arm (`§A5`).
3. From the sim2real repo: `/sim2real-bootstrap <path-to-this-folder>`.
4. `/sim2real-translate`, then `sim2real translate --resume`.
5. Capacity probe on the real fleet to establish `C_w` per workload/fleet, then
   freeze `{0.60, 0.80, 0.95} × C_w` before running any arm.
6. Build → deploy → `/sim2real-check` + `/sim2real-analyze`.

## Translation notes

`algorithms/*.go` are reference ports against llm-d-router interfaces carrying
`// TRANSLATE:` markers where the exact target API must be confirmed against the
pinned checkout. The touchpoints, hardest first:

1. **Joint enumeration.** `ProfileHandler.Pick` is called iteratively with
   accumulated `profileResults`; each profile returns a `*ProfileRunResult`
   already reduced to chosen endpoints. Getting a joint argmin requires a custom
   Picker that returns the full ranked candidate list plus a ProfileHandler that
   selects the pair in `ProcessResults`. Confirm this is legal — the alternative
   is accepting the decode-first decomposition, which forfeits the claim.
2. **Resident state.** The rollout needs, per resident: remaining decode steps,
   arrival time, realized TTFT, and resident context tokens.
   `datalayer.Metrics` exposes only aggregates — `RunningRequestsSize`,
   `WaitingQueueSize`, `KVCacheUsagePercent`, `KvCacheMaxTokenCapacity`,
   `CacheBlockSize`. The EPP must maintain its own shadow resident table from
   the request lifecycle it already observes. This is the largest engineering
   item and the main threat to fidelity.
3. **`S_pf` (resident prefill tokens)** has no vLLM metric at all. Either derive
   it from the shadow table or declare the term zero and record the deviation.
4. **Per-pod GPU type**, to select coefficients per candidate. Node label
   `nvidia.com/gpu.product` → endpoint attribute. Required for the
   heterogeneous arm; see [Scope](#scope).
5. **`N̂_out` (predicted output length)** — a **per-class configured table**, not a
   scalar. `Wd(a_r, o)` needs it for the arrival and `nHatFor(r.SLOClass).mean()`
   needs it for every resident's remaining-steps estimate. Keep it censored —
   reading true output length invalidates the arm. See `config.md`
   §"Output-length priors"; note the `rem = 1` pinning in `LIMITATIONS §C5`.

5b. **Per-request SLO class.** τ and `N̂_out` are both resolved per request via
   `targetsFor(class)` / `nHatFor(class)`. `InferenceObjective` cannot carry the
   class (it has only `Priority` and `PoolRef`), so a header plus a data producer
   is the shape — both unresolved on the pinned checkout. The plugin must
   **reject** an unknown class rather than silently defaulting. See `config.md`
   §"Request classification".
6. **KV transfer cost** `c_xfer`, size-aware (`--edpp-c-xfer-size-aware`). Must
   be measured on the target interconnect, not inherited.
7. **Single EPP replica** for the first experiment. The shadow resident table is
   in-process; replicating the EPP splits it and silently degrades the policy.
