# Configuration: joint causal-SLO-externality P/D placement

Source campaign: `campaigns/edpp-study/` on the BLIS fork at
`871b169bb13934ca8dd1e002638e1f6bf490b3b5` (see `README.md` §Provenance). This
file translates the simulated experiment into a real-cluster sim2real bundle.

Every value below is traced to the campaign source. Values marked **CONFIRM** are
cluster-specific decisions the simulation did not fix — set them to match your
hardware before bootstrap.

Known limitations of the policy, the predictor, the metric, and the platform are
in `LIMITATIONS.md`. This file states what the plugin *needs*; that one states
what it cannot do. Sections below cross-reference it as `LIMITATIONS §x`.

## vLLM Pod Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `meta-llama/Llama-3.3-70B-Instruct` | `run_decisive_campaign.py:29` uses the BLIS alias `meta-llama/llama-3.3-70b-instruct`; `defaults.yaml:10-13` maps it to this HF repo. Gated repo — the cluster needs an HF token. |
| GPU | H100_SXM_80GB | `defaults.yaml:11`. The decode pool is **mixed** on the primary fleet — see [Fleet topology](#fleet-topology). |
| `tensor_parallel_size` | 4 | `defaults.yaml:12`; `repro_llama70b.sh` pins `--tp 4` |
| `max_num_seqs` | 256 | `run_public_workload_heterogeneity_closeout.py:202` (`--max-num-running-reqs 256`) |
| `max_num_batched_tokens` | 2048 | BLIS `--max-num-scheduled-tokens` default (`cmd/root.go:1426`); campaign does not override |
| `block_size` | 16 | BLIS `--block-size-in-tokens` default (`cmd/root.go:1429`) |
| `gpu_memory_utilization` | 0.9 | **CONFIRM** — not modeled by the simulator |
| `max_model_len` | 131072 | deep-research inputs reach 121K tokens (closeout protocol: catalog max capped 150K → 121K "for the 128K model context") |
| `enable_chunked_prefill` | True | the policy's chunked-prefill overlap term assumes it; `nChunks`/`chunk` are core to the score |
| `enable_prefix_caching` | True | the llm-d baseline arm scores `precise-prefix-cache:2`; without caching that arm is degenerate |
| Number of vLLM pods | 2 | decode replicas. Prefill is a separate pool — see below. 1P2D total. |

> **Bootstrap note.** `generate_from_config.py` errors on unknown models and
> hardware. `MODEL_METADATA` has no `meta-llama/Llama-3.3-70B-Instruct` entry and
> `HARDWARE_LABELS` has `H100_SXM_80GB` but the 70B model needs a `shortName`,
> `path`, `size`, and `maxModelLen`. Add:
> `{"shortName": "meta-llama-llama-3-3-70b-instruct", "path": "models/meta-llama/Llama-3.3-70B-Instruct", "size": "1Ti", "maxModelLen": 131072}`.
> A 70B at TP=4 also wants a larger PVC than the 14B entries — **CONFIRM** the
> `size` value against actual fp16 weight footprint (~140 GB).

## Fleet topology

Every campaign run uses **1P2D**: one prefill instance, two decode instances,
each TP=4. At TP=4 that is 12 GPUs.

| Fleet | Prefill | Decode 0 | Decode 1 | Source |
|---|---|---|---|---|
| **`hetero` (primary)** | H100 | H100 | **A100-80GB** | `inputs/hetero-realistic-1p2d.yaml` |
| `homo` (control) | H100 | H100 | H100 | `topology_args()` without `--policy-config` |

The heterogeneous fleet is where the policy's advantage lives (README §What
simulation found: +0.352 goodput vs stock llm-d on interactive at 0.80 C). It is
also **not expressible in the current llm-d-benchmark scenario schema**, which has
one `decode:` block with one `acceleratorType`.

**CONFIRM — mixed decode pool.** Pick one before bootstrap:

1. Two decode model-service releases joined to one InferencePool, distinguished
   by node label. Faithful; needs scenario-schema work.
2. One decode pool, node-affinity split across two node groups. Simpler; the
   chart must not homogenize `acceleratorType`.
3. Homogeneous only. Cheapest, but tests the configuration where simulation
   predicts the focal policy loses to stock llm-d (0.0100 vs 0.0047 worst regret).

Either of (1)/(2) additionally requires **per-pod GPU type visible to the EPP**
(node label `nvidia.com/gpu.product` surfaced as an endpoint attribute), because
the policy selects coefficients per candidate. Without it, option 3 is the only
honest choice.

## Latency-law coefficients

**This is the one part of the policy with no analogue in the sibling bundles, so
read it before translating.**

### What they are

The policy does not read latency from anywhere. It *predicts* it, from a linear
iteration-time law (`sim/edpp_coeffs.go`):

```text
T_iter_decode (B_dec, KV, S_pf) = α   + c0·B_dec + c1·KV + c_pf·S_pf
T_iter_prefill(S_pf)            = α_p + c_pf·S_pf

W_p(a_p, a_r) = c_pf·a_p + c_attn·a_p·(a_r − a_p/2)      # prefill work, µs
W_d(a_r, o)   = c0·o     + c1·o·(a_r + (o−1)/2)          # decode work, µs
```

where `B_dec` is the resident decode batch size, `KV` the summed resident context
tokens, `S_pf` the resident prefill tokens, `a_r` the full input length, `a_p` the
uncached suffix, and `o` the predicted output length `N̂_out`.

Six numbers per GPU type:

| Coefficient | JSON key | H100 | A100 | Units |
|---|---|---:|---:|---|
| `α` decode fixed per-iteration | `decode.alpha_us` | 16613.54 | 25563.82 | µs |
| `α_p` prefill fixed per-iteration | `prefill.alpha_p_us` | 16617.85 | 25568.35 | µs |
| `c0` decode per-request | `decode.c0_us_per_req` | 5.3473 | 5.9453 | µs/req |
| `c1` decode KV read per resident token | `decode.c1_us_per_token` | 0.047614 | 0.078228 | µs/token |
| `c_pf` prefill compute per token | `prefill.c_pf_us_per_token` | 6.1447 | 9.7942 | µs/token |
| `c_attn` prefill attention | `prefill.c_attn_us_per_unit` | 1.00752e-4 | 1.59777e-4 | µs/token² |

Files: `inputs/coeffs-llama70b-h100-tp4.json`,
`inputs/coeffs-llama70b-a100real-tp4.json`.

The A100/H100 ratios (`c1` 1.64×, `c_pf` 1.59×, `α` 1.54×) **are the hardware
heterogeneity**. They are the only reason the focal policy can tell a slow decoder
from a fast one, and therefore the only reason it beats hardware-blind llm-d.

### Where they enter sim2real

They are **policy parameters, not simulator parameters**. In BLIS they arrive via
`--edpp-coeffs <file>` (`run_decisive_campaign.py`, `edpp_common()`), and on the
heterogeneous fleet via `coeffs_by_gpu:` in the policy-config bundle
(`inputs/hetero-realistic-1p2d.yaml`).

There is **exactly one mechanical place** they enter the transfer: as plugin
`parameters:` in each algorithm's EPP config overlay, which `sim2real translate`
emits as `workspace/translations/<hash>/generated/<algo>/<algo>_config.yaml`. Six
numbers per GPU type is small enough to inline — no ConfigMap mount needed, same
pattern as the sibling `predictive_routing` bundle's inline tier-0 model. See the
`coeffs:` block in the treatment EPP config below.

Nothing else in sim2real knows about them. The pipeline has no calibration stage
and no coefficient validation (`grep -ri 'calibrat\|coeff' pipeline/` → no hits).
Whatever number lands in that overlay is what the running policy uses.

### Provenance, and why "refit on the target cluster" is not straightforward

The frozen files were **not** fit against real vLLM. They were fit against
BLIS's own trained-physics latency model: `scripts/calibration/repro_llama70b.sh`
runs seven `blis run` sweeps with `BLIS_STEP_CSV` set, then
`fit_coeffs.py` regresses the additive law onto those per-step CSVs. That is why
the reported `r2` is 0.999999999 and validation MAPE 0.019% — the additive law is
being fit to data generated by the model it linearizes. Because the runs are
deterministic, `repro_llama70b.sh` regenerates all six coefficients **bit-exactly**
and prints `CHECKPOINT: PASS`; the provenance chain is closed and reproducible
with no GPU at all.

So the coefficients are best understood as *a compressed model of BLIS's H100/A100
physics*, handed to the policy as its forecasting prior. Three consequences:

1. There is no tool that fits these from real-cluster data. `blis calibrate`
   sounds like it does but does not — it *compares* observed vs simulated
   latencies and reports MAPE/RMSE (`cmd/calibrate.go:41`), it does not solve for
   `α, c0, c1, c_pf, c_attn`. Fitting from real data would need per-step
   `(t_iter, B_dec, KV, S_pf, pf_ctx)` tuples, which vLLM does not export.
2. Shipping the frozen H100/A100 files as-is is therefore the **default and
   defensible** choice for the first transfer: it deploys precisely the policy the
   paper evaluated. If real physics diverges from BLIS's, the policy makes
   miscalibrated decisions — which is a real and reportable finding about the
   policy's robustness to coefficient error, not a defect in the experiment.
3. What matters most is not absolute accuracy but the **A100/H100 ratio**, since
   that is what drives the placement decision. Sanity-check it on the target
   cluster: measure steady-state decode throughput at fixed batch on each GPU
   type and confirm the ratio is near 1.6×. If it is not, the heterogeneity the
   policy prices is not the heterogeneity present, and that must be declared.

**Recommended cheap addition.** Run the focal arm twice — once with the frozen
coefficients, once with the A100 set deliberately replaced by the H100 set
(making the policy hardware-blind while keeping everything else identical). The
delta isolates how much of the win is hardware awareness versus the externality
term. Two extra arms, no new tooling.

## SLO targets and the goodput definition

Source: `run_public_workload_heterogeneity_closeout.py:53-105`. The public
catalog supplies TTFT and ITL; E2E is declared as
`TTFT + mean_output_tokens × ITL`.

| Workload | τ_TTFT | τ_ITL (mean) | τ_E2E | N̂_out (mean out tok) | saturating rate (sim) | eval requests (sim) |
|---|---:|---:|---:|---:|---:|---:|
| interactive | 1,000 ms | 50 ms | 16 s | 300 | 40 rps | 300 |
| reasoning | 2,000 ms | 100 ms | 802 s | 8,000 | 4 rps | 160 |
| deep_research | 10,000 ms | 100 ms | 40 s | 300 | 8 rps | 160 |

The `N̂_out` column is not decoration — it is a **required policy parameter**, not
just the constant used to derive τ_E2E. See
[Output-length priors](#output-length-priors-n̂_out) below.

**Goodput is a per-request hard conjunction**: a request is good only if TTFT ≤
τ_TTFT **and** mean ITL ≤ τ_ITL **and** E2E ≤ τ_E2E. Composite goodput is the
fraction of injected requests that are good; a shed request counts as not good.
Computing this on the real side requires per-chunk timestamps — hence
`--record-itl` is mandatory in the observe block below, not optional.

### Two flag families, one table — only one is plugin config

The campaign feeds this single table to two different consumers
(`run_public_workload_heterogeneity_closeout.py:323-324`):

| | flags | consumer | belongs in |
|---|---|---|---|
| policy targets | `--edpp-tau-ttft` / `-itl` / `-e2e` | the **policy**, inside its score | **EPP plugin `parameters`** |
| goodput targets | `--slo-ttft` / `-itl` / `-e2e` | the **measurement harness** | analysis / `blis calibrate --slo-*` |

They are given identical values but are independent knobs. Do not put the goodput
targets in the plugin config, and do not assume changing one changes the other.

The two are also **not the same conjunction**. The policy's routing value is
TTFT × E2E only — mean ITL was deliberately removed
(`PUBLIC-LOAD-STATIC-BENCHMARK-PROTOCOL.md`: *"after mean ITL was removed from its
routing value"*) — while reported goodput keeps all three. τ_ITL still reaches the
plugin, but for the admission estimator's normalizers (`μ_nom = 1 − α/τ_ITL`), not
as a scoring conjunct. `LIMITATIONS §B2`.

### τ_ITL has a hardware floor

`μ_nom = 1 − α/τ_ITL` requires `τ_ITL > α`. With A100 α = 25,563 µs ≈ 25.6 ms,
**no SLO class may declare τ_ITL below ~26 ms on this fleet.** Interactive's 50 ms
has under 2× margin. Validate this when authoring any new class; the config guard
will reject violations. `LIMITATIONS §E7`.

### Output-length priors (N̂_out)

`Wd(a_r, o)` needs a predicted output length `o`, and every resident's remaining
decode steps come from `nHatFor(r.SLOClass).mean()` — a **per-class** prior. This
is a required plugin parameter that is easy to miss because the campaign never
varied it.

```yaml
      nOutByClass:
        interactive:   300      # gaussian mean, interactive-chat-single-turn.yaml
        reasoning:     8000     # lognormal mean, reasoning-single-turn.yaml
        deepresearch:  300      # gaussian mean, deep-research-single-turn.yaml
      defaultNOut: 300          # used for unclassified traffic — see below
```

It must be a **censored per-class estimate, never the true output length** —
reading the true value makes the arm an oracle and invalidates it.

Two consequences to be aware of before trusting results, both in `LIMITATIONS`:
residents that outlive their class prior are permanently treated as one step from
done (`§C5`), and τ_E2E derived at the mean makes ~50% of requests structurally
infeasible at nominal ITL (`§B1`).

## Request classification

The plugin resolves τ and N̂_out **per request** via `targetsFor(class)` /
`nHatFor(class)`. This is how mixed traffic is meant to work: configure all
classes once, and let each request declare its own.

The campaign did **not** use this. All three workloads declare
`slo_class: standard` and the runner reconfigures the single default τ triple per
run. So the per-class path is real and correct by inspection but has zero
empirical validation (`LIMITATIONS §D1`).

**CONFIRM — pick one before bootstrap:**

1. **Per-workload configs (campaign-faithful).** One EPP config per (arm,
   workload); each declares a single class. 5 arms × 3 workloads = 15 manifest
   entries, 4 images. Reproduces the frozen protocol exactly. Not deployable —
   the EPP must be redeployed to change workload.
2. **Per-class config (deployable).** One config per arm declaring all three
   classes; requests carry a class tag. 5 entries. This is the design that answers
   "how does the EPP know the workload" — it doesn't need to. But it runs an
   unvalidated path and makes the result non-comparable to the frozen numbers
   without re-running the sim side the same way.

Recommendation: **(1) for the reproduction, (2) as a follow-on campaign** with its
own protocol (`LIMITATIONS §D2`).

If (2): the class tag needs a carrier. `InferenceObjective` **cannot** be it —
`InferenceObjectiveSpec` has only `Priority *int32` and `PoolRef`
(`apix/v1alpha2/inferenceobjective_types.go:58`). A request header plus a data
producer is the shape (`predictive_routing` used `session-id-producer` with
`headerName: x-session-id` for the analogous problem). **TRANSLATE — unresolved:**
(a) whether a producer exists for an arbitrary header on the pinned checkout, and
(b) whether `blis observe` can inject a per-workload header. The workload specs
already carry distinct `tenant_id` values (`interactive-users`, `reasoning-jobs`,
`research-agents`) where `slo_class` is uniform, so if `tenant_id` propagates as a
header the discriminator is free.

**Unclassified traffic must not silently default.** `targetsFor` falls back to
`cfg.TauTTFTUs` with no error and no warning, and so does `nHatFor`. An untagged
reasoning request treated as interactive gets a 16 s deadline and a 300-token work
estimate against a real 8,000-token generation — `Wd` underestimates by ~26×. The
plugin **must reject or loudly log** an unknown class rather than defaulting.
`LIMITATIONS §C4`.

The simulator additionally disables the request timeout for these campaigns
(`run_public_load_static_benchmark.py:59` sets `REQUEST_TIMEOUT_SECS = -1`) so a
generic 300 s timeout cannot silently tighten reasoning's 802 s target. The real
harness must likewise not impose a timeout below 802 s.

**CONFIRM — request counts.** 160–300 requests at these rates is 4–40 seconds of
traffic. That is adequate for a deterministic simulator and far too short for
stable real-cluster percentiles or for the fleet to reach steady state. Scale by
roughly 10× and re-derive; record the chosen counts here before any arm runs.

## Load normalization

Loads are **not absolute**. Each workload/fleet has a measured fixed-plan capacity
`C_w`, and evaluation happens at `{0.60, 0.80, 0.95} × C_w`
(`run_public_load_static_benchmark.py:35-38`). Real capacity will differ from
simulated capacity, so the real experiment needs its own probe **before** any
comparison arm runs:

1. Sweep the fixed joint plan grid at a saturating offered rate with loose SLOs
   (`φ ∈ {0, 0.2, …, 1}` remote-prefill fraction; `ψ ∈ {0, 0.2, …, 1}` A100
   decode share, `ψ = 0.5` on the homogeneous control).
2. A plan is capacity-eligible only if every seed completes with zero drops.
   `C_w` = max over plans of mean completion RPS.
3. Freeze `C_w` and the three rates in this file. Do not re-derive them after
   seeing any arm's goodput — the sim protocol treats that as invalidating.

The 18 cells are `3 workloads × 2 fleets × 3 loads`.

## llm-d EPP Configuration — baseline arm (`llmdthreshold`)

Stock plugins only, no custom code. This is the paper's
`llmd_prefix_threshold_workload_tuned` arm. It is simultaneously the
**decode-first ablation**, because `disagg-profile-handler` chooses decode first
and only then asks the decider about prefill.

Per-workload `nonCachedTokens` thresholds (`LLMD_THRESHOLDS`, closeout runner
line 33): interactive `1024`, reasoning `16`, deep_research `16`. One EPP config
per workload.

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: prefix-based-pd-decider
    name: pd-decider
    parameters:
      nonCachedTokens: 1024        # interactive; 16 for reasoning and deep_research
  - type: disagg-profile-handler
    parameters:
      decodeProfile: decode
      prefillProfile: prefill
      deciderPluginName: pd-decider
  - type: precise-prefix-cache-scorer
  - type: queue-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: decode
    plugins:
      - pluginRef: precise-prefix-cache-scorer
        weight: 2.0                # sim: precise-prefix-cache:2
      - pluginRef: queue-scorer
        weight: 1.0                # sim: queue-depth:1
      - pluginRef: max-score-picker
  - name: prefill
    plugins:
      - pluginRef: precise-prefix-cache-scorer
        weight: 2.0
      - pluginRef: queue-scorer
        weight: 1.0
      - pluginRef: max-score-picker
```

The simulator also sets `--cache-signal-delay 50000` (50 ms) for this arm only,
modeling stale prefix-cache signal. Real llm-d has real staleness; do not try to
inject additional delay.

## llm-d EPP Configuration — treatment arm (`causalext`)

The focal policy. Note that this is **not** a Scorer — the joint argmin requires a
custom ProfileHandler plus Pickers that surface the full ranked candidate list.
See `README.md` §Translation notes item 1.

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: causal-slo-externality-joint-handler
    name: edpp
    parameters:
      V: 8.0                       # --edpp-v 8 (closeout runner V = 8.0)
      kernel: composite            # varKernelComposite: sigmoid TTFT x E2E value
      # SLO targets, µs. Resolved PER REQUEST via targetsFor(class) — see
      # "Request classification". Option (1) declares one class per config;
      # option (2) declares all three here. Interactive shown.
      #
      # Scalars are the DEFAULT (cfg.TauTTFTUs), used when a request's class is
      # absent from sloClasses. Under option (1) that is the intended path; under
      # option (2) hitting the default means an unclassified request — reject it,
      # do not serve it silently (LIMITATIONS §C4).
      tauTtftUs: 1000000
      tauItlUs: 50000              # MUST exceed A100 alpha = 25563 us (§E7)
      tauE2eUs: 16000000
      sloClasses:                  # --edpp-tau-{ttft,itl,e2e}-classes
        interactive:  {tauTtftUs: 1000000, tauItlUs: 50000,  tauE2eUs: 16000000}
      # Per-class predicted output length. REQUIRED — feeds Wd(a_r, o) and every
      # resident's remaining-steps estimate. See "Output-length priors".
      nOutByClass:
        interactive: 300
      defaultNOut: 300
      rejectUnknownClass: true     # do not fall back silently (§C4)
      # Ablation switches (see the ablation arm below). All false in the focal arm.
      noExternality: false         # --edpp-slo-externality-no-externality
      noOwnGood: false             # --edpp-slo-externality-no-own-good
      noCapacity: true             # --edpp-slo-externality-no-capacity  <-- ON in the focal arm
      # Rollout configuration: frozen FCFS snapshot, live (zero-staleness) read.
      tadmEstimator: rollforward   # --edpp-tadm-estimator rollforward
      xferSizeAware: true          # --edpp-c-xfer-size-aware
      chunkTokens: 2048            # must equal vLLM --max-num-batched-tokens
      blockSize: 16                # must equal vLLM --block-size
      # TRANSLATE: measure on the target interconnect. Do NOT inherit a sim value.
      cXferUsPerToken: 0.0         # CONFIRM
      # Latency-law coefficients, keyed by GPU type. Mirrors coeffs_by_gpu in
      # inputs/hetero-realistic-1p2d.yaml. Values from inputs/coeffs-*.json.
      coeffs:
        H100:
          alphaUs: 16613.537554540144
          alphaPUs: 16617.85321583337
          c0UsPerReq: 5.347316038602452
          c1UsPerToken: 0.04761401141756073
          cPfUsPerToken: 6.144687138665833
          cAttnUsPerUnit: 0.00010075247918809842
        A100:
          alphaUs: 25563.819286862163
          alphaPUs: 25568.34953836831
          c0UsPerReq: 5.945331876073271
          c1UsPerToken: 0.07822809856114352
          cPfUsPerToken: 9.794219053944662
          cAttnUsPerUnit: 0.00015977670754642687
  - type: max-score-picker
schedulingProfiles:
  - name: decode
    plugins:
      - pluginRef: edpp
  - name: prefill
    plugins:
      - pluginRef: edpp
```

`noCapacity: true` is not a typo. The focal arm is
`causal_externality_no_capacity_v8` — the occupancy-capacity term was tested and
**abandoned** in the closeout study; the surviving policy is the myopic
causal-externality rule. `V` remains at 8.0 because it still scales the
net-good cost against the (now absent) capacity term's units.

The simulator also forces, for every EDPP arm (`edpp_common()`):
`--snapshot-refresh-interval 0`, `--scheduler fcfs`, `--preemption-policy fcfs`.
Real vLLM's scheduler is not FCFS-with-zero-staleness. That is the primary
declared deviation for this transfer; record the actual scheduler policy and
metric scrape interval here once measured.

## llm-d EPP Configuration — comparator arm (`leastttft`)

Same rollout, same coefficients, objective replaced by the arrival's own
projected TTFT. No externality, no capacity, no drift.

```yaml
  - type: least-ttft-joint-handler
    name: edpp
    parameters:
      # identical rollout/coeffs/chunk/block/xfer block as causalext, plus:
      overlapAware: true           # --edpp-ttft-overlap-aware
      # tau and nOut are still needed even though the objective does not use an
      # SLO value: the admission estimator's normalizers are tau-derived and its
      # remaining-steps model is nOut-derived. Dropping them changes the rollout
      # and breaks the like-for-like comparison with causalext.
      tauTtftUs: 1000000
      tauItlUs: 50000
      sloClasses:   { interactive: {tauTtftUs: 1000000, tauItlUs: 50000} }
      nOutByClass:  { interactive: 300 }
      defaultNOut: 300
      rejectUnknownClass: true
```

The projected TTFT arithmetic must be **bit-identical** to `causalext`'s
`tHatLocal`/`tHatDisagg` (BLIS asserts this as INV-6). That identity is what makes
the paired delta attributable to the externality term alone, so the two arms must
share one implementation and one parameter block rather than two copies.

## llm-d EPP Configuration — comparator arm (`kairos`)

Paper-faithful Kairos (`--edpp-rule kairos-paper`). Distinct mechanism: it
deflects prefill *onto a decode node* with a chunk schedule sized to protect
resident TBT, rather than choosing among prefill instances.

```yaml
  - type: kairos-paper-handler
    name: kairos
    parameters:
      alpha: 1.3                   # --kairos-alpha 1.3
      beta: 1.0                    # --kairos-beta 1.0
      chunkCandidates: [2048, 1024, 512, 256, 128]   # TRANSLATE: confirm against
                                   # kairosPaperChunkCandidates in sim/edpp_kairos.go
      chunkCap: 2048               # = vLLM --max-num-batched-tokens
      tauTtftUs: 1000000           # the arriving request's TTFT gate
      # Resident TBT targets. Kairos takes the STRICTEST among a decode node's
      # current residents, NOT the arriving request's class — the constraint
      # protects incumbents. Under mixed classes this makes Kairos
      # conservative-minimax where causalext is value-maximizing, which confounds
      # the comparison (LIMITATIONS §D4).
      tauItlUsByClass: { interactive: 50000 }
      defaultTauItlUs: 50000
      coeffs: { ... }              # same per-GPU block as causalext
```

Kairos needs no `nOutByClass`: Algorithm 1 reasons only about the prompt, never
about output length.

## llm-d EPP Configuration — ablation arm (`causalextnoext`)

Identical image and config to `causalext` with one flag flipped:

```yaml
      noExternality: true          # --edpp-slo-externality-no-externality
```

Sim result for the corresponding ablation
(`sim_results/ablation/ablation_result.json`): full minus own-only `+0.0154`,
full minus decode-first `+0.0485`, full minus resident-only `−0.0016` with a
confidence interval crossing zero. The resident-only term is the one the paper
could not distinguish from zero — worth re-testing on real traffic, which is
burstier.

## Real-Cluster Load Generator (blis observe)

```bash
blis observe \
  --server-url http://<gateway>:80 \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --workload-spec <workload>.yaml \
  --max-concurrency 10000 \
  --prewarm-duration 60s \
  --warmup-requests 50 \
  --timeout 3600 \
  --record-itl \
  --saturation-report saturation.json
```

Notes:

- The three workload specs are BLIS v2 and carry `aggregate_rate` plus
  `arrival: {process: poisson}`, so observe drives **open-loop Poisson** directly
  from the spec. No `--session-mode` / concurrency flags: these are independent
  single-turn requests, not sessions.
- `--record-itl` is **required**, not tuning. Mean ITL is a goodput conjunct;
  without per-chunk timestamps `first_chunk_time == E2E` and the metric cannot be
  computed at all.
- `--timeout 3600` must stay above reasoning's 802 s E2E target.
- `aggregate_rate` in each spec is overwritten per run with the frozen
  `{0.60, 0.80, 0.95} × C_w` rate; `num_requests` likewise. Mirrors
  `make_spec()` in the closeout runner.

## BLIS Simulation Flags (for the sim-vs-real comparison)

Reproduces the campaign's simulator side. From `topology_args()`,
`edpp_common()`, and `policy_args()` in the campaign runners.

| Flag | Value | Notes |
|------|-------|-------|
| `--model` | meta-llama/llama-3.3-70b-instruct | → H100 / TP 4 via `defaults.yaml` |
| `--num-instances` | 3 | 1P2D |
| `--prefill-instances` | 1 | |
| `--decode-instances` | 2 | |
| `--max-num-running-reqs` | 256 | = vLLM `max_num_seqs` |
| `--policy-config` | `inputs/hetero-realistic-1p2d.yaml` | heterogeneous fleet only; supplies `hw_config_by_gpu` + `coeffs_by_gpu` |
| `--decode-routing-scorers` | `queue-depth:1` | all arms except `llmdthreshold` |
| `--pd-decider` | `edpp` | `causalext`, `leastttft`, `kairos` |
| `--snapshot-refresh-interval` | 0 | live snapshot; the rollout needs a non-stale FCFS view |
| `--scheduler` / `--preemption-policy` | `fcfs` / `fcfs` | matches the validated estimator |
| `--edpp-coeffs` | `inputs/coeffs-llama70b-h100-tp4.json` | see [Latency-law coefficients](#latency-law-coefficients) |
| `--edpp-tadm-estimator` | `rollforward` | |
| `--edpp-c-xfer-size-aware` | on | |
| `--edpp-joint-slo-externality` | on | focal arm |
| `--edpp-slo-externality-no-capacity` | on | focal arm |
| `--edpp-v` | 8 | |
| `--edpp-tau-ttft` / `--edpp-tau-itl` / `--edpp-tau-e2e` | per workload | see SLO table |
| `--slo-ttft` / `--slo-itl` / `--slo-e2e` | per workload | goodput accounting, separate from the policy's τ |
| `--timeout` | -1 | disabled; see SLO section |
| `--edpp-joint-candidate-trace` | path | per-candidate score trace; enables the argmin identity check |

## Sim-vs-real comparison

`blis calibrate` compares a real run against the simulator on the same workload.
It cannot fit the coefficients (`LIMITATIONS §A6`) but it is the instrument that
can **falsify** them: large TTFT/E2E MAPE means the frozen coefficients do not
describe this cluster.

```bash
blis calibrate \
  --trace-header <observe>/trace_header.yaml \
  --trace-data   <observe>/trace_data.csv \
  --itl-data     <observe>/itl.csv \
  --sim-results  <replay>/results.json \
  --slo-ttft standard=1000ms \
  --slo-itl  standard=50ms \
  --slo-e2e  standard=16000ms \
  --report calibration.json
```

With `--itl-data` present and all three `--slo-*` set to the workload's targets,
the report carries **real-vs-simulated goodput on the study's own headline
metric**, per cell — turning the comparison into one number rather than a
hand-reading of two reports.

Two cautions:

- It requires a `blis replay` step (replaying the observed trace through the
  simulator) to produce `--sim-results`. **Unverified** whether the sim2real
  pipeline runs one; check before relying on this.
- `--slo-itl` is *silently skipped with a warning* when ITL data is missing on
  either side. A misconfigured run then reports two-way goodput and looks better
  than it is. Assert the ITL conjunct is present in every report.

## Seeds

Held-out confirmation seeds: `2000000011`, `2000000033`, `2000000063`,
`2000000087` (`run_public_load_static_benchmark.py:31`). Development seeds — used
for capacity measurement and static calibration only, never for confirmation —
are `42`, `123`, `2024`.

The sim protocol forbids inspecting confirmation outcomes before all
capacity/rate selections are frozen. Real-cluster runs are not deterministic, so
seeds control workload generation only; **CONFIRM** how many repeats per cell the
real budget allows (the sim used 4).

## Validity gates

Carried over from the sim protocol. Every run must satisfy:

```text
completed + dropped = injected
still_queued = 0
still_running = 0
timed_out = 0
length_capped = 0
```

The confirmation reference reports zero hard-invalid runs, zero drops, zero
timeouts, zero length caps, and `chosen_argmin_trace_exact: true` across 432
runs. On a real cluster, drops and timeouts will occur; count them as not-good
rather than discarding them, and report them separately.

## Commits

| Component | Commit/Version | Notes |
|-----------|---------------|-------|
| BLIS fork | `infocom-implementation` @ `871b169b` | `sim2real`'s `inference-sim` submodule; superproject index still records `583f7195` |
| Upstream BLIS merge base | `3340de78` | |
| llm-d-router | **CONFIRM** — latest release carrying `disagg-profile-handler`, the `deciderPlugin` interface, and a Picker able to return ranked candidates | bootstrap Task 1 pins this as a submodule; v0.9.0 has all three |
| llm-d-router custom image | `ghcr.io/<you>/llm-d-router:<tag>` | built by sim2real from the translated plugin sources |
| vLLM | **CONFIRM** | the sibling bundles pin `docker.io/vllm/vllm-openai:v0.25.1` over the benchmark default |
| Sim reference results | `sim_results/`, 16 files | verified against the branch `CHECKSUMS.sha256` |
