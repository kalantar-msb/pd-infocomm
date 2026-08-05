# Declared limitations

Limitations of this transfer, and of the simulation results it is built on. Every
entry states what was verified, where, and what follows from it. Nothing here is
a defect in the routing policy's logic — these are properties of the predictor,
the metric, the available signals, and the platform.

Written against the BLIS fork at `871b169b` and llm-d-router v0.9.0. Read with
`config.md` (what the plugin needs) and `README.md` (scope and arms).

Categories: [A. Predictor](#a-the-predictor) · [B. Metric](#b-the-metric) ·
[C. Signals](#c-signal-fidelity) · [D. Untested paths](#d-untested-code-paths) ·
[E. Platform](#e-platform-gaps) · [F. Scope](#f-scope)

---

## A. The predictor

### A1. The coefficients were derived from the simulator, not from hardware

`repro_theta_by_gpu.sh` states it in its header: *"fit from the simulator's OWN
trained-physics execution at that HWConfig."* The chain is: a human writes
spec-sheet numbers into a synthetic `hw_config_by_gpu` bundle → BLIS runs seven
deterministic simulated sweeps → `BLIS_STEP_CSV` taps its per-step latencies →
`fit_coeffs.py` does one OLS → six frozen numbers.

No real hardware appears anywhere in that chain. `mfu_prefill` and `mfu_decode`
are hardcoded to `0.5` in the script's heredoc, **identically for both GPU
types**.

Verification that the numbers are spec-sheet arithmetic rather than measurement:

| A100/H100 ratio | value |
|---|---:|
| `c1` (decode KV read per token) | 1.642964 |
| peak HBM bandwidth, 3.35 / 2.039 | 1.642962 |

Identical to six significant figures.

**Consequence.** The coefficients are a compressed model of BLIS's physics, not
of your cluster. Shipping them is still the defensible default — it deploys
exactly the policy the paper evaluated — but predictive accuracy on real
hardware is unknown and cannot be assumed.

### A2. `a100real` does not mean measured on an A100

The label distinguishes real A100 *spec numbers* from the sibling files
`coeffs-llama70b-a100crippled-tp4.json` and the `ratio1p0…ratio5p0` family, which
are the same script run with invented hardware. It does not mean measured. A
reader encountering the filename cold will assume otherwise.

### A3. Compute never enters the coefficients

| A100/H100 ratio | value | |
|---|---:|---|
| `c_pf` (prefill per token) | 1.593933 | ≈ bandwidth ratio |
| `c_attn` | 1.585834 | ≈ bandwidth ratio |
| peak FLOPS, 1979 / 624 | 3.171474 | **not reflected anywhere** |

The fitted operating points sat in the memory-bound regime of BLIS's
`max(compute, memory)` law, so the A100's 3.2× weaker compute is invisible to the
policy. This matters most for **deep research**, whose 45K–121K token inputs are
the case most likely to be genuinely compute-bound on real hardware — where the
true A100 prefill penalty could approach 3× rather than 1.6×, and the policy
would systematically under-price remote prefill on the A100.

*Status: the `c1`-equals-bandwidth match is verified; the prefill-regime
explanation is inferred and has not been confirmed against BLIS's latency-model
source.*

### A4. The A100 coefficients were never validated for additivity

| | H100 | A100 |
|---|---|---|
| produced by | `repro_llama70b.sh` | `repro_theta_by_gpu.sh` |
| `--validate` passed | yes | **no** |
| `validation` block in JSON | present | **absent** |

`repro_theta_by_gpu.sh` never passes `--validate`, so the A100 file has no
regime-stratified MAPE — the check that exists specifically to answer whether
frozen additive coefficients are safe to feed the policy. The A100 set carries the
entire heterogeneity signal the headline result rests on.

**Cheap to close:** re-run `fit_coeffs.py` on the A100 CSVs with `--validate`
against three mixed-regime runs. No GPU required.

### A5. The evaluation is circular by design

`repro_theta_by_gpu.sh`: *"fitting them from the same trained-physics engine that
will EXECUTE the device removes the model/hardware confound."*

This is a deliberate and defensible choice for a policy-comparison paper — it
isolates policy quality from predictor error. But it means the simulation results
contain **no information about robustness to predictor error**, and predictor
error is precisely what transfer to real hardware introduces.

**Suggested mitigation, cheap:** run the focal arm twice, once with the frozen
coefficients and once with the A100 set replaced by the H100 set (making the
policy hardware-blind, everything else identical). The delta separates
hardware-awareness from the externality term.

### A6. There is no way to refit from real data

`blis calibrate` sounds like the tool but is not: it *compares* observed vs
simulated TTFT/E2E and reports MAPE, Pearson r, and percentiles
(`cmd/calibrate.go:41`). Fitting would need per-step
`(t_iter, B_dec, KV, S_pf, pf_ctx)` tuples, which vLLM does not export.

What *is* available: `blis calibrate` can falsify the coefficients by measuring
sim-vs-real discrepancy end to end. See `config.md` §"Sim-vs-real comparison".

### A7. The policy cannot self-correct

Verified: the coefficient fields are never assigned after `LoadEDPPCoeffs`,
`coeffsFor()` is a map lookup, and there are no fitting primitives anywhere in
the decider. Coefficients are read-only constants for the lifetime of the run.

If they are wrong for the cluster, the policy is equally wrong on request 1 and
request 100,000. There is no feedback path.

**Note a visible opportunity:** the EPP already observes per-token timings for
every request it routed, so it could regress observed ITL against `(B_dec, KV)`
from live traffic and recover α, c0, c1 continuously. That is arguably the more
deployable algorithm — but it is *not* the paper's policy and must not be
smuggled into the reproduction. Propose it as its own arm. The prefill side
cannot be recovered this way (`c_pf`/`c_attn` need per-step prefill timings, and
`S_pf` has no metric at all — see C1).

---

## B. The metric

### B1. τ_E2E is calibrated at the mean, making ~half of all requests infeasible

The campaign derives `τ_E2E = τ_TTFT + mean_output_tokens × τ_ITL`. A request is
E2E-feasible at the nominal ITL iff

```text
o ≤ (τ_E2E − τ_TTFT) / τ_ITL  =  mean_output_tokens
```

The critical output length **is exactly the distribution mean, by construction**.
Against the frozen output distributions:

| workload | o_crit | output distribution | P(infeasible at nominal ITL) |
|---|---:|---|---:|
| interactive | 300 | gaussian(μ=300, σ=150) | **50.0%** |
| deep_research | 300 | gaussian(μ=300, **σ=850**) | **50.0%** |
| reasoning | 8,000 | lognormal(8.7337, 0.7120) | **36.1%** |

**Consequence.** Reported goodput of 0.85–0.92 is achieved almost entirely out of
**ITL headroom** — real ITL running well under 50/100 ms — not because output
lengths sit near the mean. Goodput in this study is therefore substantially a
measure of how much slack the fleet has, and the E2E conjunct binds hardest on
exactly the long generations the router can least predict. Note `deep_research`'s
σ=850 against μ=300: that distribution's variance is its dominant term, and it is
the weakest workload in the results (0.786 at 0.95·C) — consistent with the
metric, not the router, being the binding constraint.

**Possible fix, deferred:** compute `τ_E2E = τ_TTFT + N̂_out × τ_ITL` *per
request*, using the same formula the campaign used to derive the constants. The
policy already reasons per-request about work (`Wd(a_r, o)` takes the request's
own predicted output length); only the deadline is a per-class constant. This
would remove the structural infeasibility and eliminate per-workload
configuration. Two costs: results would no longer be comparable to the frozen sim
numbers without re-running the sim side identically, and goodput would inherit
the `N̂_out` predictor's error — the coupling the fixed targets deliberately
avoid. Recommended as a **secondary analysis**, not a replacement.

### B2. The policy's objective and the reported metric are different conjunctions

| | conjuncts | consumer |
|---|---|---|
| routing value | TTFT × E2E | the policy's score |
| reported goodput | TTFT × mean ITL × E2E | the scorekeeper |

`PUBLIC-LOAD-STATIC-BENCHMARK-PROTOCOL.md`: *"after mean ITL was removed from its
routing value."* τ_ITL still reaches the plugin, but for the admission
estimator's normalizers (`μ_nom = 1 − α/τ_ITL`), not as a scoring conjunct. This
is intentional. Do not "fix" it by adding ITL to `compositeValue()`.

---

## C. Signal fidelity

The rollout needs per-resident state that BLIS reads from its own event queue.
`datalayer.Metrics` exposes only aggregates: `RunningRequestsSize`,
`WaitingQueueSize`, `KVCacheUsagePercent`, `KvCacheMaxTokenCapacity`,
`CacheBlockSize`, `CacheNumBlocks`. The EPP must maintain a shadow resident table
from the request lifecycle it observes. Where that is insufficient:

### C1. `S_pf` (resident prefill tokens) has no signal at all

vLLM exports no metric for tokens currently mid-prefill. The shadow table can
approximate it from residents that have not yet reached first token, but on the
**dedicated prefill pool** `tIterPrefill(S_pf)` degrades toward α_p and the
policy under-prices prefill-pool contention.

BLIS calls the decode-side asymmetry dominant (`sim/edpp_var.go`,
`varPrefillDisagg`: *"first-order contention model … the decode-side asymmetry is
the dominant mechanism"*), so this is the least damaging place to degrade — but
expect a **higher remote fraction on the real cluster than the sim's**, and report
it.

### C2. KV occupancy is not Σ resident context

BLIS's `KV` is the exact summed resident `ProgressIndex`. `KVCacheUsagePercent`
reports cache *occupancy*, which also counts prefix-cache blocks retained for
finished requests — an over-count on a prefix-caching deployment, inflating
`C1·KV` uniformly. The shadow-table sum under-counts instead (it misses residents
this EPP replica did not place). The ports prefer the shadow sum so the error has
a known direction.

**Quantify once on the target cluster:** drive a known batch, compare Σ shadow
context against `KVCacheUsagePercent × KvCacheMaxTokenCapacity`, record the ratio.

### C3. Real vLLM is not FCFS with a zero-staleness snapshot

Every EDPP arm forces `--snapshot-refresh-interval 0`, `--scheduler fcfs`,
`--preemption-policy fcfs` (`edpp_common()`), because the validated rollout
consumes an arrival-time FCFS snapshot. Real vLLM uses continuous batching that
admits eagerly when KV allows (true wait often shorter) and preempts/reorders
(sometimes longer), and metrics are scraped on an interval rather than read live.

This is the **primary declared deviation** for the transfer. Record the actual
scheduler policy and scrape interval in `config.md`.

### C4. Unclassified traffic silently gets the default τ and N̂_out

```go
tauTTFTUs, tauITLUs = d.cfg.TauTTFTUs, d.cfg.TauITLUs
if v, ok := d.cfg.TauTTFTByClassUs[class]; ok { tauTTFTUs = v }
```

An unknown or absent class tag falls back with no error and no warning — for both
τ and `nHatFor(class)`. An untagged reasoning request treated as interactive gets
a 16 s deadline and a 300-token work estimate against a real 8,000-token
generation: `Wd` underestimates its decode demand by ~26×. The same fallback
applies to residents via `varSLOFor(r.SLOClass)`.

**Requirement:** the plugin must reject or loudly log unknown classes rather than
defaulting. See `config.md` §"Request classification".

### C5. Residents outliving their class prior become invisible

```go
rem = int64(math.Max(math.Max(nHat, float64(r.StepsDone))-float64(r.StepsDone), 1))
```

When `StepsDone > nHat`, this evaluates to `max(0, 1) = 1`. **Any resident that
outlives its class's mean output length is treated as one step from completion,
permanently.** The externality term then prices it at nearly zero and the policy
piles work onto the instance holding it.

Since `N̂_out` is the class mean, the affected fraction is the mass above the mean
— ~50% for interactive and deep_research, ~36% for reasoning (same figures as
B1). The term systematically under-prices exactly the long-running residents that
do the most co-resident damage.

**This affected the published results.** `jointSLOExternalityCandidateScore` sets
`extDecider.varDeployable = true` explicitly, so the censored path with this floor
is what produced the reported 0.92/0.90 goodputs — under single-class traffic
matching its own generating distribution exactly. Real traffic will match its
declared prior less well.

**Instrument it:** log the fraction of residents pinned at `rem = 1` per
decision. One counter, turns an argument into a measurement.

### C6. Arrival-side estimate error partly cancels; resident-side error does not

`N̂_out` for the arrival is the same across all candidates, so it shifts absolute
scores more than the ranking. Resident `rem` errors do **not** cancel — each
instance holds a different resident population, so biased values corrupt the
comparison *between* instances, and that comparison is the entire decision. The
resident-side estimate is the one that matters, and it is the one with the hard
floor (C5).

---

## D. Untested code paths

### D1. Multi-class traffic was never exercised

All three workloads declare `slo_class: standard`. Every one of the 432
confirmation runs used a single class, reconfiguring the default τ triple per run
rather than varying class. The `--edpp-tau-*-classes` flags are guarded by
`if len(targets) > 1`, which never fired.

The per-class machinery is real and correct by inspection — each co-resident is
priced against its own class (`slo: d.varSLOFor(r.SLOClass)`, `rem` from
`d.nHatFor(r.SLOClass)`) — but it has no empirical validation. Running mixed
classes means running a path the simulation never validated.

### D2. Mixed classes are a new experiment, not this reproduction

Mixed traffic is arguably where this policy *should* shine: the externality term
is only an interesting signal when residents differ, and with one uniform class it
degenerates toward a load count. But it needs its own protocol, its own capacity
selection, and its own held-out seeds. Do not fold it into the reproduction.

### D3. There is no fairness or priority term

The objective is `V·(externality − ownGood)` — a sum of value across residents. It
will starve a loose class to protect a tight one, or the reverse, purely by which
yields more total sigmoid value. No per-class guarantee exists.

And llm-d's priority mechanism is a **separate subsystem this policy never
consults**: `InferenceObjectiveSpec` carries only `Priority *int32` and `PoolRef`
(`apix/v1alpha2/inferenceobjective_types.go:58`). Flow control admits by
priority, then the router places ignoring it — priority inversion is possible.

### D4. Kairos handles mixed classes differently, confounding the comparison

`kairosResidentTBTTarget` takes the **strictest** TBT among residents; the header
notes *"Paper mode also uses the strictest TBT target among current decode
residents when SLO classes differ."* So under mixed traffic the focal arm is
value-maximizing while the comparator is conservative-minimax — a difference
unrelated to joint-vs-decode-first placement. Under mixed classes the arms differ
for two reasons at once and the result is not attributable.

---

## E. Platform gaps

### E1. A mixed-accelerator decode pool is not expressible

The llm-d-benchmark scenario schema has one `decode:` block with one
`acceleratorType`. The heterogeneous fleet — H100 prefill, one H100 decoder, one
A100 decoder — needs either two decode model-services joined to one InferencePool
or a node-affinity split, plus per-pod GPU type visible to the EPP.

**This is a blocking prerequisite, not an extension.** Without it the experiment
reduces to the homogeneous control, where simulation predicts the focal policy
*loses* to stock llm-d (F1).

### E2. Per-pod GPU type is not currently available to a plugin

`coeffsFor(gpuType)` is the load-bearing signal for the entire experiment. It
needs node label `nvidia.com/gpu.product` surfaced as an endpoint attribute.
`plugin.Handle` exposes `PodList() []types.NamespacedName` — names only, no
metrics and no labels. Whether the pinned checkout can surface node labels to a
plugin is **unresolved**. If it cannot, the mixed fleet is not runnable.

### E3. The joint argmin needs a custom ProfileHandler and Picker

`pd_profile_handler.go:186` calls
`decider.disaggregate(ctx, request, profileResults[decodeProfile].TargetEndpoints[0])`
— decode is chosen first, then prefill is conditionally added. Stock llm-d is
structurally **decode-first**, which is exactly the decomposition the ablation
beats by +0.0485 goodput. A Scorer cannot express a joint argmin over the cross
product. Whether a Picker may return the full ranked candidate list, letting a
custom handler select the pair in `ProcessResults`, is **unresolved**.

### E4. Kairos's chunk schedule cannot be executed

Algorithm 1 requires the engine to run the computed per-request chunk schedule;
BLIS honours it via `req.PrefillChunkSchedule`. Real vLLM has no per-request chunk
control — chunked prefill is global via `--max-num-batched-tokens`. The schedule
can only be used to *decide*.

The comparator is therefore handicapped relative to its paper, in the direction
that flatters us. **Do not report the result as a refutation of Kairos.**

### E5. The manifest has no workload dimension

`manifest.py` allows exactly `name`, `source`, `defaults`, `byo` per algorithm
entry, with globally unique names. Since τ and the llm-d thresholds are
per-workload, 5 arms × 3 workloads means 15 entries with near-duplicate configs.
Four images cover all 15 (the overlay differs, the image does not).

### E6. A replicated EPP splits the shadow resident table

Each replica would see only the requests it placed and systematically under-price
contention. Run **one** EPP replica for the first experiment, or move the table to
shared state before scaling out.

### E7. τ_ITL has a hardware floor

`muDNom` computes `1 − α/τ_ITL` and its comment notes *"Caller guarantees
τ_itl > AlphaD (config guard)."* A100 α = 25,563 µs ≈ 25.6 ms, so **no class may
declare τ_ITL below ~26 ms on this fleet.** Interactive's 50 ms has under 2×
margin. A genuinely latency-tight class is unrepresentable on A100 hardware.

---

## F. Scope

### F1. On the homogeneous fleet the focal policy does not win

Computed from `sim_results/main/confirmation_result.json`, 9 cells per fleet:

| policy | homo worst regret | homo mean | hetero worst regret | hetero mean |
|---|---:|---:|---:|---:|
| causal externality | 0.0100 | 0.9372 | **0.0031** | **0.9045** |
| workload-tuned llm-d | **0.0047** | **0.9383** | 0.3517 | 0.7977 |

The entire advantage is heterogeneous, concentrated in interactive chat at load
(0.9058 vs 0.5542 at 0.80·C). A homogeneous-only transfer tests the one
configuration where simulation predicts no edge.

### F2. A tuned static plan still beats the online policy

Paired delta focal-minus-comparator across all 18 cells, 95% CI:

| vs | mean Δ | CI |
|---|---:|---|
| goodput-tuned static plan | **−0.0043** | [−0.0067, −0.0020] |

This is a *static-plan gap*, not policy regret — the static plan knows the
workload, fleet, and offered rate in advance and is not deployable. Report it as
such; never call it an oracle gap.

### F3. The counterfactual campaign is not transferable

144 sampled decisions with 432 forced one-request deviations plus replay gates
requires re-running the same request against a forced alternative from identical
state. Not reproducible on a live cluster. Vendored under
`sim_results/counterfactual/` as sim-side evidence only.

### F4. Request counts are far too small for real percentiles

160–300 requests at the campaign rates is 4–40 seconds of traffic — adequate for
a deterministic simulator, inadequate for steady state or stable percentiles on
hardware. Scale ~10× and re-derive; record the chosen counts in `config.md`.

### F5. Capacity must be re-measured

Loads are `{0.60, 0.80, 0.95} × C_w` where `C_w` is *measured* fixed-plan
capacity. Real capacity differs from simulated capacity, so the real experiment
needs its own probe before any comparison arm runs, and the rates must be frozen
before any arm's goodput is inspected.

### F6. The sim2real submodule pin is behind

`sim2real`'s superproject index records `583f719507f31c24dd675d30daa79c33950c5516`
for `inference-sim` while the working checkout is at `871b169b` — the commit this
bundle's artifacts came from. Commit the submodule bump or the provenance chain
does not close.
