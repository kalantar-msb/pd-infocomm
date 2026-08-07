# Signal mapping: what we will use

The decided mapping from BLIS signals to llm-d values for the **causal SLO
externality** arm. One row per signal, one route per row, no options.

`docs/required-signals.md` is the analysis that produced these choices — the
alternatives, the impact numbers, and the evidence live there. This document is
the answer, not the reasoning.

Pins: BLIS `871b169b` · llm-d-router `v0.9.0` · vLLM `v0.26.0`.

**Nothing in sections 1-5 requires a change to vLLM or to llm-d-router.** Section 7
lists what a later vLLM patch would improve. It is not needed to run.

Scope: single SLO class, one EPP replica, focal arm only. Times are microseconds.

How to read the tables:

| Column | Meaning |
|---|---|
| **BLIS signal** | the name in the simulator |
| **Route** | where the value comes from |
| **How** | the exact accessor or formula |

---

## 1. Read directly off the router

These arrive on `datalayer.Metrics`, scraped from vLLM by the default extractor.
No config, no computation.

| BLIS signal | Route | How |
|---|---|---|
| `BatchSize` | `vllm:num_requests_running` | `Metrics.RunningRequestsSize` |
| `QueueDepth` | `vllm:num_requests_waiting` | `Metrics.WaitingQueueSize` |
| KV block size | `vllm:cache_config_info` | `Metrics.CacheBlockSize` |
| KV total blocks | `vllm:cache_config_info` | `Metrics.CacheNumBlocks` |
| KV usage fraction | `vllm:kv_cache_usage_perc` | `Metrics.KVCacheUsagePercent` |

**Do not call this `B_dec`.** The symbol in the latency law
`T_iter = alpha + C0*B_dec + C1*KV + C_pf*S_pf` is decode-only, and this value is
not. `vllm:num_requests_running` is `len(self.running)`; BLIS's `BatchSize()` is
`len(sim.RunningBatch.Requests)` (`simulator.go:699`). Both include requests still
prefilling, so the **mapping is exact** — but the quantity is a running total, not
a decode count, and we preserve BLIS's behavior deliberately.

Worth knowing that BLIS is inconsistent with itself here, and we inherit it:

- Its calibration tap records `b_dec` (decode-only) and `batch_size` (running
  total) as **separate CSV columns** (`simulator.go:1224-1244`,
  `step_recorder.go:22`), classifying `ProgressIndex < InputLen` as prefilling.
- `fit_coeffs.py:82` fits `C0` against the **decode-only** `b_dec` column.
  `batch_size` is recorded and never used as a regressor.
- The policy consumes `ds.BatchSize`, the running total (`simulator.go:632`,
  `edpp_kairos.go:329`).

So `C0` is fitted on one definition and consumed against another. Two reasons this
is not worth acting on: the fit is drawn only from rows with `s_pf == 0`, where no
prefilling requests exist and the two definitions coincide; and `C0` is ~5.3-5.9
µs/request, so wrongly counting four prefilling requests costs ~21 µs against a
17,000-27,000 µs iteration — **under 0.1%**. Preserve the BLIS behavior, keep the
name honest.

> **Do not use `Metrics.KvCacheMaxTokenCapacity`.** It looks like the answer for
> KV capacity but is never assigned anywhere in v0.9.0 — it exists only in the
> struct declaration and in `Clone()`. It always reads 0, so any code guarded on
> `KvCacheMaxTokenCapacity > 0` never executes. Use `CacheBlockSize *
> CacheNumBlocks`.

---

## 2. Computed from section 1

| BLIS signal | How | Note |
|---|---|---|
| `KvTokensInUse` (`KV`) | `KVCacheUsagePercent * CacheBlockSize * CacheNumBlocks` | **No division by 100** — unit trap below. Also a **declared approximation**, see §6 |
| Per-resident `KVBlocks` | `ceil((InputLen + TokensStreamed) / CacheBlockSize)` | from the shadow table |

> **Unit trap.** `KVCacheUsagePercent` is a **fraction in [0,1]**, despite the
> name. vLLM documents the gauge as "1 means 100 percent usage", the router passes
> it through unscaled, and the router's own saturation detector compares it to a
> threshold defaulting to `0.8`. Dividing by 100 under-counts KV by 100x and
> collapses the `C1*KV` term that carries hardware heterogeneity — which is the
> experiment. Note the sim2real pipeline's own mapping table states the opposite
> and mandates the `/100`; it is wrong, and the bug has already appeared in
> generated port code once.

We use the metric route rather than summing per-resident context from the shadow
table. The metric is block-granular, so a shared prefix block is counted once in
total — matching what the BLIS policy **consumes** (`UsedBlocks * BlockSize`). The
shadow sum counts a shared prefix once *per resident*, running about +28% high on
`reasoning` and `interactive-chat`. Keep the shadow sum only as a cold-start path
and a cross-check.

This matches BLIS's decision function but **not** the definition its coefficients
were fitted against. That is an accepted approximation, not an exact mapping — see
§6 for what diverges and by how much.

---

## 3. Configuration we set

| BLIS signal | Route | How |
|---|---|---|
| `GPUType` | pod label | `ep.GetMetadata().Labels["pd-infocomm/gpu-product"]` |
| `alpha`, `C0`, `C1`, `C_pf`, `C_attn` | `inputs/coeffs-*.json`, per GPU type | plugin config |
| chunk budget | plugin config | must equal the engine's `--max-num-batched-tokens` |
| `tau_ttft`, `tau_itl`, `tau_e2e` | plugin config | one triple; `sloClasses` left empty |
| `c_xfer` | measured on the target interconnect | **must actually be measured** |

### GPU type

`nvidia.com/gpu.product` is a *node* label and plugins see endpoints, not nodes.
Set your own label on the pod template and pin the `nodeSelector` to the same
value, so a mismatch fails at scheduling time instead of serving traffic under the
wrong physics:

```yaml
# Deployment: decode-h100
spec:
  template:
    metadata:
      labels:
        pd-infocomm/gpu-product: NVIDIA-H100-80GB-HBM3
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
```

```yaml
# EPP plugin config
parameters:
  gpuTypeLabel: pd-infocomm/gpu-product
  gpuTypeByLabel:
    NVIDIA-H100-80GB-HBM3: h100
    NVIDIA-A100-SXM4-80GB: a100
  defaultGpuType: ""      # empty => reject unknown, never default
```

**Fail loudly on an unknown GPU type.** A silent default makes the policy
hardware-blind, which is the exact failure mode it exists to beat.

**Heterogeneity rides `alpha`, not `C1`.** At B=8 and KV=16,384, pure-decode
`T_iter` is 17,437 µs on H100 against 26,893 µs on A100 — a 1.54x ratio that is
almost exactly the intercept ratio (16,614 vs 25,564 = 1.539). `C1·KV` contributes
about 500 µs of that ~9,500 µs gap. `required-signals.md` §5.3 justifies GPU typing
with the `C1` spread (0.0476 vs 0.0782); the conclusion is right but the mechanism
is the intercept. This *strengthens* the case for per-endpoint typing, since the
discriminating term is present on every iteration regardless of KV state.

All decode endpoints must live in **one** `InferencePool`. Two pools means two
EPPs, each blind to the other's endpoints, and no joint argmin is possible — so
per-pool coefficients cannot work.

> **Accessor note.** A pod label is `ep.GetMetadata().Labels[key]`. This is *not*
> the same as the attribute API (`ep.Get(key)`) or the custom-metric API
> (`attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), key)`). The port's
> current `endpointAttr()` marker points at the wrong one.

### SLO class: not needed

Every cell of the registered campaign is single-class (`slo_class: standard`).
Populate `SLO`, leave `sloClasses` empty, and `sloFor()` returns the single triple
for every request. The `SLOClass` field can stay `""` forever. No header, no
gateway change, no data producer, no plugin change.

This is faithful, not a shortcut — the simulator was single-class too.

---

## 4. Per-endpoint request state

| BLIS signal | Route | How |
|---|---|---|
| `a_p` uncached suffix | prefix-cache data producer, **per candidate endpoint** | `(TotalBlocks() - CachedBlockCount()) * BlockSizeTokens()` |

Retrieved inside the loop over candidates, the same way the prefix scorer does it:

```go
info, ok := endpoint.Get(prefixMatchDataKey.String())
if !ok { /* declare degraded; do not read zero as "no cache" */ }
pmi := info.(*attrprefix.PrefixCacheMatchInfo)
apBlocks := pmi.TotalBlocks() - pmi.CachedBlockCount()
ap := maxInt(apBlocks, 0) * pmi.BlockSizeTokens()
```

Two corrections to `required-signals.md` §2, both verified in
`plugins/datalayer/attribute/prefix/data_types.go:27-41`:

- **Use `CachedBlockCount()`, not `MatchBlocks()`.** Under the precise prefix
  cache, `matchBlocks` is a *device-tier-weighted* score (RAM-tier blocks count as
  less than 1.0) meant for relative endpoint ranking. `cachedBlockCount` is the
  literal cached-block count, and the type's own doc says it exists precisely so
  that "consumers that convert blocks to a token count get an accurate cached-token
  figure rather than a tier-attenuated one." Since weighted `matchBlocks <=
  cachedBlockCount`, using it would **over-estimate** `a_p` and over-price prefill
  work. It defaults to `matchBlocks` when unset, so the correct call is safe either
  way.
- **`BlockSizeTokens()` comes from the info, not from `Metrics.CacheBlockSize`.**
  The prefix producer carries its own configurable block size; it is not
  necessarily vLLM's KV block size.

`a_p` is available **per candidate endpoint** — `endpoint.Get(...)` is keyed by
endpoint. BLIS calls `apForInstance(req, id)` per instance, so evaluate it inside
the candidate loop and match that. `required-signals.md` §6.0 lists a single
per-request value as the recommended route; that was a simplification for brevity
and there is no reason to keep it, since the per-endpoint form costs nothing.

---

## 5. Shadow table — state the EPP tracks itself

vLLM exports only aggregates, so per-resident state has to come from the request
lifecycle. The EPP saw every request it placed, so these are as exact as the
simulator's.

| BLIS signal | Populated at | Value |
|---|---|---|
| `ArrivalUs` | `PreRequest` | EPP clock |
| `InputLen` | `PreRequest` | prompt length |
| endpoint | `PreRequest` | from the scheduling result |
| `FirstTokenUs`, `TTFTSet` | `ResponseBody`, first invocation | EPP clock — **not** `ResponseHeader` |
| `StepsDone` | `ResponseBody`, every chunk | `Response.Usage.CompletionTokens` |
| deregister, `N_out` update | `ResponseBody` with `EndOfStream` | censored running mean |

### First-token time comes from the body hook

`ResponseHeader` fires when the model server *begins responding*, before any token
exists. Using it inflates every resident's recorded TTFT accuracy in the wrong
direction. The router itself gets this right in the handler layer
(`handlers/response.go:55`):

```go
if reqCtx.FirstTokenTimestamp.IsZero() && len(responseBytes) > 0 {
    reqCtx.FirstTokenTimestamp = time.Now()
}
```

A plugin **cannot** apply that rule directly: the `Response` struct passed to
requestcontrol plugins carries `RequestID`, `Headers`, `StartOfStream`,
`EndOfStream`, `ReqMetadata`, `Usage`, and `DynamicMetadata` — **no body bytes**,
so there is nothing to test for emptiness. Use one of:

- `Response.StartOfStream` — the first body invocation. Simple, and correct unless
  vLLM emits a leading empty chunk.
- `Response.Usage.CompletionTokens` first reaching >= 1 — exact, and available
  because we already require per-chunk usage for `StepsDone`.

Prefer the usage route, since it is the same signal we already depend on.

> **Timing caveat, declare it.** Non-final chunks run on a per-request background
> goroutine (`director.go:545`, drained by `processResponseBodyQueue`). A
> `time.Now()` taken inside the plugin's `ResponseBody` is the *dequeue* time, not
> the chunk arrival time. Recorded TTFT feeds the TTFT conjunct of `g()` for every
> resident, so this is a real if small bias. Also: the shadow table is written from
> that goroutine and read from the scheduling path, so it needs a mutex.

### Per-chunk usage is required

`Response.Usage.CompletionTokens` only updates when the chunk actually carried
usage. Enable one of:

- `stream_options.continuous_usage_stats` on the request, or
- `--enable-force-include-usage` on the server.

Without either, usage arrives only in the final chunk, `StepsDone` stays 0 for the
whole request, and `remainingSteps()` is wrong for every resident. Fall back to
counting chunks only as a declared degradation.

### Two limits, both accepted

- **One EPP replica.** Replicas split the table — each sees only what it placed.
  Run a single replica for the campaign.
- **Bypassing traffic is invisible.** Use `Metrics.RunningRequestsSize` for
  `BatchSize` (it counts everything) and the shadow table only for per-resident detail.

---

## 6. Approximations we accept

Each is a real deviation from the simulator. The risk is shipping them silently, so
each gets a counter or a log line.

| BLIS signal | What we do | Error direction |
|---|---|---|
| `KvTokensInUse` (`KV`) | block occupancy from the metric, below | over-counts vs the fitted definition when a prefill is in flight |
| `ResidentPrefillTokens` (`S_pf`) | capped shadow sum, below | over-estimate, biases away from local prefill |
| Scheduler rollout | closed-form `rollforward` over the shadow table | under-estimates admission delay at deep queues |
| `collocPrefill` | omitted | under-prices contention on the decode node |
| Prefill-pool externality | degrades toward zero | under-prices remote prefill |

### `KV` — block occupancy, not decoding-request contexts

The coefficient `C1` was fitted against `Σ ProgressIndex` over **decoding requests
only** (`simulator.go:1238`, inside the `else if len(req.OutputTokens) > 0`
branch; `step_recorder.go:56` documents it as "the summed resident decode
context"). Block occupancy is a different quantity. Three divergences, smallest
first:

1. **Null-block factor.** vLLM computes `1 − free/(num_gpu_blocks − 1)`
   (`block_pool.py:805-816`); BLIS divides by `TotalBlocks`. An `N/(N−1)` offset,
   negligible for N in the thousands.
2. **Prefix sharing.** Block-granular occupancy counts a shared prefix block once;
   the fitted per-request sum counts it per resident. Roughly +28% on `reasoning`
   and `interactive-chat` for the shadow route, which is why we do not use it.
3. **Prefilling requests' allocated blocks — the dominant one.** The fitted `kv`
   excludes them entirely; occupancy includes them. A 45k-token prefill in flight
   adds ~45,000 tokens to occupancy that the fitted definition omits:
   `45,000 × 0.0476 = 2,142 µs` at H100 coefficients, about **12% of `T_iter`**.
   It also co-varies with `S_pf`, so that context is charged twice — once through
   `C1·KV` and again through `C_pf·S_pf`.

Why we accept it anyway: BLIS's own consumed value,
`KvTokensInUse = UsedBlocks × BlockSize`, is *also* block-level and *also* includes
prefilling requests' blocks. So the metric route reproduces BLIS's decision
function faithfully, and the fit/consume mismatch is one we **inherit rather than
introduce**. For a transfer study that is the right choice — the decision function
is what transfers. But it is an approximation and must be reported as one.

Divergence 3 is worth measuring once on the target cluster: drive a known batch
with a long prefill in flight and compare `Σ` decoding-resident context against the
occupancy estimate.

> **The port currently does the opposite of this decision.** `kvTokensFor`
> (`causal_slo_externality.go:325`) prefers the shadow sum and guards its metric
> fallback on `m.KvCacheMaxTokenCapacity > 0` — the field that is never assigned,
> so the fallback is unreachable and `KV` reads **0** whenever the shadow table is
> empty. Its comment also claims the metric counts prefix blocks retained for
> finished requests; it does not — `free_blocks()` returns those to the free queue
> with hashes intact (`block_pool.py:719-740`), so `get_usage` excludes them.
> Tracked in §9.

### `S_pf` — cap it

`S_pf` is what is being prefilled **this step**, not the outstanding prompt
backlog. Summing each prefilling resident's whole remaining prompt over-estimates
by roughly `nChunks`, which at 45k input is ~10x:

| Workload | mean input | true `S_pf` | naive sum | `T_iter` inflation |
|---|---:|---:|---:|---:|
| `reasoning` | 1,000 | 1,000 | 1,000 | 1.0x |
| `interactive-chat` | 4,000 | 2,048 | 4,000 | 1.4x |
| `deep-research` | 45,000 | 2,048 | 45,000 | **9.8x** |

So cap at what the scheduler could actually schedule in one step:

```
S_pf(endpoint) = min(
    Σ over residents on endpoint without a first token: min(remaining_prompt, chunk),
    maxNumBatchedTokens
)
```

This removes the bulk of the error at zero signal cost. It is still an
approximation — the EPP cannot know *which* requests the engine scheduled.

Note the error only fires once a prefill is in flight on a decode node, i.e. after
the policy has elected local prefill. On a clean 1P2D fleet `S_pf` is 0 either way.

### Scheduler rollout — substitute and say so

BLIS predicts admission and first token by replaying vLLM's scheduler over the
ordered wait queue with per-request prompt and computed token counts. The EPP has
`WaitingQueueSize`: one integer. There is no mapping.

We use the closed-form `rollforward` estimator over the shadow table. Upstream this
substitution happens **silently** — when the rollout returns `ok == false`, BLIS
keeps the closed form with no error and no log. So:

**Log loudly, once per decision or as a counter, that the rollout path is
unavailable.** The fallback is fine; the silence is the defect. A port that
substitutes quietly is running a different TTFT estimator than the one that
produced the published goodputs, and nothing says so.

Also worth doing before the cluster run: re-run the simulation with the rollout
disabled and report *that* number as the deployable baseline. No cluster and no
code, and it is the honest headline.

---

## 7. Deferred: one vLLM patch, not required to run

Four scalar gauges. All four ride the existing `/metrics` path into endpoint
attributes through `customMetrics` config with **zero llm-d-router code**, read
back with `attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), key)`. Three are
export-only.

| Gauge | vLLM work | Replaces |
|---|---|---|
| `num_ctx_tokens` | export only — already computed | the capped `S_pf` estimate |
| `device_compute_capability` | export only — already computed | trust in the hand-set pod label |
| `scheduler_step_index` | export only — already exists | nothing; enables torn-read detection |
| `num_pending_prefill_tokens` | new O(queue) sum | `PrefillTokensAhead` — **Kairos only** |

`PrefillTokensAhead` does not appear in the focal arm. It only matters when the
Kairos comparator is stood up, and there it is worth more than accuracy: reading
zero holds Kairos's eligibility gate permanently open, turning it into a more
aggressive policy than the published one and inverting the comparator.

The per-request wait-queue contents of section 6 are **not** scalars and are not in
this bundle. Defer them behind the measured cost of the substitution.

---

## 8. The largest gap is not a signal

The coefficients `alpha`, `C0`, `C1`, `C_pf`, `C_attn` were fit against BLIS's own
trained-physics model, not against real vLLM hardware. Every latency projection in
every arm rests on them, so this is a bigger fidelity gap than any individual
signal in this document.

The fit reports confirm it. `inputs/coeffs-*.json` carries `r2` of
`0.9999999983` (H100) and `0.9999999994` (A100), with `cond_b_dec_kv` of 2.50 and
2.38. Real hardware does not fit a three-parameter linear model to ten significant
figures — this is OLS recovering the linear model that generated the rows. Note
`coeffs-llama70b-a100real-tp4.json` is named "a100real" and sources
`/tmp/theta_a100real/*.csv`, but carries the same synthetic signature; the name
should not be read as evidence of hardware measurement.

The instrument for fixing it already exists:
`--enable-logging-iteration-details` emits `(num_ctx_tokens,
num_generation_requests, elapsed_ms)` per iteration — a real-hardware analogue of
BLIS's calibration tap. Harvest it and refit.

This matters for sequencing: comparing predicted against realized completion times
on the cluster mostly measures coefficient misfit, not predictor structure. Refit
before drawing conclusions about the predictor.

---

## 9. Still open

| Item | Decision needed |
|---|---|
| `QueueDepth` and the skipped-waiting queue | vLLM tracks `waiting` **and** `num_skipped_waiting_reqs`; BLIS has one `WaitQ`. If `num_requests_waiting` excludes the skipped queue, `QueueDepth` under-counts requests genuinely ahead of an arrival. Decide once (probably the sum) and declare it |
| `c_xfer` | must be measured on the target interconnect. Inheriting the simulated value is the unfaithful path and is easy to do by accident |
| Joint `(d, p)` placement registration | a disaggregated placement should arguably register the resident on **both** endpoints. If prefill endpoints are never registered, their `S_pf` reads 0, `tIterPrefill` collapses to `alpha_p`, and prefill-pool contention is under-priced by ~1.8x |
| `kvTokensFor` contradicts §2 | the port prefers the shadow sum and its metric fallback is dead code (guarded on the never-assigned `KvCacheMaxTokenCapacity`), so `KV` reads 0 on an empty table. Needs the guard replaced with `CacheBlockSize * CacheNumBlocks`, the preference order settled to match §2, and its prefix-retention comment corrected. **Code change, not a doc change** |

---

## Checklist to a runnable focal arm

- [ ] GPU-product pod label on each decode Deployment, `nodeSelector` pinned to the same value
- [ ] Single-class config: `SLO` populated, `sloClasses` empty
- [ ] Single EPP replica
- [ ] `chunkTokens` config equals the engine's `--max-num-batched-tokens`
- [ ] Per-chunk usage enabled (`continuous_usage_stats` or `--enable-force-include-usage`)
- [ ] `c_xfer` measured, not inherited
- [ ] Shadow table wired to `PreRequest` / `ResponseBody` / `ResponseBody+EndOfStream`, mutex-guarded
- [ ] First-token time from `Usage.CompletionTokens >= 1`, not `ResponseHeader`
- [ ] `S_pf` capped per resident and in total
- [ ] `a_p` per candidate endpoint, via `CachedBlockCount()`
- [ ] No `/100` on `KVCacheUsagePercent`
- [ ] `KV` read from the metric route, not the shadow sum — `kvTokensFor` currently inverts this and its fallback is dead (§9)
- [ ] Rollout-unavailable log or counter firing
- [ ] Unknown GPU type rejected, not defaulted
