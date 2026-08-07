# Required signals: BLIS to llm-d mapping

What each algorithm reads from the simulator, and where that value comes from on
a real cluster. One row per signal.

Pins: BLIS `871b169b` · llm-d-router `v0.9.0` · vLLM `v0.26.0`.

Status legend:

| | meaning |
|---|---|
| **Direct** | an existing llm-d/vLLM value with the same definition |
| **Derived** | computable from existing values |
| **Shadow** | the EPP must track it itself, from request lifecycle hooks |
| **MISSING** | no mapping exists; needs a change to vLLM or the router |

---

## 1. Signals shared by all three algorithms

All three score candidates with the same latency law:

```
T_iter(decode) = alpha_d + C0*B_dec + C1*KV + C_pf*S_pf
T_iter(prefill) = alpha_p + C_pf*S_pf
```

So all three need the same four state variables per candidate instance, plus the
GPU type that selects which coefficient set to use.

| BLIS signal | What it is | Why the algorithm needs it | llm-d mapping | Status |
|---|---|---|---|---|
| `BatchSize` | count of running requests on the instance, **including those still prefilling** | linear term in iteration time: more concurrent requests, slower steps. Note this is *not* the decode-only `B_dec` of the law above — see below | `Metrics.RunningRequestsSize` <- `vllm:num_requests_running` | **Direct** |
| `KvTokensInUse` (`KV`) | block-level KV occupancy in tokens (`UsedBlocks × BlockSize`), **not** Σ per-resident context | dominant term — attention cost grows with total context held | `KVCacheUsagePercent * CacheBlockSize * CacheNumBlocks`, or the shadow-table sum | **Derived** (see 5.1) |
| `ResidentPrefillTokens` (`S_pf`) | tokens being prefilled this step | prefill work stalls decode steps in the same batch | none | **MISSING** (see 5.2) |
| `GPUType` | e.g. `A100-80GB` vs `H100` | picks the coefficients. **This is the experiment.** Pure-decode `T_iter` at B=8, KV=16,384 is 17,437 µs (H100) vs 26,893 µs (A100), ~1.54x. That ratio is carried by the **intercept** (`alpha_d` 16,614 vs 25,564 = 1.539), not by `C1` (0.0476 vs 0.0782), which contributes ~500 µs of the ~9,500 µs gap | pod label in the Deployment template, read from `EndpointMetadata.Labels`. The *node* label `nvidia.com/gpu.product` is not visible to plugins | **Config** (see 5.3) |
| `QueueDepth` | waiting queue length | how many requests are ahead of the arrival | `Metrics.WaitingQueueSize` <- `vllm:num_requests_waiting` | **Direct** |
| `BlockSizeTokens`, capacity | KV block size, total blocks | converts occupancy fraction to tokens | `Metrics.CacheBlockSize`, `CacheNumBlocks` <- `vllm:cache_config_info` | **Direct** |
| `C_pf`, `C_attn` | prefill per-token and causal-attention coefficients | **not** part of `T_iter` above. The focal arm's externality charges the arrival's *trajectory* prefill work on top of the iteration time — `C_pf*processed + C_attn*processed*(cachedPrefix + processed/2)` — so `c_attn_us_per_unit` is load-bearing, not decorative | `inputs/coeffs-*.json`, per GPU type | **Config** |
| chunk budget | per-step prefill token budget | sets `nChunks` for the TTFT projection and `chunk = min(a_p, budget)` for the charge above. Must equal the engine's real value or the overlap window is mispriced | plugin config, = vLLM `--max-num-batched-tokens` | **Config** |

**Definitions confirmed to match on the consume side.** BLIS's `BatchSize` is
`len(RunningBatch.Requests)` including requests still prefilling
(`sim/simulator.go:699`); vLLM's `num_requests_running` is `len(self.running)`
(`scheduler.py:2431`). Same definition, so the mapping needs no correction.

**But that pair is not what `C0` was fitted against, and the symbol name is
wrong.** The calibration tap records `b_dec` and `batch_size` as *separate* CSV
columns (`sim/simulator.go:1224-1244`, `sim/step_recorder.go:22`), classifying
`ProgressIndex < InputLen` as prefilling and counting `b_dec` only in the
`else if len(req.OutputTokens) > 0` branch. `scripts/calibration/fit_coeffs.py:82`
then fits `C0` against the **decode-only** `b_dec`; `batch_size` is recorded and
never used as a regressor. The policy meanwhile consumes `ds.BatchSize`, the
running total (`sim/simulator.go:632`, `sim/edpp_kairos.go:329`).

So BLIS fits `C0` on one definition and consumes it against another — the same
fit/consume split documented for `KV` in §5.1, and we inherit it. Two reasons it
does not warrant action:

- **The fit is clean.** `fit_decode` selects rows with `s_pf == 0`, where no
  prefilling requests exist and the two definitions coincide. The mismatch appears
  only when `C0` is extrapolated to mixed steps at decision time.
- **The term is negligible.** `C0` is 5.35 µs/req (H100) and 5.95 (A100), so
  wrongly counting four prefilling requests costs ~21 µs against a 17,000-27,000 µs
  iteration — under 0.1%. A prefilling request is double-charged (through `C0·B` and
  again through `C_pf·S_pf`), but unmeasurably so.

**Do not write `B_dec` for this signal**, and do not write it for
`num_generation_requests` either (§5.2) — those are different quantities and
conflating them would corrupt any refit. Preserve BLIS's running-total behavior and
name it `BatchSize`.

---

## 2. Focal arm — causal SLO externality

Adds the resident externality term: for every request already on the instance,
how much SLO value does placing this arrival there destroy?

```
score(d,p) = V * ( sum_residents [ g(before) - g(after) ]  -  g(arrival) )
```

That sum needs **per-resident** state, which no metric provides — it is
per-request, and vLLM exports only aggregates.

| BLIS signal | What it is | Why the algorithm needs it | llm-d mapping | Status |
|---|---|---|---|---|
| `RunningReqState.StepsDone` | output tokens the resident has produced | with `N_out`, gives remaining steps -> when it finishes | count streaming chunks; `Response.Usage.CompletionTokens` | **Shadow** (see 5.4) |
| `RunningReqState.ArrivalUs` | when the resident arrived | its E2E deadline is `arrival + tau_e2e` | EPP clock at `PreRequest` | **Shadow** |
| `RunningReqState.FirstTokenUs`, `TTFTSet` | whether it has produced its first token | decides TTFT-side vs ITL-side risk: a resident past first token can no longer miss TTFT | first `ResponseBody` invocation — `StartOfStream`, or `Usage.CompletionTokens >= 1`. **Not `ResponseHeader`** (see 5.4) | **Shadow** |
| `RunningReqState.SLOClass` | its SLO class | resolves its `tau_ttft`/`tau_itl`/`tau_e2e` | inert under single-class; otherwise `req.Headers[x-…-inference-objective]`, no data producer needed | **Not needed** (see 5.5) |
| `RunningReqState.KVBlocks` | its KV footprint | admission estimate: how much KV frees when it leaves | `ceil(context / CacheBlockSize)` from shadow table | **Derived** |
| resident remaining prefill tokens | prompt tokens left to prefill | prefill-pool contention term, per occupant | `promptTokens - tokensPrefilled` from shadow table | **Shadow** |
| same, on the **decode** node (`RunningPrefill`) | occupants mid-prefill where the arrival would decode | BLIS's externality has **three** populations — `decode`, `collocPrefill`, `prefillPool` — and the focal arm enables all three (`varCollocPrefill = true`, `sim/edpp.go:1708,1749`). `collocPrefill` prices the first-token delay this placement inflicts on occupants already prefilling on the decode node. Genuinely 0 on a clean 1P2D, but **not** when the policy elects local prefill | none — same root cause as `S_pf` | **MISSING** (5.2) |
| `N_out` per class | mean output length, running mean over completions | the arrival's decode demand `Wd`, and each resident's remaining steps | EPP maintains the same running mean | **Derived** |
| `a_p` uncached suffix | prompt tokens not already cached on that endpoint | prefill work `Wp` — a cached prefix is free | `PrefixCacheMatchInfo`, **per candidate endpoint**: `(TotalBlocks() - CachedBlockCount()) * BlockSizeTokens()`. **Not `MatchBlocks()`** — see below | **Direct** (block granular) |
| `tau_ttft`, `tau_itl`, `tau_e2e` | per-class SLO targets | the deadlines in `g()` | plugin config | **Config** |
| `c_xfer` | KV transfer cost, remote prefill | price of disaggregating | must be measured on the target interconnect | **Config, unmeasured** |
| ~~SLO virtual queues `z_TTFT`, `z_ITL`~~ | accumulated violation per class | **not used by this arm** — `jointSLOExternalityCandidateScore` has no historical-deficit term; the z-terms belong to the work-currency deciders (`sim/edpp.go:922`, `:1339`) | n/a | **Not needed** (see §6.0) |
| scheduler-rollout state | **ordered** wait queue with per-request prompt/computed tokens | BLIS replays vLLM's scheduler forward to predict admission and first token | none today; exportable as position-indexed gauges consumed via `customMetrics` | **MISSING** (see 5.6) |

#### `a_p`: use `CachedBlockCount()`, and evaluate it per candidate

Two corrections, both from
`plugins/datalayer/attribute/prefix/data_types.go:27-41`:

- **`MatchBlocks()` is the wrong accessor.** Under the precise prefix cache it is a
  *device-tier-weighted* longest-prefix score (RAM-tier blocks count as less than
  1.0), documented as "suitable for relative endpoint ranking".
  `cachedBlockCount` is the literal cached-block count, and its own doc comment
  says it exists so that "consumers that convert blocks to a token count get an
  accurate cached-token figure rather than a tier-attenuated one." Since weighted
  `matchBlocks <= cachedBlockCount`, using it **over-estimates** `a_p` and
  over-prices prefill work. It defaults to `matchBlocks` when unset, so
  `CachedBlockCount()` is safe either way.
- **`BlockSizeTokens()` comes from the info object**, not from
  `Metrics.CacheBlockSize`. The prefix producer carries its own configurable block
  size, which need not equal vLLM's KV block size.

`endpoint.Get(...)` is keyed by endpoint, so the per-candidate `a_p(d)` that BLIS's
`apForInstance(req, id)` computes is available with no upstream change. Evaluate it
inside the candidate loop, the way `scorer/prefix/plugin.go:108-121` does.

---

## 3. Least-projected-TTFT arm

Same enumeration, same latency law, same rollout — objective replaced by the
arrival's own projected TTFT. It exists to isolate the value of the externality
term, so it must share the focal arm's code.

Needs, from section 2: `StepsDone`, `KVBlocks`, `N_out`, `a_p`, `c_xfer`, and the
rollout.

**Does not need:** per-resident `ArrivalUs`, `FirstTokenUs`, `SLOClass`, or the
virtual queues. It never prices residents — only its own wait.

This makes it strictly cheaper to deploy than the focal arm. It needs no SLO class
carrier under any configuration — not even in the mixed-class follow-on, where the
focal arm and Kairos both would (5.5).

---

## 4. Kairos (published comparator)

Different mechanism. It asks whether to deflect the prefill *onto a decode node*,
chunked finely enough not to hurt the residents already there. `disaggregate =
false` here means "prefill on the decode node", the opposite polarity from the
other arms.

| BLIS signal | What it is | Why the algorithm needs it | llm-d mapping | Status |
|---|---|---|---|---|
| `PrefillTokensAhead` | remaining prompt tokens across running batch **and** wait queue (`sim/simulator.go:651`) | queue-wait term of the prefill-pool TTFT estimate | none | **MISSING** (see 5.7) |
| `RunningDecode[].SLOClass` | classes of current decode residents | chunk budget is `beta *` the **strictest** resident `tau_itl` — it protects incumbents, not the arrival (`sim/edpp_kairos.go:170`) | inert under single-class (strictest of one class = that class); otherwise as §2 | **Not needed** (5.5) |
| `ResidentPrefillTokens` > 0 or `PrefillTokensAhead` > 0 | is a prefill already in flight here | eligibility gate: at most one deflected prefill per decode node | none | **MISSING** (5.2, 5.7) |
| `BatchSize`, `KvTokensInUse` | as section 1 | per-chunk step time | as section 1 | Direct / Derived |
| chunk candidate list, `ChunkCap` | discrete descending chunk sizes | greedy schedule build | plugin config | **Config** |

**Does not need:** `a_p` (Algorithm 1 uses the full prompt length, deliberately —
do not substitute the uncached suffix), `N_out`, or the externality state.

Note both Kairos gates degrade toward "always deflect" if `ResidentPrefillTokens`
and `PrefillTokensAhead` read zero. Since neither has a real signal today, an
unfixed port silently becomes a much more aggressive policy than the published
one. That inverts the comparator the arm exists to provide.

---

## 5. Missing signals and how to get them

### 5.1 `KV` — two derivations that disagree

Two routes, and they do not measure the same thing:

- **Shadow sum**: add up `inputLen + tokensStreamed` over tracked residents.
  Per-request token counts, so a shared prefix is counted once **per resident**.
- **From the metric**: `KVCacheUsagePercent * CacheBlockSize * CacheNumBlocks`.
  Block-granular, so a shared prefix block is counted **once in total**, and
  requests still prefilling are included.

Both are available today — the metric route needs no upstream change, since every
engine config supplies block geometry either as `CacheInfo` labels (vLLM, SGLang)
or as separate gauges (trtllm-serve, Triton TRT-LLM). This is a definitional
choice, not a signal-availability gap.

The choice matters because BLIS is inconsistent with itself: its coefficients were
*fitted* on `Σ ProgressIndex` over **decoding requests only** — the calibration tap
accumulates `kv` inside the `else if len(req.OutputTokens) > 0` branch
(`sim/simulator.go:1238`), and `sim/step_recorder.go:56` documents it as "the summed
resident decode context" — but its policy *consumes* the block-level value —
`KvTokensInUse = UsedBlocks × BlockSize` (`sim/cluster/instance.go:289`,
`sim/kv/cache.go:446`, declared at `sim/routing.go:28`). So each route is faithful
to a different half of the simulator:

| Route | Faithful to |
|---|---|
| Shadow sum | what `C1` was **fitted** on — BLIS's simulated physics |
| Metric × capacity | what the policy **consumes** — the published decision function |

**Recommendation: prefer the metric route.** What transfers is the decision
function, and the metric matches it near-exactly: vLLM's usage is
`1 − free/(num_gpu_blocks − 1)` (`block_pool.py:get_usage`) against BLIS's
`(TotalBlocks − FreeBlockCnt)` (`kv/cache.go:446`, whose free-list counter is
annotated "vLLM parity"), both GPU-tier only. Use the shadow sum as the cold-start
and cross-check path.

**This is faithful to what the policy consumes, not to what `C1` was fitted on, so
report it as an accepted approximation rather than an exact mapping.** The largest
divergence is not prefix sharing but **prefilling requests' allocated blocks**: the
fitted `kv` excludes them entirely, occupancy includes them. A 45k-token prefill in
flight adds ~45,000 tokens that the fitted definition omits — `45,000 × 0.0476 =
2,142 µs` at H100 coefficients, roughly **12% of `T_iter`**, and an order of
magnitude larger than the prefix double-count below. That context also co-varies
with `S_pf`, so it is charged twice: once through `C1·KV` and again through
`C_pf·S_pf`. Since BLIS's own consumed value has the same property, this is a
mismatch we inherit rather than introduce — which is why the metric route remains
the right choice, and why it must still be declared.

The fitted-definition argument for the shadow sum is weaker than it looks: the
coefficients are the policy's *forecasting prior*, fit against BLIS's own
trained-physics model rather than against real vLLM, so matching them buys fidelity
to the simulator's physics, not to hardware.

**Still measure the ratio once on the target cluster.** The known divergence is the
shadow sum's per-resident prefix double-count. At `BatchSize = 8` that is roughly +28%
on `reasoning` (prefix 250 / mean input 1,000) and `interactive-chat` (1,000 /
4,000), and roughly +4% on `deep-research` (2,000 / 45,000) — material, workload-
dependent, and nowhere near the 100x the unit trap below guards against.

> **Unverified assumption.** "The metric matches BLIS near-exactly" assumes BLIS's
> KV cache does hash-based prefix sharing, so a shared block is counted once as in
> vLLM. Its `CacheHits`/`CacheHitRate` accounting implies it does, and the
> definitions above match structurally, but the allocation path was not traced. If
> this becomes load-bearing for a paper claim, test it directly: drive a known
> batch with prefix caching enabled and compare Σ shadow context against the metric
> estimate.

Two traps for translation to get right. Both are properties of the router and the
metric, not of any particular port.

> **Unit trap — `KVCacheUsagePercent` is a fraction, not a percentage.** Despite
> the name, `vllm:kv_cache_usage_perc` is in **[0,1]**: vLLM documents the gauge as
> *"KV-cache usage. 1 means 100 percent usage"* (`loggers.py:563`), the router
> passes it through unscaled (`extractor.go:127`), and the router's own saturation
> detector compares it against a threshold defaulting to `0.8`, validated to
> `(0,1]` (`utilization/config.go:33,153`). **Translation must not divide by 100** —
> doing so under-estimates KV by 100x, which collapses the `C1*KV` term that
> carries hardware heterogeneity.
>
> Note the sim2real pipeline repo's own `docs/transfer/blis_to_llmd_mapping.md`
> states the opposite and mandates the `/100` (inference-sim/sim2real#816). Trust
> the pinned checkout over that table. This trap is not hypothetical — the `/100`
> has already appeared in generated port code once.

> **Capacity trap — `KvCacheMaxTokenCapacity` is unusable.** It looks like the
> direct answer and earlier drafts of this doc named it, but it is **never
> assigned** on llm-d-router v0.9.0: declared at `datalayer/metrics.go:35`, copied
> in `Clone()` at `:77`, and populated nowhere. The extractor assigns exactly
> `WaitingQueueSize`, `RunningRequestsSize`, `KVCacheUsagePercent`,
> `CacheBlockSize`, and `CacheNumBlocks` (`extractor.go:109-170`).
>
> So it always reads 0, and **any code guarded on `KvCacheMaxTokenCapacity > 0` is
> unreachable** — a metric fallback written that way silently never fires and
> returns whatever its zero value is. Derive capacity from
> `CacheBlockSize * CacheNumBlocks` instead; both are populated by default from
> `vllm:cache_config_info` (`options.go:134` -> `populateCacheInfoMetrics`).
>
> `LIMITATIONS.md §C2` also names the dead field and needs the same correction.

### 5.2 `S_pf` — vLLM computes it and does not export it

vLLM already calculates exactly this quantity. `compute_iteration_details`
(`vllm/v1/utils.py:797`) produces `num_ctx_tokens` — the sum of scheduled tokens
over requests in context (prefill) phase, the same definition as BLIS's
`ResidentPrefillTokens`. It lands in `SchedulerIterationDetails`
(`vllm/v1/metrics/stats.py:171`) alongside `num_generation_requests` — the
**decode-only** count, which is BLIS's fitted `b_dec` and *not* the `BatchSize` the
policy consumes (§1) — and `elapsed_ms` (`T_iter`).

It goes to a **log line** (`loggers.py:182`), behind
`--enable-logging-iteration-details`. It is not a metric.

**Definitions confirmed identical.** Both sides are *per-iteration scheduled*
prefill tokens, so both are bounded by the engine's batch token budget:

- BLIS: `Σ req.NumNewTokens` over requests with
  `ProgressIndex < len(InputTokens)` (`sim/cluster/instance.go`).
- vLLM: `Σ num_scheduled_tokens[req]` over context-phase requests
  (`compute_iteration_details`, `vllm/v1/utils.py`).

This matters for the approximation below: `S_pf` is **not** the outstanding prompt
backlog, it is what is being prefilled *this step*.

#### Two runtime routes

**Route A — approximate from the shadow table** (what the port does today):
residents with no first token yet, each contributing its remaining prompt tokens.
Works without touching anything. See the impact analysis below — the error is
larger than previously documented, and in the opposite direction.

**Route B — expose it, then consume it.** Two changes, both required; neither is
sufficient alone.

1. *vLLM*: promote `num_ctx_tokens` to a Prometheus gauge. Small upstream patch;
   the value is already computed, this is serialization only. It also delivers
   `num_generation_requests` (decode-only; BLIS's fitted `b_dec`, not the consumed
   `BatchSize` — see §1) and `elapsed_ms` (`T_iter`).
2. *Router → plugin*: declare it under the model-server extractor's
   `customMetrics` in EPP config —
   `{attributeKey: "...", metricSpec: "vllm:num_ctx_tokens"}`
   (`factories.go:64`). The extractor scrapes it and stores it as an **endpoint
   attribute**, not as a `datalayer.Metrics` field:
   `ep.GetAttributes().Put(key, attrmetrics.ScalarMetricValue(...))`
   (`extractor.go:181`). The plugin reads it back with
   `attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), key)`
   (`attribute/metrics/data_types.go:28`), which returns `(value, ok)`.

   **No router fork is needed** — this path is config plus plugin code. But note
   the consequences: the value arrives on the *attribute* API rather than the
   `Metrics` struct, so it uses a different accessor from `BatchSize`/`KV`/`QueueDepth`;
   it is absent unless the config declares it; and the plugin must handle `ok ==
   false` explicitly rather than reading a zero.

The log is **not** a third runtime route. Iteration details are emitted per step
behind a debug flag, and the policy needs this value at every routing decision —
scraping logs is not a viable transport. Its use is separate and offline:
`(num_ctx_tokens, num_generation_requests, elapsed_ms)` per iteration is a
real-hardware analogue of BLIS's `BLIS_STEP_CSV` calibration tap, i.e. the missing
input for **refitting the coefficients against hardware** instead of against the
simulator. Worth harvesting for that reason alone, independent of `S_pf`.

#### Impact of Route A

The port sums each prefilling resident's *entire remaining prompt*, but `S_pf` is
only what is scheduled *this step*. So the approximation over-estimates by roughly
`nChunks = ceil(a_p / chunk)`.

At H100 coefficients, `BatchSize = 8`, `KV = 16,384`, chunk budget 2,048:

| Workload | mean input | nChunks | true `S_pf` | approx | `T_iter(decode)` true → approx | inflation |
|---|---:|---:|---:|---:|---|---:|
| `reasoning` | 1,000 | 1 | 1,000 | 1,000 | 23,581 → 23,581 µs | 1.0x |
| `interactive-chat` | 4,000 | 2 | 2,048 | 4,000 | 30,021 → 42,015 µs | 1.4x |
| `deep-research` | 45,000 | 22 | 2,048 | 45,000 | 30,021 → 293,948 µs | **9.8x** |

Three consequences:

- **The direction is over-estimation, not under.** Earlier drafts of this section
  and the port's own `sPfFor` comment both say the approximation under-counts and
  that `tIterPrefill` collapses toward `alpha_p`. That is wrong for the decode
  side.
- **It only fires where a prefill is already in flight on a decode node** — i.e.
  after the policy has elected *local* prefill for some earlier request. On a
  pure-disaggregated 1P2D fleet `S_pf` is 0 either way and the error is nil.
- **When it fires, it biases away from local placement**, since the node looks up
  to ~10x slower than it is. That is the *opposite* direction from the missing
  causal prefill-attention term (§1, `C_attn`), so the two partially offset — but
  they scale differently with prompt length, so they do not cancel.

The `alpha_p` collapse claim *is* right for the **prefill pool**, for a different
reason: if the EPP never registers residents on prefill endpoints (a joint `(d, p)`
placement must record on both, which is an open translation question), `S_pf` reads
0 there and `tIterPrefill = alpha_p` — 16,618 µs against ~29,202 µs for one 2,048
token chunk, so contention on the prefill pool is under-priced by ~1.8x.

**Cheap improvement, if Route B is deferred:** cap the estimate at the engine's
batch token budget — per resident `min(remaining, chunk)`, and cap the sum at
`--max-num-batched-tokens`. That cannot exceed what the scheduler could actually
schedule in one step, which removes the bulk of the 9.8x error at no cost and with
no new signal. It is still an approximation: the EPP cannot know *which* requests
the engine scheduled this step.

### 5.3 `GPU type` — the load-bearing one

Without it every candidate is scored under one GPU's physics and the policy
becomes hardware-blind, which is the exact failure mode it is meant to beat. The
heterogeneous result (0.906 vs stock llm-d's 0.554) depends entirely on this.

`nvidia.com/gpu.product` is a **node** label, and the EPP sees endpoints, not nodes.
But it does see **pod** labels: `EndpointMetadata` carries
`Labels map[string]string` per endpoint, alongside `NamespacedName`, `PodName`,
`Address`, `Port`, and `RankIndex` (`endpoint_metadata.go:27`). That makes this far
cheaper than it first appears.

> **Why pool-level config cannot work here.** An earlier draft proposed mapping each
> `InferencePool` to a coefficient set. That is not viable for the heterogeneous arm.
> Joint argmin over a mixed decode fleet requires **all** decode endpoints in one
> `InferencePool` (two decode model-services joined to one pool — see README
> "Scope"); two pools means two EPPs, each blind to the other's endpoints, so no
> joint choice is possible. One pool therefore holds both H100 and A100 endpoints,
> and a pool-level coefficient set makes the policy hardware-blind by construction —
> exactly the failure mode above. Discrimination has to be **per endpoint**.

**Option 1 — pod label you set yourself. Recommended; needs no new machinery.**

You control the Deployments, and each decode Deployment is already pinned to its
accelerator. So set the label directly in the pod template:

```yaml
# Deployment: decode-h100
spec:
  template:
    metadata:
      labels:
        pd-infocomm/gpu-product: NVIDIA-H100-80GB-HBM3   # you set this
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3     # already required for placement
```

```yaml
# EPP plugin config — the port's parameters already have this shape
plugins:
  - type: causal-slo-externality-joint-handler
    parameters:
      gpuTypeAttribute: pd-infocomm/gpu-product
      gpuTypeByLabel:
        NVIDIA-H100-80GB-HBM3: h100
        NVIDIA-A100-SXM4-80GB: a100
      defaultGpuType: ""        # empty => reject unknown, never default
```

No device plugin, no DaemonSet, no node-label propagation, and **no data producer** —
the label is already on the endpoint.

The obvious objection is that a hand-set label can lie, if a pod lands on a
different GPU. Keying the `nodeSelector` on the *same value* removes that: a
mismatch means the pod does not schedule at all, rather than serving traffic under
the wrong physics. **The static config becomes self-validating, and failure is loud
and at deploy time.**

**Option 2 — `PodName` / `NamespacedName` pattern match.** Same config shape, keyed
on pod-name prefix instead of a label, if labels cannot be set. Works, but is *not*
self-validating and breaks on a Deployment rename.

**Option 3 — vLLM exports the device. Worth bundling with the §5.2 patch.**

vLLM already knows: `CudaPlatform.get_device_name()` (`platforms/cuda.py:757`),
`get_device_capability()` → `(major, minor)` (`:734`), `get_device_total_memory()`
(`:770`). None is exported — all vLLM metrics carry only
`labelnames = ["model_name", "engine"]` (`loggers.py:468`).

Two patch shapes, and the difference is entirely about what the **router** can carry:

| Shape | vLLM change | Router change | Discriminates |
|---|---|---|---|
| **3a. scalar gauge** — `device_compute_capability` (e.g. `90`, `80`), optionally `device_total_memory` | small, same file as §5.2 | **none** — `customMetrics` already carries scalars | GPU *architecture* |
| **3b. info gauge** — value `1`, `device_name` in a label, matching the `vllm:cache_config_info` pattern (`loggers.py:1075-1098`) | small | **new feature** — generic label→attribute extraction does not exist | exact SKU |

3b is the "right" answer and is blocked on the router: `customMetrics` extracts a
scalar (`extractValue(metric) float64`, `spec.go:152`) stored as
`ScalarMetricValue float64`, and an info gauge's value is always 1 — the payload is
in labels. The only label-reading paths are hardcoded for LoRA and cache config.

**3a is the one to bundle.** If you are already patching `vllm/v1/metrics/loggers.py`
to promote `num_ctx_tokens` (§5.2), adding a compute-capability gauge is marginal
work in the same file and the same pattern, and it is consumable **today** through
`customMetrics` with no router change at all. Caveat: compute capability identifies
architecture, not SKU — H100 PCIe and SXM are both `9.0`, A100 40GB and 80GB both
`8.0` — so it discriminates this fleet correctly but would conflate SKUs elsewhere,
and the coefficients are fit per SKU and TP degree. Emitting `device_total_memory`
alongside narrows that, still with zero router change.

**Recommendation:** ship option 1 for the experiment — it is free, self-validating,
and unblocks immediately. Bundle 3a with the §5.2 vLLM patch anyway: near-zero
marginal cost, it removes the hand-set-label footgun for everyone, and it decouples
the vLLM release cadence (the long pole) from any later router work.

Whichever is chosen, **fail loudly on an unknown GPU type** rather than defaulting —
a silent default is what turns this into a hardware-blind policy.

> **Port note.** `endpointAttr(ep, params.GPUTypeAttribute)` currently assumes the
> *attribute* API (`ep.Get(key)`). For a pod label the accessor is
> `ep.GetMetadata().Labels[key]`; for a `customMetrics` scalar it is
> `attrmetrics.ReadScalarMetricValue(ep.GetAttributes(), key)`. The three are not
> interchangeable, and the current `TRANSLATE` marker points at the wrong one for
> options 1 and 2.

### 5.4 Per-resident state — the shadow table

The router has the hooks needed
(`pkg/epp/framework/interface/requestcontrol/plugins.go`):

| Hook | Fires | Populates |
|---|---|---|
| `PreRequest` | after placement, before dispatch | register resident: endpoint, arrival time, prompt length, class |
| `ResponseBody`, first invocation | first body chunk | first-token time, `TTFTSet` — **not `ResponseHeader`**, see below |
| `ResponseBody` | **every streaming chunk** | `StepsDone` |
| `ResponseBody` with `EndOfStream` | completion | deregister; update the `N_out` running mean |

#### First-token time comes from the body hook, not the header hook

`ResponseHeader` fires when the model server *begins responding*, before any token
exists. The router gets this right in its own handler layer
(`handlers/response.go:55`): `if reqCtx.FirstTokenTimestamp.IsZero() &&
len(responseBytes) > 0`.

A plugin cannot apply that rule directly. The `Response` struct passed to
requestcontrol plugins carries `RequestID`, `Headers`, `StartOfStream`,
`EndOfStream`, `ReqMetadata`, `Usage`, and `DynamicMetadata` — **no body bytes** —
so there is nothing to test for emptiness. Use `Response.StartOfStream`, or
`Response.Usage.CompletionTokens` first reaching >= 1. Prefer the usage route,
since per-chunk usage is already required for `StepsDone`.

Two further interface facts:

- **Non-final chunks run on a background goroutine.** `director.go:545` enqueues
  them to a per-request queue drained by `processResponseBodyQueue`, so a
  `time.Now()` taken inside the plugin's `ResponseBody` is the *dequeue* time, not
  chunk arrival. Recorded TTFT feeds the TTFT conjunct of `g()` for every resident.
  Declare the bias, or plumb the handler-layer timestamp through.
- **The shadow table needs a mutex** — written from that goroutine, read from the
  scheduling path.
- **`Response.Headers` is nil during body processing.** Anything header-borne (the
  class carrier of §5.5) must be captured at `PreRequest`.

`Response.Usage.CompletionTokens` gives exact counts rather than inferred ones,
provided usage is emitted per chunk — set `stream_options.continuous_usage_stats`
(`vllm/entrypoints/openai/engine/protocol.py:286`) or run the server with
`--enable-force-include-usage`. **Without one of those, usage arrives only in the
final chunk** and `StepsDone` must be inferred from chunk counts instead.

Two known limits: a **replicated EPP** splits the table, so each replica sees only
the residents it placed; and traffic bypassing the EPP is invisible. Prefer
vLLM's `RunningRequestsSize` for `BatchSize` (it counts everything) and use the shadow
table for the per-resident detail only.

### 5.5 SLO class carrier — not needed for the registered campaign

Used by the focal arm (resident deadlines) and Kairos (strictest resident TBT).
Not used by least-TTFT.

#### The registered campaign is single-class, so no carrier is required

All three workload specs declare `slo_class: standard`. The three *shapes* are a
cell dimension, not concurrent traffic — cells are keyed `fleet:workload:load`
(`h100_homogeneous:interactive:low_0p60`, …), 18 cells = 2 fleets x 3 workloads x
3 loads, 432 runs = 18 x 24 seeds. Each run drives one shape, so a second class is
never present to distinguish from.

One class means one tau triple and one N̂_out prior, which the port already handles:

```go
func (h *Handler) sloFor(class string) SLO {
    if s, ok := h.params.SLOClasses[class]; ok { return s }
    return h.params.SLO          // <- always this branch when sloClasses is empty
}
```

Populate `SLO`, leave `sloClasses` empty. Every request — arrival and resident —
gets identical targets regardless of what `SLOClass` holds; the field can stay `""`
forever, and `N̂_out` is one running mean instead of a table. No header, no gateway
change, no data producer, no plugin change.

This is **faithful, not a shortcut**: the simulator was single-class too.

#### `tenant_id` is not the class

An earlier draft suggested reusing `tenant_id` (`interactive-users`,
`reasoning-jobs`, `research-agents`) as a free discriminator. **Do not.** They are
orthogonal dimensions, and BLIS uses both in the same expression for different jobs
(`sim/admission.go:267`):

```go
if t.tracker.IsOverBudget(req.TenantID) && t.priorityMap.IsSheddable(req.SLOClass) {
    return false, "tenant-budget-shed"
}
```

`TenantID` drives per-tenant budget shedding (`bundle.go:23`, `tenant_budgets`).
`SLOClass` drives `targetsFor()` and `nHatFor()` — the policy's targets. Wiring the
fairness key into the SLO targets would give the three workloads *different* tau and
N̂_out, when the campaign gave them the same. That is a different experiment, not a
translation of this one.

`blis observe` keeps them separate too — two distinct headers (below).

#### For the mixed-class follow-on, the carrier is largely already built

Per-class *state* is the easy half: `SLOClasses` already exists and `sloFor()`
already looks up; `N̂_out` becomes a map keyed by class. The hard half is knowing
which class a request **is** — and most of that path is verified in code:

| Step | Status | Evidence |
|---|---|---|
| Generator emits the header | **done** | `blis observe` sets `x-gateway-inference-objective: req.SLOClass` (`cmd/observe.go:220`), and `x-gateway-inference-fairness-id: req.TenantID` (`:217`). Reads the same `--workload-spec` YAML the simulator used |
| All headers ingested, no allowlist | **done** | `handlers/request.go:50-52` copies every header, lowercased |
| Old header name honored | **done** | `GetLowerCaseHeaderValue` resolves `headerAliases`; `metadata/consts_test.go:28` asserts `HeaderNames(ObjectiveKey) == [ObjectiveKey, OldObjectiveKey]` |
| Pre-extracted | **done** | `handlers/request.go:54` sets `reqCtx.ObjectiveKey` |
| Reaches plugins | **done** | `requestcontrol/director.go:269` — `Headers: reqCtx.Request.Headers` |
| Plugins read headers | **proven in-tree** | `agentidentity` — `if id := request.Headers[header]; id != "" { request.FairnessID = id }` (`agent_identity.go:114-119`) |
| Plugin resolves + rejects unknown | **ours to write** | a few lines |
| Gateway forwards it to ext_proc | **unverified** | deployment config, not router code |

So no data producer is needed — `InferenceRequest.Headers` is on the type every
scheduling plugin receives.

Three traps:

- **Do not read the class from `Objectives`.** It is collapsed to
  `type RequestObjectives struct { Priority int }`. Consistent with
  `InferenceObjectiveSpec` carrying only `Priority` and `PoolRef` — neither the
  resource nor the parsed form can hold a class name. The **header value** is the
  carrier.
- **Keys are lowercased on ingest, and aliases only resolve through the helper.** A
  raw `req.Headers["x-llm-d-inference-objective"]` lookup misses observe's
  `x-gateway-*` form. Use `metadata.GetLowerCaseHeaderValue` or `reqCtx.ObjectiveKey`.
- **Objective headers are stripped on egress, not ingress.** `generateHeaders()`
  removes them so they do not leak to vLLM (`util/request/headers.go:30`). Plugins
  still see them; do not conclude from a backend packet capture that the header was
  dropped.

#### How to verify, cheapest first

1. **Unit test in the port's package, no cluster** — build an `InferenceRequest` with
   `Headers{"x-gateway-inference-objective": "premium"}`, assert `sloFor` resolves
   premium targets and that an unknown value is *rejected*. Same shape as
   `handlers/request_test.go:124`.
2. **In-process integration, still no cluster** — `test/integration/epp/harness.go`
   drives requests through ext_proc with synthetic endpoint metrics, exercising the
   real `HandleRequestHeaders` -> director -> plugin path.
3. **One live request** — `blis observe --num-requests 1`, plugin logging
   `req.Headers` at TRACE. Confirm the key is present, and that vLLM did *not*
   receive it (egress stripping working as designed).
4. **Ongoing guard** — count requests by *resolved* class and export it. A non-zero
   unknown/default bucket fails a mixed-class run. Pair with the coverage gate in
   §5.4; both catch "running on degraded inputs with nothing said".

#### Unknown classes must not default — in the follow-on

`targetsFor` (`sim/edpp.go:682`) falls back silently, and so does `nHatFor`. Once a
per-class table exists, an untagged reasoning request treated as interactive gets a
16 s deadline and a 300-token work estimate against a real 8,000-token generation —
under-pricing its decode demand by **~108x** (`Wd` is quadratic in output length).
Reject or loudly log instead.

Note this hazard is **specific to multi-class operation**. Under the single-class
config above the same silent fallback is the *desired* behaviour, since there is only
one target set to fall back to.

A classification failure also propagates: the externality needs every *resident's*
class, recorded at placement. A mistagged request corrupts the deadline estimates
used to score every other request for as long as it lives.

### 5.6 Scheduler-rollout state — no mapping, and it fails silently

BLIS's focal policy predicts admission and first token by replaying vLLM's
scheduler over a frozen snapshot. That needs the **ordered** wait queue with each
request's prompt and computed token counts (`sim/admission_estimator.go:41`).

The EPP has `WaitingQueueSize`: one integer. vLLM exports nothing per-queued
request.

This is the largest gap. It is also easy to miss, because the BLIS call site
degrades quietly:

```go
// sim/edpp.go:1702, inside jointSLOExternalityCandidateScore
if rolloutAdm, rolloutTTFT, ok := d.rolloutLocalTTFT(ec, ds, thetaD); ok {
    tAdmD, tHatLocal = rolloutAdm, rolloutTTFT
}
```

When `ok` is false it keeps the closed-form estimate — no error, no log. So a port
that cannot supply the queue contents runs a **different TTFT estimator** than the
one that produced the published 0.92/0.90 goodputs, and nothing says so.

#### What the rollout actually reads

Gated on `SchedulerStateObserved && MaxScheduledTokens > 0 && MaxBatchSize > 0`
(`schedulerRolloutTimes`), it consumes:

| Need | Source |
|---|---|
| `MaxScheduledTokens`, `MaxBatchSize`, `LongPrefillTokenThreshold` | static config — `--max-num-batched-tokens`, `--max-num-seqs`, `--long-prefill-token-threshold` |
| `BlockSizeTokens` | already have — `CacheBlockSize` |
| `FreeKVBlocks` | derivable — `CacheNumBlocks * (1 - KVCacheUsagePercent)` |
| `SchedulerRunning`, `SchedulerWaiting` (**ordered**), `CurrentScheduled`, `CurrentStepStartUs` | not exported today |

Per request, `SchedulerReqState` has 8 fields, and the rollout's arithmetic touches
only four of them — `prompt`, `computed`, `kvBlocks`, and `outputRemaining`
(derived). `id` is declared on `schedulerRolloutReq` and **never read**; the target
is identified by a `target bool`. `SLOClass` is used once, to pick N̂_out:

```go
outputEstimate := d.nHatFor(state.SLOClass).mean()   // schedulerReqForRollout
```

#### Options

1. **Accept the substitution** — the `rollforward` estimator over the shadow table
   (what the ported algorithm already does). Then **measure the cost**: re-run the
   simulation with the rollout disabled and report that number as the deployable
   baseline. No cluster, no code, and it is the honest headline. *Caveat: no
   rollout-disable flag was found in `cmd/root.go`, so this may need a small flag or
   a way to suppress `SchedulerStateObserved`.*
2. **Export it on `/metrics`, indexed by queue position.** See below — this is much
   cheaper than an earlier draft of this section claimed.
3. **At minimum, log loudly** in the plugin when the rollout path is unavailable,
   so the substitution is visible rather than silent.

Do 1 and 3 now; reach for 2 only if the measured cost in 1 is large.

#### Option 2: position-indexed gauges, not a new endpoint

An earlier draft called for a bespoke `/scheduler_state` JSON endpoint plus a new
router collector — two upstream features. That was too pessimistic.

Prometheus *can* carry this, and the ecosystem already does: `vllm:lora_requests_info`
is a gauge whose **labels carry lists** (`loggers.py:1065-1074`,
`waiting_lora_adapters` / `running_lora_adapters`), and llm-d-router already parses
label-borne lists from it (`populateLoRAMetrics`, `addAdapters`).

What makes the naive encoding wrong is **cardinality**, not expressiveness. In
Prometheus the label set *identifies the series*, so putting request IDs in labels —
or a mutating CSV of them — creates a new series every time the queue changes. The
queue changes every iteration (30-60 Hz), and `/metrics` is scraped by the monitoring
Prometheus as well as by the EPP, so that degrades shared infrastructure to serve one
consumer. vLLM already flags the LoRA metric as fragile for related reasons
(`loggers.py:1058`).

Index by **queue position** instead. Position is bounded; the numbers go in the
metric *value*, which is what gauges are for:

```text
vllm:scheduler_waiting_prompt_tokens{position="0"}    4096
vllm:scheduler_waiting_computed_tokens{position="0"}  2048
vllm:scheduler_waiting_kv_blocks{position="0"}         128
vllm:scheduler_step_age_us                          340000
```

- Bounded series count: K positions x ~4 fields, fixed.
- Ordering is explicit in the label — exactly what the FCFS replay needs.
- **Relative times.** `CurrentStepStartUs` and `ArrivalUs` are engine-clock instants
  compared against the EPP's `nowUs` inside `schedulerRollout`. The simulator had one
  clock; across hosts, ordinary NTP skew of 1-2 ms lands against 17-30 ms iterations.
  A value-field encoding cannot carry an absolute timestamp anyway, so it forces the
  correct design: export *age* ("µs since step start"), not a wall-clock instant.
- **K can be small.** The rollout pops from the head until the target is admitted, so
  requests 50-300 deep do not change the answer. K = 8-16 is likely sufficient.

Consumption may need **no new router code**: `Spec` already supports label-selected
gauges — the Triton config uses
`nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type=tokens_per}` — so
`customMetrics` entries of the form
`metricSpec: "vllm:scheduler_waiting_prompt_tokens{position=0}"` land each value as a
scalar endpoint attribute over the path already verified in §5.2. Verbose (K x fields
config entries), but config rather than a feature.

That puts option 2 in roughly the same effort class as §5.2 — one bounded vLLM patch
in the same file, plus router config.

#### The three limitations, and what they mean for the algorithms

| Limitation | Single-class | Mixed-class | Direction |
|---|---|---|---|
| No request ID | **none** | wrong N̂_out per queued request | unsigned |
| Truncation at K | none at shallow queues; real at high load | same | **under-estimates delay** |
| No snapshot atomicity | bounded noise | same | unsigned |

**No request ID — costs essentially nothing.** The rollout never reads `id`.
Identity is needed only for the `nHatFor(SLOClass)` lookup that sets
`outputRemaining`, i.e. how long each request holds a batch slot. Under single-class
`nHatFor("")` is one global mean, so there is nothing to look up. Under mixed classes
every queued request would take the default class's N̂_out, wrong by the class spread
(§5.5's example is 300 vs 8,000 tokens, ~27x on `outputRemaining`). Fixable *within*
this encoding: what you need is the **class**, not the ID, and class is
low-cardinality — `{position="0", slo_class="premium"}` is a safe label.

**Truncation at K — the one that matters.** The target is appended to the *end* of
the waiting queue (`waiting = append(waiting, ctx.target)`), so the rollout drains
everything ahead of it first. Expose only K of N and the target sits at position K+1
instead of N+1: admitted too early, so admission delay is **under-estimated** by
roughly the drain time of the (N-K) hidden requests. Two reasons this is the worst of
the three:

- It bites exactly where it discriminates. Queues at 0.60/0.80 x C_w are covered by
  K = 8-16; the 0.95 x C_w cells are where they deepen.
- The direction is dangerous and it **compounds**. A saturated node looks faster than
  it is, so the policy routes *toward* congestion — the same direction as the `tAdm`
  empty-`rems` case in §5.4, so the two stack rather than cancel.

*Mitigation, using a signal already present:* `WaitingQueueSize` gives the true N.
Detect `N > K` and model the tail — add `(N-K)` x mean observed per-request drain from
the K you can see. Silent truncation becomes a declared approximation at zero signal
cost.

**No snapshot atomicity — bounded, same order as the staleness already accepted.**
Within one scrape the K positions are mutually consistent if vLLM renders the page
from a single gather (likely, but not guaranteed by the format). Across the read, the
plugin pulls ~4K separate attributes and a concurrent 50 ms refresh can update
position 0 after position 1 was read, so the rollout replays a queue that never
existed — a duplicated or skipped request. The result is a plausible-but-wrong
estimate, not a crash, and the snapshot is already ~2-3 iterations stale
(`RefreshMetricsInterval` = 50 ms, `options.go:127`, against 17-30 ms iterations), so
a torn read adds noise of the same magnitude rather than a new class of error.

*Mitigation:* a seqlock. Export `vllm:scheduler_step_index` as one monotonic gauge,
read it before and after, and discard the snapshot if it moved — falling back to the
closed-form estimator, which already exists.

#### Instrument the degradation; do not try to eliminate it

None of the three introduces a new failure mode — **all three degrade into option 1**.
Missing class → default N̂_out, which is what single-class does anyway. Truncated →
bounded under-estimate, detectable via `WaitingQueueSize`. Torn → detect and fall
back.

So the requirement is detection, not elimination. That is the same defect this section
opens with: `ok == false` silently keeps the closed form. The fallback was never the
problem; the silence was. Every mitigation above is a detector plus that same
fallback — which makes the position-indexed encoding a strict improvement, provided
each degradation is counted and reported rather than assumed absent.

For the registered campaign, limitation 1 is nil and limitation 3 is noise. Only
truncation needs a decision, and `WaitingQueueSize` covers it.

### 5.7 `PrefillTokensAhead` (Kairos)

Remaining prompt tokens across both the running batch and the wait queue. No
metric exposes it.

- **Approximate from the shadow table**: sum remaining prompt tokens over
  requests this EPP routed to that node which have not reached first token. Misses
  the wait-queue portion for requests placed by other replicas.
- **Crude fallback**: `WaitingQueueSize * averagePromptLength`. The simulator
  explicitly moved away from this; if used, declare it.
- **Clean fix — the cheapest of the remaining gaps.** It is a **scalar**, so the §5.2
  consumption path applies with **zero router code**: one `customMetrics` entry, value
  lands as an endpoint attribute, plugin reads it with `ReadScalarMetricValue`.

#### The clean fix in detail

BLIS's definition (`sim/simulator.go:651`):

```go
remaining(req) = max(req.InputLen() - req.ProgressIndex, 0)
total          = Σ over RunningBatch.Requests + Σ over WaitQ
```

Every input exists in vLLM: `Request.num_prompt_tokens` (`v1/request.py:140`) and
`Request.num_computed_tokens` (`:168`), over `scheduler.running` and
`scheduler.waiting`. `SchedulerStats` is the natural home — it already carries
`num_running_reqs`, `num_waiting_reqs`, `kv_cache_usage`.

Three things to get right:

1. **This is new computation, not serialization.** Unlike §5.2's `num_ctx_tokens`
   (which vLLM already computes and merely does not export), this gauge does not
   exist — it is an O(queue-depth) sum added to the stats path. Small, but a larger
   vLLM diff than a field copy.
2. **Decide whether to include the skipped-waiting queue.** vLLM tracks
   `num_skipped_waiting_reqs` as a queue distinct from `waiting`; BLIS has a single
   `WaitQ`. Those requests are still ahead of an arrival, so probably include them —
   but declare the choice either way.
3. **Guard preempted requests.** A preempted request returned to `waiting` carries
   decode progress in `num_computed_tokens`, so `prompt - computed` goes negative.
   Mirror BLIS's `if n < 0 { return 0 }`, or preempted requests will *subtract* from
   the total.

Prefix caching works out on both sides: a cache-hit request has
`num_computed_tokens > 0` and so contributes only its uncached suffix, matching
ProgressIndex semantics. Worth one confirmation that BLIS advances `ProgressIndex` on
cache hits.

#### Why this one is worth more than an accuracy improvement

Reading zero here (and for `S_pf`) opens Kairos's eligibility gate **permanently** —
`ds.ResidentPrefillTokens > 0 || ds.PrefillTokensAhead > 0` is the "at most one
deflected prefill per decode node" check. Per §4, an unfixed port therefore becomes a
much more aggressive policy than the published one, **inverting the comparator the arm
exists to provide**.

So this fix protects a comparator's *validity*, not just the precision of an estimate.
That is a stronger argument than any of the other gaps carry. Guard against zero
explicitly rather than treating it as "idle", whichever route is taken.

---

## 6. Summary

| Signal | Focal | Least-TTFT | Kairos | Status |
|---|:-:|:-:|:-:|---|
| `BatchSize` (**not** `B_dec`) | x | x | x | Direct — running total including prefilling requests; `C0` was fitted on the decode-only count, negligibly (§1) |
| `QueueDepth` | x | x | x | Direct |
| Block size / capacity | x | x | x | Direct |
| `a_p` uncached suffix | x | x | | Direct — per candidate endpoint, via `CachedBlockCount()` (§2) |
| `KV` | x | x | x | Derived, and a **declared approximation** — prefer the metric route (matches what the policy consumes, not what `C1` was fitted on); shadow sum as cross-check (5.1) |
| `C_pf`, `C_attn` coefficients | x | x | x | Config — `C_attn` is load-bearing, see §1 |
| chunk budget (`--max-num-batched-tokens`) | x | x | x | Config — must match the engine |
| Per-resident `KVBlocks` | x | x | | Derived |
| `N_out` per class | x | x | | Derived (EPP-internal) |
| Virtual queues | | | | **Not needed** — no deficit term in this arm's score (§6.0) |
| `StepsDone` | x | x | | Shadow |
| `ArrivalUs` | x | | | Shadow |
| `FirstTokenUs` | x | | | Shadow |
| Resident remaining prefill | x | | | Shadow |
| Decode-node `RunningPrefill` (`collocPrefill`) | x | | | **MISSING** — omitted, declared as a DEVIATION |
| `S_pf` | x | x | x | **MISSING** — approximate (cap at chunk budget), or vLLM gauge **plus** router `customMetrics` config (5.2) |
| `GPUType` | x | x | x | **Config** — pod label set in the Deployment, read from `EndpointMetadata.Labels`; no vLLM/router change needed (5.3) |
| Resident `SLOClass` | x | | x | **Not needed** — campaign is single-class. Mixed-class: header already sent by `blis observe` and surfaced in `req.Headers` (5.5) |
| `PrefillTokensAhead` | | | x | **MISSING** — approximate from shadow table, or one scalar vLLM gauge + `customMetrics`; protects Kairos's eligibility gate (5.7) |
| Scheduler rollout | x | x | | **MISSING** — substitute and measure first; exportable as position-indexed gauges if the measured cost is large (5.6) |
| `c_xfer` | x | x | | Config, must be measured |

**Required by every arm:** `GPUType`. Without it the heterogeneous experiment is
meaningless — but it is no longer an *open* problem. Pod labels reach plugins
already, so this is a deployment-and-config task, not a signal gap (5.3).

**Blocking for the focal arm and Kairos:** nothing. The SLO class carrier was
previously listed here; it is not required, because every cell of the registered
campaign is single-class (5.5). It becomes a prerequisite only for the mixed-class
follow-on, and even there most of the path already exists.

**So nothing on this list is currently blocking.** `GPUType` is deployment config
(5.3), the class carrier is unnecessary (5.5), and the remaining gaps are declared
approximations rather than missing mappings.

**Not blocking, but must be declared:** `S_pf`, decode-node `RunningPrefill`
(`collocPrefill`), `PrefillTokensAhead`, and the scheduler rollout substitution.
Each degrades the policy in a known direction; the risk is shipping them silently.
The two prefill-population omissions both **under-price** contention, so the argmin
is biased toward whichever candidate carries more unpriced prefill work.

### 6.0 Recommended route vs. most faithful route

Two different questions, deliberately separated. **Recommended** is what to run now.
**Most faithful** is what best reproduces the *intent* of the BLIS signal, assuming
willingness to patch vLLM, the EPP, and `blis observe` — which is the standing
assumption for this study. Where they differ, the difference is the declared deviation.

`≡` means the recommended route already *is* the faithful one.

#### Already faithful — no gap

| Signal | Route | Why it is faithful |
|---|---|---|
| `BatchSize` | `RunningRequestsSize` ← `vllm:num_requests_running` | ≡ definitions verified identical on the **consume** side: `len(self.running)` vs `len(RunningBatch.Requests)`, both counting prefilling requests. `C0` was fitted on the decode-only count instead, but the fit uses only pure-decode rows where the two coincide and `C0` is ~5.3-5.9 µs/req, so the gap is under 0.1% of `T_iter` (§1) |
| Block size / capacity | `CacheBlockSize`, `CacheNumBlocks` ← `vllm:cache_config_info` | ≡ same quantities |
| `N̂_out` | EPP-internal censored per-class running mean | ≡ BLIS supplies it the same way by design; must stay censored (INV-9) |
| Resident `ArrivalUs`, `InputLen` | shadow table via requestcontrol hooks | ≡ the EPP routed the request and observes its lifecycle, so these are as exact as the simulator's |
| Resident `FirstTokenUs` | first `ResponseBody` invocation | **near-faithful**: the event is observed exactly, but non-final chunks run on a background queue so the plugin's timestamp is dequeue time, not chunk arrival (§5.4). Faithful means reading the handler's `FirstTokenTimestamp`, taken synchronously |
| `SLOClass` | single-class config, `sloClasses` empty | ≡ every registered cell is single-class, exactly as the simulator ran (§5.5) |
| `c_xfer` | measured on the target interconnect | ≡ **provided it is actually measured**; inheriting the simulated value is the unfaithful path |
| chunk budget, block size, `max-num-seqs` | config mirroring the engine flags | ≡ same values |

#### Gap closable by the §6.1 scalar bundle

| Signal | Recommended now | Most faithful | What closes it |
|---|---|---|---|
| `S_pf` | shadow-table sum **capped at the batch token budget**, declared | `vllm:num_ctx_tokens` — vLLM already computes the identical per-step quantity | export-only gauge |
| `PrefillTokensAhead` | shadow-table approximation, guard against reading zero | `vllm:num_pending_prefill_tokens` — exact `Σ max(prompt − computed, 0)` over running + waiting | one new sum |
| `GPUType` | pod label in the Deployment template, enforced by a matching `nodeSelector` | the **engine reporting its own device** rather than a human assertion | `device_compute_capability` (architecture-level), or a device-name info gauge + a router-side label extractor for exact SKU |

The pod-label route is *more precise* (exact SKU) but *asserted*; the gauge is
*self-reported* but coarser. Pairing the label with `nodeSelector` closes most of the
trust gap, which is why it is recommended despite not being the faithful shape.

#### Larger gaps

| Signal | Recommended now | Most faithful | Note |
|---|---|---|---|
| Scheduler rollout | accept the `rollforward` substitution, measure its cost, log loudly (§5.6 options 1+3) | position-indexed gauges with an `slo_class` label, `WaitingQueueSize` tail model, and a `scheduler_step_index` seqlock | still an approximation at bounded K; fully faithful means unbounded K, which is impractical — so "faithful within a declared bound" |
| `collocPrefill` | omitted, declared as a DEVIATION | per-request prefill state on the *decode* node | **an interim improvement needs no upstream work**: approximate from the same shadow subset `sPfFor` already uses (residents without a first token on that endpoint) |
| Residents this EPP did not place | shadow table + a single EPP replica | per-request state from vLLM, so the policy sees *all* residents rather than only its own | single replica makes the shadow table near-complete, which is why it is acceptable (§5.4) |
| `KV` | metric route: `KVCacheUsagePercent * CacheBlockSize * CacheNumBlocks` | `Σ ProgressIndex` over **decoding requests only** — what `C1` was fitted on | matches what the policy *consumes*, not what the coefficients were fitted on. Dominant divergence is prefilling requests' allocated blocks (~12% of `T_iter` at 45k prefill), which BLIS's consumed value also includes — so the mismatch is inherited, not introduced. Declare it (§5.1) |
| `a_p` uncached suffix | **per-candidate** `a_p(d)` via `endpoint.Get(...)` in the candidate loop | ≡ the same thing — BLIS calls `apForInstance(req, id)` per instance | recommended *is* faithful: plugin-side only, no upstream change. The port carries one value for brevity; there is no reason to keep it (§2) |

#### The least faithful part of the transfer, and it is not a signal

| | Recommended now | Most faithful |
|---|---|---|
| Coefficients `alpha`, `C0`, `C1`, `C_pf`, `C_attn` | the frozen `inputs/coeffs-*.json`, used as the policy's forecasting prior | **refit against real hardware** using the calibration tap in §5.2 option 3 — `(num_ctx_tokens, num_generation_requests, elapsed_ms)` per iteration |

The coefficients were fit against BLIS's own trained-physics model, not against real
vLLM. Every latency projection in every arm rests on them, so this is the largest
fidelity gap in the whole transfer — larger than any individual signal. It also
resolves §5.1's fit/consume mismatch (coefficients fitted on decoding-request
context sums but consumed against block-level occupancy) and §1's (`C0` fitted on
the decode-only count but consumed against the running total), and the instrument is
already in the §6.1 bundle.

**The fit reports are direct evidence.** `inputs/coeffs-*.json` records `r2` of
`0.9999999983` (H100) and `0.9999999994` (A100), with `cond_b_dec_kv` of 2.50 and
2.38 respectively. Real hardware does not fit a three-parameter linear model to ten
significant figures; this is OLS recovering the linear model that generated the
rows. Note also that `coeffs-llama70b-a100real-tp4.json` is named "a100real" and
sources `/tmp/theta_a100real/*.csv` — the name should not be read as evidence of
hardware measurement, since it carries the same synthetic signature as the H100 set.

#### Reclassified: not needed by the focal arm at all

| Signal | Finding |
|---|---|
| Virtual queues `z_TTFT`, `z_ITL` | **Not used by this arm.** `jointSLOExternalityCandidateScore` contains no historical-deficit term; the z-terms are consumed by the work-currency drift-plus-penalty deciders (`sim/edpp.go:922`, `:1339`). The port's own `Score` doc says the contract is "with NO historical TTFT/ITL deficit term". The §2 and §6 rows have been corrected accordingly. |
| `Wd` | Computed but unused in the focal arm — upstream it feeds only the capacity term, which `NoCapacity=true` disables. Dead code here is *correct*. |

#### One definitional decision still open

`QueueDepth` maps to `WaitingQueueSize` ← `vllm:num_requests_waiting`, but BLIS's
`WaitQ` is a single queue while vLLM tracks `waiting` **and** a skipped-waiting queue
(`SchedulerStats.num_skipped_waiting_reqs`, surfaced in §5.7). If
`num_requests_waiting` excludes the skipped queue, `QueueDepth` under-counts requests
that are genuinely ahead of an arrival. Faithful is to decide explicitly — most likely
the sum — and declare it. Same decision as §5.7's, and it should be made once for both.

### 6.1 The upstream ask: one vLLM patch, four scalars, no router change

Treating each gap as its own upstream negotiation overstates the work. Four of them
reduce to **scalar gauges on the existing `/metrics` path**, which means all four ride
`customMetrics` into endpoint attributes with **zero llm-d-router code** (the path
verified in §5.2). One PR, essentially one file (`vllm/v1/metrics/`):

| Gauge | vLLM work | Serves | Section |
|---|---|---|---|
| `num_ctx_tokens` | **export only** — already computed by `compute_iteration_details` | `S_pf` | §5.2 |
| `num_pending_prefill_tokens` | **new sum** over `running` + `waiting` | `PrefillTokensAhead`; protects Kairos's gate | §5.7 |
| `scheduler_step_index` | **export only** — `SchedulerStats.step_counter` already exists | seqlock for torn-read detection | §5.6 |
| `device_compute_capability` | **export only** — `get_device_capability()` already exists | GPU-type discrimination (secondary to the pod label) | §5.3 |

Three of the four are export-only; only `num_pending_prefill_tokens` needs new
arithmetic. Consumption for all four is `customMetrics` config plus a
`ReadScalarMetricValue` call.

What this bundle does **not** solve, and deliberately so:

- **§5.6's per-request queue contents.** Those are not scalars. They need the
  position-indexed encoding, which is a larger and separable ask — defer it behind
  option 1's measurement.
- **§5.3's exact GPU SKU.** `device_compute_capability` distinguishes architecture
  (H100 `9.0` vs A100 `8.0`), not SKU, and the pod label (§5.3 option 1) is both
  cheaper and more precise. The gauge is a durable secondary, not the primary route.
- **`collocPrefill`** (decode-node `RunningPrefill`), which is per-population rather
  than a scalar.

Sequencing note: none of this is on the critical path for the registered campaign —
every item has a declared approximation. The bundle is worth doing because it is cheap
and because `num_pending_prefill_tokens` removes a **comparator-inverting** failure
mode rather than merely improving an estimate (§5.7).

**Cheapest path to a runnable experiment:** GPU-product pod label set in each decode
Deployment and enforced by a matching `nodeSelector` (5.3 option 1), single-class
config with `sloClasses` left empty and no class carrier at all (5.5), shadow table for
per-resident state (5.4), shadow-table approximations for `S_pf` and
`PrefillTokensAhead` — **capped at the engine batch token budget**, see 5.2 — and
the `rollforward` substitution with its cost measured in simulation first (5.6
option 1). No upstream vLLM change required for any of it.

**Fix before running:** the `/100.0` unit bug in 5.1.
