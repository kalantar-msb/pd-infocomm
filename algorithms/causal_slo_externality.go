// Package causalsloexternality is the sim2real ALGORITHM SOURCE for the INFOCOM
// 2027 study's treatment arm. It is a faithful reference port of the BLIS policy
// at
//
//	sim/edpp.go             jointSLOExternalityCandidateScore  (the score)
//	sim/edpp_var.go         varDecodeContribution, sloCompositeValue, goodSelf
//	sim/edpp_coeffs.go      tIterDecode/tIterPrefill/Wp/Wd    (the latency law)
//	sim/edpp_scheduler_rollout.go  frozen-snapshot admission/first-token rollout
//
// (fork vishakha-ramani/inference-sim @ 871b169b, branch infocom-implementation)
// expressed against the llm-d-router Endpoint-Picker (EPP) plugin interfaces.
// The sim2real-translate agent team reconciles every `// TRANSLATE:` touchpoint
// below against the pinned llm-d-router checkout and wires this into cmd/ +
// pkg/ + the treatment overlay.
//
// The SCORE ALGEBRA and the LATENCY LAW are the scientific content and are
// ported verbatim in intent. The state acquisition is NOT — see "Fidelity gap".
//
// ─── What this policy does ───────────────────────────────────────────────────
// JOINT CAUSAL-SLO-EXTERNALITY PLACEMENT. Each arriving request is placed by a
// single joint action: pick a decode instance d, and either prefill locally on d
// or remotely on a prefill instance p and transfer the KV. Every candidate
// (d, nil) and (d, p) is scored and the ARGMIN wins:
//
//	score(d, p) = V * ( externality(d, p) − ownGood(d, p) )
//
//	externality = Σ_residents [ g(baseline completion) − g(placed completion) ]
//	ownGood     = g( projected TTFT, projected E2E )  for the arrival itself
//	g(ttft,e2e) = sigmoid((τ_ttft − ttft)/τ_ttft) · sigmoid((τ_e2e − e2e)/τ_e2e)
//
// Both completions come from the same latency law under the CANDIDATE'S OWN
// coefficients θ_i. That is the whole mechanism: an A100 decoder has c1 = 0.0782
// µs/token against the H100's 0.0476, so an identical request is correctly
// priced ~1.6x more expensive there. Hardware-blind policies cannot do this and
// collapse on a mixed fleet (llm-d threshold: 0.554 goodput where this policy
// holds 0.906 — see README.md).
//
// ─── Honest status (READ THIS) ───────────────────────────────────────────────
// In simulation this policy WON on the heterogeneous H100+A100 decode fleet
// (worst-cell regret 0.0031 vs 0.3517 for stock llm-d) and DID NOT WIN on the
// homogeneous H100 fleet (0.0100 vs llm-d's 0.0047). A per-condition-tuned
// STATIC (φ, ψ) split still beat it by 0.0043 goodput across all 18 cells. So:
// the transfer is worth running only on a mixed-accelerator decode pool, and the
// pre-registered expectation is "competitive everywhere, decisively better where
// hardware is heterogeneous" — not a uniform win.
//
// ─── Fidelity gap (the main threat to validity) ───────────────────────────────
// The rollout needs, per resident: remaining decode steps, arrival time,
// realized TTFT, and resident context tokens. In BLIS these come from the
// simulator's own event queue. llm-d's datalayer.Metrics exposes ONLY aggregates
// (RunningRequestsSize, WaitingQueueSize, KVCacheUsagePercent,
// KvCacheMaxTokenCapacity, CacheBlockSize). The EPP therefore maintains its own
// SHADOW RESIDENT TABLE, populated from the request lifecycle it already
// observes. Where a quantity is unobtainable the port degrades explicitly and
// says so — every such spot is a `// TRANSLATE:` or `// DEVIATION:` marker.
package causalsloexternality

import (
	"context"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// HandlerType is the plugin `type` string referenced from EndpointPickerConfig
// `plugins:` and from a schedulingProfile `pluginRef`.
//
// NOTE this is a ProfileHandler, NOT a Scorer. A Scorer cannot express this
// policy: it sees one profile's candidate list and returns one score per
// endpoint, whereas the joint argmin ranges over the CROSS PRODUCT of decode and
// prefill candidates. llm-d's stock disagg-profile-handler is strictly
// decode-first (pd_profile_handler.go:186 calls the decider with the
// already-chosen decode endpoint), which is precisely the decomposition this
// policy's ablation beats by +0.0485 goodput. Reproducing that decomposition
// would forfeit the paper's claim.
//
// TRANSLATE: confirm the ProfileHandler + Picker composition actually permits a
// joint choice on the pinned checkout. The shape assumed here is:
//   - Pick() requests BOTH profiles on the first call (not decode-then-prefill).
//   - Each profile's Picker returns the FULL ranked []Endpoint in
//     ProfileRunResult.TargetEndpoints rather than a single argmax.
//   - ProcessResults() enumerates pairs, scores them, and emits the winner.
//
// If a Picker may not return the full list, the fallback is to have this handler
// read both pools from plugin.Handle and score pairs in Pick(); Handle.PodList()
// returns only NamespacedNames, so endpoint metrics would then have to come from
// a datalayer lookup. Resolve this before writing pkg/.
const HandlerType = "causal-slo-externality-joint-handler"

// ─── Latency law (sim/edpp_coeffs.go, ported verbatim) ───────────────────────

// Coeffs are the frozen E3 latency-law coefficients for ONE GPU type. Units:
// AlphaD/AlphaP µs, C0 µs/req, C1/CPf µs/token, CAttn µs/token².
//
// These are POLICY parameters, not simulator parameters — they are the policy's
// forecasting prior. They arrive inline under this plugin's `parameters.coeffs`,
// keyed by GPU type (see config.md "Latency-law coefficients"). They were fit
// against BLIS's own trained-physics model, not against real vLLM; see that
// section for what that does and does not license.
type Coeffs struct {
	AlphaD float64 `json:"alphaUs"`
	AlphaP float64 `json:"alphaPUs"`
	C0     float64 `json:"c0UsPerReq"`
	C1     float64 `json:"c1UsPerToken"`
	CPf    float64 `json:"cPfUsPerToken"`
	CAttn  float64 `json:"cAttnUsPerUnit"`
}

// tIterDecode is the decode iteration time (µs) at the given batch state:
// α + C0·B_dec + C1·KV + CPf·S_pf.
func (c Coeffs) tIterDecode(bDec int, kv, sPf int64) float64 {
	return c.AlphaD + c.C0*float64(bDec) + c.C1*float64(kv) + c.CPf*float64(sPf)
}

// tIterPrefill is the prefill iteration time (µs). A dedicated prefill server
// runs no decode work, so B_dec = KV = 0.
func (c Coeffs) tIterPrefill(sPf int64) float64 {
	return c.AlphaP + c.CPf*float64(sPf)
}

// wp is the prefill demand (µs) of ap uncached tokens for a prompt of full
// length ar. The trajectory sum of the causal per-step charge CPf·s +
// CAttn·s·(prefix + s/2), integrated over prefix ar−ap to ar — hence
// (ar − ap/2). At ap == ar this is CPf·ar + 0.5·CAttn·ar².
func (c Coeffs) wp(ap, ar int) float64 {
	a, r := float64(ap), float64(ar)
	return c.CPf*a + c.CAttn*a*(r-a/2.0)
}

// wd is the decode demand (µs) for a prompt of length ar generating o output
// tokens: the exact discrete sum Σ_{k=0}^{o-1}(C0 + C1·(ar+k)).
//
// o is the N̂_out ESTIMATE at routing time. It must never be the true output
// length — that would make the arm an oracle and invalidate it.
func (c Coeffs) wd(ar int, o float64) float64 {
	if o <= 0 {
		return 0
	}
	r := float64(ar)
	return c.C0*o + c.C1*o*(r+(o-1)/2.0)
}

// ─── SLO value kernel (sim/edpp_var.go, kernel "composite") ──────────────────

// SLO holds one class's targets in µs. A zero target disables that conjunct.
type SLO struct {
	TauTTFTUs float64 `json:"tauTtftUs"`
	TauITLUs  float64 `json:"tauItlUs"`
	TauE2EUs  float64 `json:"tauE2eUs"`
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

// compositeValue is g(): the smooth TTFT x E2E SLO value in [0,1]. Note the
// asymmetry with REPORTED goodput, which is the HARD three-way conjunction
// including mean ITL. The routing value deliberately drops the ITL conjunct
// (see PUBLIC-LOAD-STATIC-BENCHMARK-PROTOCOL.md, "Frozen question"); reported
// goodput keeps it. Do not "fix" this by adding ITL here.
func compositeValue(slo SLO, ttftUs, e2eUs float64) float64 {
	u := 1.0
	if slo.TauTTFTUs > 0 {
		u *= sigmoid((slo.TauTTFTUs - ttftUs) / slo.TauTTFTUs)
	}
	if slo.TauE2EUs > 0 {
		u *= sigmoid((slo.TauE2EUs - e2eUs) / slo.TauE2EUs)
	}
	return u
}

// goodSelf is the arrival's own projected SLO value. tHatFromNow is projected
// TTFT measured from the routing instant; tIterAfter is the decode iteration
// time AFTER the arrival joins the batch (full B+1 re-timing).
func goodSelf(slo SLO, tHatFromNow, tIterAfter, nOut float64) float64 {
	return compositeValue(slo, tHatFromNow, tHatFromNow+nOut*tIterAfter)
}

// ─── Shadow resident table ───────────────────────────────────────────────────

// resident is one in-flight request as the EPP believes it to be. This is the
// substitute for BLIS's exact per-request state.
//
// DEVIATION from simulation, per field:
//   - ArrivalUs, FirstTokenUs, InputLen: EXACT. The EPP routed this request and
//     observes its lifecycle, so these are as good as the simulator's.
//   - RemainingSteps: ESTIMATED as N̂_out minus tokens streamed so far. BLIS uses
//     a censored per-class estimate too (censorOracleRemaining), so this is a
//     like-for-like degradation rather than a new one.
//   - ResidentPrefillTokens (S_pf): NOT OBTAINABLE. vLLM exports no metric for
//     tokens currently being prefilled on a node. See sPfFor below.
type resident struct {
	ArrivalUs      int64
	FirstTokenUs   int64
	FirstTokenSet  bool
	InputLen       int
	TokensStreamed int
	NOutEstimate   float64
	SLOClass       string
}

// remainingSteps is the resident's estimated remaining decode iterations.
// Negative marks censored/unknown, which contributes ZERO externality (matching
// BLIS's `if cr.rem < 0 { continue }`) rather than a guessed value.
func (r *resident) remainingSteps() int64 {
	if r.NOutEstimate <= 0 {
		return -1
	}
	rem := int64(math.Round(r.NOutEstimate)) - int64(r.TokensStreamed)
	if rem < 0 {
		return 0
	}
	return rem
}

// residentTable is the EPP's own view of every in-flight request, keyed by
// endpoint address then request ID.
//
// TRANSLATE: the correct llm-d hooks to populate this are the requestcontrol
// lifecycle extension points — the post-schedule observer (where the placement
// becomes fact), the first-token/streaming hook, and the completion hook. In the
// sibling bundles this is the "observer after argmax" (RecordPlacement). Confirm
// the exact interface names and that a ProfileHandler may register on them.
//
// TRANSLATE: a REPLICATED EPP splits this table and silently degrades the
// policy — each replica would see only the residents it placed and systematically
// under-price contention. Run ONE EPP replica for the first experiment, or move
// this to shared state before scaling out.
type residentTable struct {
	mu sync.RWMutex
	by map[string]map[string]*resident
}

func newResidentTable() *residentTable {
	return &residentTable{by: map[string]map[string]*resident{}}
}

func (t *residentTable) snapshot(endpointKey string) []*resident {
	t.mu.RLock()
	defer t.mu.RUnlock()
	src := t.by[endpointKey]
	out := make([]*resident, 0, len(src))
	for _, r := range src {
		copied := *r
		out = append(out, &copied)
	}
	// Deterministic order so the traced argmin is reproducible. BLIS sorts
	// snapshots by instance ID for the same reason (sortedSnapshotsByID).
	sort.Slice(out, func(i, j int) bool { return out[i].ArrivalUs < out[j].ArrivalUs })
	return out
}

// RecordPlacement / RecordFirstToken / RecordCompletion are the lifecycle
// observers. Bodies elided in the reference port — the translate team wires them
// to the real extension points.
func (t *residentTable) RecordPlacement(endpointKey, requestID string, r resident) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.by[endpointKey] == nil {
		t.by[endpointKey] = map[string]*resident{}
	}
	t.by[endpointKey][requestID] = &r
}

func (t *residentTable) RecordCompletion(endpointKey, requestID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.by[endpointKey], requestID)
}

// ─── Instance state ──────────────────────────────────────────────────────────

// instState is one candidate instance's batch state at the routing instant —
// the port of BLIS's RoutingSnapshot, restricted to what a real EPP can know.
type instState struct {
	Key     string
	GPUType string // selects Coeffs; see gpuTypeFor
	BDec    int    // resident decode batch size
	KV      int64  // Σ resident context tokens
	SPf     int64  // resident prefill tokens — see sPfFor
	Queue   int    // waiting queue depth
}

// snapshotOf builds an instState from an llm-d endpoint plus the shadow table.
//
// TRANSLATE: exact accessor names on scheduling.Endpoint / datalayer.Metrics.
// The v0.9.0 field set is RunningRequestsSize, WaitingQueueSize,
// KVCacheUsagePercent, KvCacheMaxTokenCapacity, CacheBlockSize, CacheNumBlocks.
func (h *Handler) snapshotOf(ep fwksched.Endpoint) instState {
	m := h.getMetrics(ep)
	key := endpointKey(ep)
	residents := h.residents.snapshot(key)

	// B_dec: prefer vLLM's own count — it includes requests this EPP replica did
	// not place. Fall back to the shadow table.
	bDec := m.RunningRequestsSize
	if bDec == 0 {
		bDec = len(residents)
	}

	return instState{
		Key:     key,
		GPUType: h.gpuTypeFor(ep),
		BDec:    bDec,
		KV:      kvTokensFor(m, residents),
		SPf:     h.sPfFor(key, residents),
		Queue:   m.WaitingQueueSize,
	}
}

// kvTokensFor recovers Σ resident context tokens.
//
// DEVIATION: BLIS's KV is the exact summed ProgressIndex over residents. vLLM
// reports KV cache OCCUPANCY, which also counts prefix-cache blocks retained for
// requests that already finished. On a prefix-caching deployment that
// over-counts, inflating C1·KV and making every instance look uniformly busier.
// The shadow-table sum under-counts instead (it misses residents this replica did
// not place). We prefer the shadow sum when it is non-trivial and fall back to
// the metric, so the systematic error is at least in a known direction.
//
// TRANSLATE: quantify this gap once on the target cluster — drive a known batch,
// compare Σ shadow context against KVCacheUsagePercent·KvCacheMaxTokenCapacity,
// and record the ratio in config.md as a declared deviation.
func kvTokensFor(m *metricsView, residents []*resident) int64 {
	var shadow int64
	for _, r := range residents {
		shadow += int64(r.InputLen + r.TokensStreamed)
	}
	if shadow > 0 {
		return shadow
	}
	if m != nil && m.KvCacheMaxTokenCapacity > 0 {
		// UNITS: KVCacheUsagePercent is a FRACTION in [0,1], not a percentage,
		// despite the name — do NOT divide by 100. vLLM documents the gauge as
		// "KV-cache usage. 1 means 100 percent usage" (vllm/v1/metrics/loggers.go:563
		// -> loggers.py:563); llm-d-router passes it through unscaled
		// (extractor.go:127), scores 1 - KVCacheUsagePercent
		// (kvcache_utilization.go:79), and validates its own utilization threshold
		// to (0,1] with a 0.8 default (utilization/config.go:33,153).
		// NOTE docs/transfer/blis_to_llmd_mapping.md in the sim2real pipeline repo
		// states the opposite and mandates /100 — it is wrong; see sim2real#816.
		return int64(m.KVCacheUsagePercent * float64(m.KvCacheMaxTokenCapacity))
	}
	return 0
}

// sPfFor is the resident prefill-token count S_pf.
//
// DEVIATION — NO REAL SIGNAL EXISTS. vLLM exports no metric for tokens currently
// mid-prefill. The shadow table can approximate it: a resident whose first token
// has not arrived is still prefilling, and its remaining prompt tokens are a
// bounded estimate of its contribution. That is what we do here.
//
// Consequence: on the DEDICATED PREFILL pool, tIterPrefill(S_pf) reduces toward
// α_p and the policy under-prices prefill-pool contention. The paper's own
// asymmetry analysis says the DECODE-side term is dominant (sim/edpp_var.go,
// varPrefillDisagg: "first-order contention model ... the decode-side asymmetry
// is the dominant mechanism"), so this is the least damaging place to degrade —
// but it must be declared, not hidden.
func (h *Handler) sPfFor(_ string, residents []*resident) int64 {
	var sPf int64
	for _, r := range residents {
		if r.FirstTokenSet {
			continue // past prefill
		}
		remaining := int64(r.InputLen - r.TokensStreamed)
		if remaining > 0 {
			sPf += remaining
		}
	}
	return sPf
}

// gpuTypeFor selects which Coeffs entry applies to this endpoint. THIS IS THE
// LOAD-BEARING SIGNAL FOR THE ENTIRE EXPERIMENT: without it every candidate is
// scored under one GPU's physics and the policy becomes hardware-blind — i.e.
// exactly the failure mode it is supposed to beat.
//
// TRANSLATE: surface the node label `nvidia.com/gpu.product` as an endpoint
// attribute (H100 -> "NVIDIA-H100-80GB-HBM3", A100 -> "NVIDIA-A100-SXM4-80GB"),
// then map label -> coeffs key. If the pinned checkout cannot expose node labels
// to a plugin, this experiment cannot run on a mixed fleet; say so loudly rather
// than defaulting.
func (h *Handler) gpuTypeFor(ep fwksched.Endpoint) string {
	if t, ok := endpointAttr(ep, h.params.GPUTypeAttribute); ok {
		if mapped, found := h.params.GPUTypeByLabel[t]; found {
			return mapped
		}
	}
	return h.params.DefaultGPUType
}

func (h *Handler) coeffsFor(gpuType string) Coeffs {
	if c, ok := h.params.Coeffs[gpuType]; ok {
		return c
	}
	return h.params.Coeffs[h.params.DefaultGPUType]
}

// ─── Rollout: projected TTFT ─────────────────────────────────────────────────

// tAdm estimates admission delay (µs) — how long the arrival waits before its
// first scheduled step.
//
// DEVIATION: BLIS runs the "rollforward" estimator over a frozen FCFS snapshot
// with per-resident remaining steps, simulating vLLM's scheduler forward until
// the arrival is admitted. This port keeps that shape but over the shadow table,
// and vLLM's real scheduler is not plain FCFS. Two ways the estimate is
// systematically wrong: continuous batching admits eagerly when KV allows (so
// the true wait is often shorter), and preemption reorders (so it is sometimes
// longer). Record the actual scheduler policy in config.md.
func (h *Handler) tAdm(st instState, c Coeffs, residents []*resident) float64 {
	if st.BDec < h.params.MaxNumSeqs && st.Queue == 0 {
		return 0 // batch has room and nothing is waiting: admitted at once
	}
	// FCFS roll-forward: advance whole decode iterations until a slot frees.
	tIter := c.tIterDecode(st.BDec, st.KV, st.SPf)
	slotsNeeded := st.Queue + 1
	rems := make([]int64, 0, len(residents))
	for _, r := range residents {
		if rem := r.remainingSteps(); rem >= 0 {
			rems = append(rems, rem)
		}
	}
	sort.Slice(rems, func(i, j int) bool { return rems[i] < rems[j] })
	if slotsNeeded <= len(rems) {
		return float64(rems[slotsNeeded-1]) * tIter
	}
	if len(rems) > 0 {
		return float64(rems[len(rems)-1]) * tIter
	}
	return 0
}

// projectedLocalTTFT: admission, then the arrival's prefill chunks co-scheduled
// on the decode batch, then its prefill work, then token post-processing.
func (h *Handler) projectedLocalTTFT(tAdmD, nChunks, tIterD, wpLocal float64) float64 {
	return tAdmD + nChunks*tIterD + wpLocal + h.params.OutputTokenProcessingUs
}

// projectedDisaggTTFT ends at the decode pod's first client-visible token.
// decodeJoinUs already covers prefill admission + prefill work + KV transfer +
// decode admission, treated as CONCURRENT clocks from the routing instant (the
// max, not the sum) — this is the validated paper form, and getting it wrong by
// serializing inflates every remote candidate.
func (h *Handler) projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode float64) float64 {
	return decodeJoinUs + tIterFirstDecode + h.params.OutputTokenProcessingUs
}

// ─── The externality term ────────────────────────────────────────────────────

// reTiming holds the batch-level per-iteration decode times the completion model
// needs, plus the prefill-attention parameters the overlap window charges. Every
// co-resident in one decode batch shares one per-iteration time, so the timings
// are batch-level, not per-resident.
//
//	tIter0:       current batch B time — the baseline
//	tIterOverlap: while the arrival prefills co-scheduled on the decode batch.
//	              RETAINED FOR REFERENCE ONLY — unused under the exact overlap
//	              model the focal arm forces (see cLocalAfter).
//	tIterAfter:   after the arrival joins the batch — FULL B+1 re-timing, i.e.
//	              recompute tIterDecode(B+1, KV+Δkv, S_pf), not a marginal add
//
// cPf/cAttn/ap/ar/chunk parameterize the CAUSAL PREFILL-ATTENTION charge that the
// arrival's co-scheduled chunks impose on every decode co-resident. This is the
// paper's causal prefill-attention model (INFOCOM_REPRODUCIBILITY.md names it as a
// retained contribution) and it is load-bearing: it is quadratic in the processed
// prefill span, so on long prompts it dominates the local-placement charge.
type reTiming struct {
	tIter0       float64
	tIterOverlap float64
	tIterAfter   float64

	cPf   float64
	cAttn float64
	ap    float64 // uncached suffix tokens
	ar    float64 // full prompt length
	chunk float64 // per-step prefill token budget, = min(ap, ChunkTokens)
}

func (h *Handler) reTimingFor(c Coeffs, st instState, arrivalInputLen, ap int) reTiming {
	chunk := float64(h.chunkTokens(ap))
	return reTiming{
		tIter0:       c.tIterDecode(st.BDec, st.KV, st.SPf),
		tIterOverlap: c.tIterDecode(st.BDec, st.KV, st.SPf+int64(chunk)),
		tIterAfter:   c.tIterDecode(st.BDec+1, st.KV+int64(arrivalInputLen), st.SPf),
		cPf:          c.CPf,
		cAttn:        c.CAttn,
		ap:           float64(maxInt(ap, 0)),
		ar:           float64(arrivalInputLen),
		chunk:        chunk,
	}
}

// cBase is a resident's projected completion with rem steps left at the current
// (pre-arrival) per-iteration time.
func (rt reTiming) cBase(nowUs float64, rem int64) float64 {
	return nowUs + float64(rem)*rt.tIter0
}

// cLocalAfter is the resident's completion under LOCAL placement when the
// arrival waits admissionSteps baseline iterations before joining. Residents
// first run min(admissionSteps, rem) iterations undisturbed; of the surviving
// tail, the first min(nChunks, remaining) iterations overlap the arrival's
// prefill and the rest run at the B+1 re-timed rate.
//
// The overlap window runs at the BASELINE tIter0 plus the arrival's exact marginal
// prefill work — NOT at tIterOverlap. This is the "exact prefill overlap" form,
// which the focal arm forces unconditionally on both the local and disagg paths
// (sim/edpp.go:1707 and :1748 set VarExactPrefillOverlap = true), so the legacy
// tIterOverlap + c_attn·chunk²·overlap²/2 branch in sim/edpp_var.go:155-157 is
// unreachable for this policy and is deliberately not ported.
func (rt reTiming) cLocalAfter(nowUs float64, rem int64, admissionSteps, nChunks float64) float64 {
	pre := math.Min(math.Max(admissionSteps, 0), float64(rem))
	remaining := float64(rem) - pre
	overlap := math.Min(math.Max(nChunks, 0), remaining)
	return nowUs + pre*rt.tIter0 + overlap*rt.tIter0 +
		prefillMarginalWork(rt.cPf, rt.cAttn, rt.ap, rt.ar, rt.chunk, overlap) +
		(remaining-overlap)*rt.tIterAfter
}

// prefillMarginalWork is the exact E3 work (µs) added by the first `iterations`
// chunks of the arrival's uncached prefill, ported verbatim from
// sim/edpp_var.go:168. The known cached prefix is ar−ap, processed =
// min(ap, iterations·chunk), and the integrated causal charge is
//
//	CPf·processed + CAttn·processed·(cachedPrefix + processed/2)
//
// It EXCLUDES baseline iteration time — co-residents would pay that even if the
// arrival were absent, which is why cLocalAfter charges overlap·tIter0 separately.
func prefillMarginalWork(cPf, cAttn, ap, ar, chunk, iterations float64) float64 {
	if ap <= 0 || chunk <= 0 || iterations <= 0 {
		return 0
	}
	processed := math.Min(ap, iterations*chunk)
	cachedPrefix := math.Max(ar-ap, 0)
	return cPf*processed + cAttn*processed*(cachedPrefix+processed/2.0)
}

// cDisagg is the DISAGG mirror: the arrival prefills elsewhere and joins the
// decode batch after arrivalSteps iterations, so only the resident's tail is
// re-timed and there is no prefill-overlap window.
func (rt reTiming) cDisagg(nowUs float64, rem int64, arrivalSteps float64) float64 {
	pre := math.Min(math.Max(arrivalSteps, 0), float64(rem))
	return nowUs + pre*rt.tIter0 + (float64(rem)-pre)*rt.tIterAfter
}

// externality sums the SLO value DESTROYED among decode residents by this
// placement: Σ [ g(baseline completion) − g(placed completion) ].
//
// A resident whose remaining steps are unknown (rem < 0) contributes zero, never
// a guess — matching BLIS's censoring. Its realized TTFT is exact (the EPP saw
// the first token), so only the E2E conjunct moves.
//
// DEVIATION: BLIS's externality is a THREE-way breakdown — varPathBreakdown{decode,
// collocPrefill, prefillPool} (sim/edpp_var.go:~683) — and the focal arm enables all
// three (varCollocPrefill = true at sim/edpp.go:1708,1749). This port returns only
// the DECODE component:
//   - collocPrefill (occupants mid-prefill ON the decode node, whose first token this
//     placement delays) needs ds.RunningPrefill, which has no real signal — same root
//     cause as sPfFor. On a clean 1P2D fleet decode pods run no prefill and the term is
//     genuinely 0, but it is NOT 0 whenever this policy picks local prefill.
//   - prefillPool: see the note in Score's disagg branch.
//
// Both omissions under-price contention, so the argmin is biased toward whichever
// candidate carries more unpriced prefill work. Quantify before trusting the split.
func (h *Handler) externality(nowUs float64, residents []*resident, rt reTiming, admissionSteps, nChunks float64, disagg bool) float64 {
	var sum float64
	for _, r := range residents {
		rem := r.remainingSteps()
		if rem < 0 || !r.FirstTokenSet {
			continue
		}
		slo := h.sloFor(r.SLOClass)
		ttft := float64(r.FirstTokenUs - r.ArrivalUs)
		cb := rt.cBase(nowUs, rem)
		var cp float64
		if disagg {
			cp = rt.cDisagg(nowUs, rem, admissionSteps)
		} else {
			cp = rt.cLocalAfter(nowUs, rem, admissionSteps, nChunks)
		}
		gBase := compositeValue(slo, ttft, cb-float64(r.ArrivalUs))
		gPlaced := compositeValue(slo, ttft, cp-float64(r.ArrivalUs))
		sum += gBase - gPlaced
	}
	return sum
}

// ─── The score ───────────────────────────────────────────────────────────────

// candidateScore is the per-candidate breakdown, mirroring BLIS's
// sloJointCandidateScore. The fields are traced per candidate so the argmin
// identity can be verified offline — the sim protocol gates on
// `chosen_argmin_trace_exact`, and the real run should too.
type candidateScore struct {
	DecodeKey   string
	PrefillKey  string // "" for local
	Externality float64
	OwnGood     float64
	NetGoodCost float64
	Capacity    float64
	Total       float64
}

// Score implements the focal contract:
//
//	total = V * (externality − ownGood) + capacity
//
// with NO historical TTFT/ITL deficit term, no transfer residue, and no
// per-decision normalization. In the focal arm the capacity term is DISABLED
// (params.NoCapacity, i.e. --edpp-slo-externality-no-capacity): the
// occupancy-capacity controller was tested and abandoned in the closeout study.
// V stays at 8.0 regardless.
func (h *Handler) Score(nowUs float64, req *arrival, d instState, p *instState) candidateScore {
	thetaD := h.coeffsFor(d.GPUType)
	residents := h.residents.snapshot(d.Key)
	slo := h.sloFor(req.SLOClass)

	tIterD := thetaD.tIterDecode(d.BDec, d.KV, d.SPf)
	tIterFirstDecode := thetaD.tIterDecode(d.BDec+1, d.KV+int64(req.InputLen), d.SPf)
	tAdmD := h.tAdm(d, thetaD, residents)

	sc := candidateScore{DecodeKey: d.Key}

	if p == nil {
		// ── LOCAL: prefill and decode co-resident on d ──
		ap := req.UncachedSuffixTokens // TRANSLATE: from the prefix-cache match info
		nChunks, _ := h.chunkTerms(ap)
		wpLocal := thetaD.wp(maxInt(ap, 0), req.InputLen)
		tHat := h.projectedLocalTTFT(tAdmD, nChunks, tIterD, wpLocal)

		rt := h.reTimingFor(thetaD, d, req.InputLen, ap)
		// Ceil: admission happens on an iteration boundary, so the wait is a WHOLE
		// number of baseline decode steps (sim/edpp_var.go:865).
		admissionSteps := math.Ceil(tAdmD / math.Max(rt.tIter0, 1))
		sc.Externality = h.externality(nowUs, residents, rt, admissionSteps, nChunks, false)
		sc.OwnGood = goodSelf(slo, tHat, rt.tIterAfter, req.NOutEstimate)
	} else {
		// ── DISAGG: decode on d, prefill on p ──
		sc.PrefillKey = p.Key
		thetaP := h.coeffsFor(p.GPUType) // p's OWN physics, not d's
		ap := req.UncachedSuffixTokens
		nChunksP, _ := h.chunkTerms(ap)
		wpP := thetaP.wp(maxInt(ap, 0), req.InputLen)
		tIterP := thetaP.tIterPrefill(p.SPf)
		tAdmP := h.tAdm(*p, thetaP, h.residents.snapshot(p.Key))

		prefillCompletionUs := tAdmP + nChunksP*tIterP + wpP
		remoteLeadUs := prefillCompletionUs + h.cXferUs(req)
		// CONCURRENT clocks: remote preparation and decode-queue drain both run
		// from the routing instant, so take the max. Summing them double-charges
		// the wait and biases every remote candidate.
		decodeJoinUs := math.Max(remoteLeadUs, tAdmD)
		tHat := h.projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode)

		rt := h.reTimingFor(thetaD, d, req.InputLen, ap)
		arrivalSteps := math.Ceil(decodeJoinUs / math.Max(rt.tIter0, 1))
		sc.Externality = h.externality(nowUs, residents, rt, arrivalSteps, 0, true)
		// DEVIATION: BLIS also prices the externality imposed on PREFILL-POOL
		// co-residents (varPrefillDisagg). That term needs S_pf on the prefill
		// pool, which has no real signal (see sPfFor), so it degrades toward
		// zero here. The paper calls the decode-side asymmetry dominant, but this
		// systematically under-prices remote prefill — expect a higher remote
		// fraction on the real cluster than the sim's, and report it.
		sc.OwnGood = goodSelf(slo, tHat, rt.tIterAfter, req.NOutEstimate)
	}

	if h.params.NoExternality {
		sc.Externality = 0 // ablation arm: causalextnoext
	}
	if h.params.NoOwnGood {
		sc.OwnGood = 0
	}
	sc.NetGoodCost = sc.Externality - sc.OwnGood
	if !h.params.NoCapacity {
		// TRANSLATE: the occupancy-capacity queues (sloCapacityQueue /
		// sloCapacityTerm). NOT NEEDED for the focal arm, which sets
		// NoCapacity=true. Port only if the occupancy controller is revived.
		sc.Capacity = 0
	}
	sc.Total = h.params.V*sc.NetGoodCost + sc.Capacity
	return sc
}

// PickJoint enumerates every joint candidate and returns the ARGMIN.
//
// Ties are broken by (decode key, prefill key) lexicographic order so the choice
// is deterministic and the traced argmin is reproducible. BLIS sorts snapshots by
// ID for the same reason.
func (h *Handler) PickJoint(nowUs float64, req *arrival, decodes, prefills []instState) (candidateScore, []candidateScore) {
	traced := make([]candidateScore, 0, len(decodes)*(1+len(prefills)))
	best := candidateScore{Total: math.Inf(1)}
	for _, d := range decodes {
		cands := []candidateScore{h.Score(nowUs, req, d, nil)}
		for i := range prefills {
			cands = append(cands, h.Score(nowUs, req, d, &prefills[i]))
		}
		for _, c := range cands {
			traced = append(traced, c)
			if c.Total < best.Total ||
				(c.Total == best.Total && (c.DecodeKey < best.DecodeKey ||
					(c.DecodeKey == best.DecodeKey && c.PrefillKey < best.PrefillKey))) {
				best = c
			}
		}
	}
	return best, traced
}

// ─── Plugin plumbing ─────────────────────────────────────────────────────────

// arrival is the deciding request's routing-time view.
type arrival struct {
	RequestID string
	InputLen  int
	// UncachedSuffixTokens is a_p: input tokens NOT already in the target's
	// prefix cache. TRANSLATE: llm-d exposes this via the prefix-cache match
	// info (attrprefix.PrefixCacheMatchInfo — use CachedBlockCount() *
	// BlockSizeTokens(), the UNWEIGHTED count, not the tier-weighted score; the
	// stock prefix decider has a comment explaining why the weighted score
	// over-estimates the uncached suffix). It is PER-CANDIDATE, so strictly
	// a_p(d) — this reference port carries one value for brevity.
	UncachedSuffixTokens int
	// NOutEstimate is N̂_out. MUST be a censored per-class estimate, never the
	// true output length. TRANSLATE: source it from a per-SLO-class running mean
	// maintained by the EPP, matching BLIS's censored estimator.
	NOutEstimate float64
	SLOClass     string
}

// Params is the JSON block under this plugin's `parameters:` in the EPP config.
// See config.md "llm-d EPP Configuration — treatment arm" for the populated form.
type Params struct {
	V      float64 `json:"V"`
	Kernel string  `json:"kernel"` // "composite" — the only kernel this arm uses

	// Per-class SLO targets. Single-class ("standard") in every campaign cell.
	SLO        SLO            `json:",inline"`
	SLOClasses map[string]SLO `json:"sloClasses"`

	// Ablation switches. Focal arm: NoCapacity=true, others false.
	NoExternality bool `json:"noExternality"`
	NoOwnGood     bool `json:"noOwnGood"`
	NoCapacity    bool `json:"noCapacity"`

	// Rollout configuration.
	TAdmEstimator           string  `json:"tadmEstimator"` // "rollforward"
	XferSizeAware           bool    `json:"xferSizeAware"`
	ChunkTokens             int     `json:"chunkTokens"` // = vLLM --max-num-batched-tokens
	BlockSize               int     `json:"blockSize"`   // = vLLM --block-size
	MaxNumSeqs              int     `json:"maxNumSeqs"`  // = vLLM --max-num-seqs
	CXferUsPerToken         float64 `json:"cXferUsPerToken"`
	OutputTokenProcessingUs float64 `json:"outputTokenProcessingUs"`

	// Coefficients keyed by GPU type, plus how to discover an endpoint's type.
	Coeffs         map[string]Coeffs `json:"coeffs"`
	DefaultGPUType string            `json:"defaultGpuType"`
	// TRANSLATE: attribute key carrying nvidia.com/gpu.product, and the mapping
	// from label value to Coeffs key.
	GPUTypeAttribute string            `json:"gpuTypeAttribute"`
	GPUTypeByLabel   map[string]string `json:"gpuTypeByLabel"`
}

// Handler is the joint ProfileHandler.
type Handler struct {
	typedName plugin.TypedName
	params    Params
	residents *residentTable
}

func (h *Handler) TypedName() plugin.TypedName { return h.typedName }

func (h *Handler) sloFor(class string) SLO {
	if s, ok := h.params.SLOClasses[class]; ok {
		return s
	}
	return h.params.SLO
}

// chunkTokens is the per-step prefill token budget actually used for an uncached
// suffix of ap tokens: min(ap, --max-num-batched-tokens).
//
// NOT simply ChunkTokens. A prompt shorter than the engine budget is prefilled in
// ONE chunk of ap tokens, so charging the full budget over-states the co-scheduled
// prefill span. BLIS recomputes this at every call site — sim/edpp.go chunkTerms,
// and "chunkLoc"/"chunkP" in sim/edpp_var.go varJointCandidateBreakdownCore.
func (h *Handler) chunkTokens(ap int) int {
	if ap <= 0 {
		return 0
	}
	if h.params.ChunkTokens > 0 && h.params.ChunkTokens < ap {
		return h.params.ChunkTokens
	}
	return ap
}

// chunkTerms returns (number of prefill chunks, tokens per chunk) for an
// uncached suffix of ap tokens under the engine's chunk budget.
func (h *Handler) chunkTerms(ap int) (nChunks, chunk float64) {
	c := h.chunkTokens(ap)
	if c <= 0 {
		return 0, 0
	}
	chunk = float64(c)
	nChunks = math.Ceil(float64(ap) / chunk)
	return nChunks, chunk
}

// cXferUs is the KV-transfer cost for this request, size-aware.
//
// TRANSLATE: MEASURE THIS on the target interconnect — it is the price of every
// remote candidate and inheriting a simulated value silently retunes the whole
// local-vs-remote balance. Drive a fixed-size prefill through the disaggregated
// path and time the transfer.
func (h *Handler) cXferUs(req *arrival) float64 {
	if !h.params.XferSizeAware {
		return h.params.CXferUsPerToken
	}
	return h.params.CXferUsPerToken * float64(req.InputLen)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ─── Adapters the translate team must supply ─────────────────────────────────
//
// These stand in for llm-d accessors whose exact names must be confirmed on the
// pinned checkout. They are deliberately NOT guessed at call sites, so that a
// wrong guess fails to compile here rather than silently mis-scoring.

// metricsView mirrors the subset of datalayer.Metrics this policy reads.
// TRANSLATE: replace with the real *datalayer.Metrics.
type metricsView struct {
	RunningRequestsSize     int
	WaitingQueueSize        int
	KVCacheUsagePercent     float64
	KvCacheMaxTokenCapacity int
	CacheBlockSize          int
}

// TRANSLATE: real signature is ep.GetMetrics() *datalayer.Metrics.
func (h *Handler) getMetrics(ep fwksched.Endpoint) *metricsView { panic("TRANSLATE") }

// TRANSLATE: stable per-endpoint key. GetMetadata() gives Address/Port; a
// NamespacedName is more stable across pod restarts than host:port.
func endpointKey(ep fwksched.Endpoint) string { panic("TRANSLATE") }

// TRANSLATE: endpoint attribute lookup — ep.Get(key) in v0.9.0 returns (any, bool).
func endpointAttr(ep fwksched.Endpoint, key string) (string, bool) { panic("TRANSLATE") }

// Unused-import guards for the reference port.
var (
	_ = context.Background
	_ = time.Now
)
