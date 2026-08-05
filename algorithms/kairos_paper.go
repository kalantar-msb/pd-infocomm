// Package kairospaper is the sim2real ALGORITHM SOURCE for the INFOCOM 2027
// study's published-comparator arm. It is a faithful reference port of
//
//	sim/edpp_kairos.go  decideKairosPaper, kairosDiscreteDeflectTTFT,
//	                    kairosPaperPrefillTTFT, kairosResidentTBTTarget
//	                    (--edpp-rule kairos-paper --kairos-alpha 1.3 --kairos-beta 1.0)
//
// (fork vishakha-ramani/inference-sim @ 871b169b, branch infocom-implementation),
// which itself implements the printed decision rule of
//
//	"Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving",
//	arXiv:2607.02043 — Algorithm 1.
//
// ─── IMPORTANT: this is a DIFFERENT MECHANISM, not a different score ─────────
// The focal and least-TTFT arms choose WHERE to prefill among the prefill pool.
// Kairos instead asks whether to DEFLECT the prefill ONTO A DECODE NODE, chunked
// finely enough that no chunk pushes resident time-between-tokens past its
// budget. So "disaggregate = false" in Kairos means "prefill on the decode node
// with a protective chunk schedule", which is the opposite polarity from the
// other arms. Getting this backwards silently inverts the comparator.
//
// ─── The rule (Algorithm 1) ──────────────────────────────────────────────────
//  1. Estimate TTFT via the PREFILL POOL: queue wait + prefill execution
//     (printed Equation 1; no KV-transfer term, no decode admission).
//  2. For each decode node with no prefill already in flight, greedily build a
//     chunk schedule from a DISCRETE descending candidate list, taking the
//     LARGEST chunk whose step time fits the resident TBT budget β·τ_resident.
//     Sum the step times to get the deflected TTFT.
//  3. Deflect to the best decode node iff BOTH
//     - margin gate:  ttft_deflect <= α · ttft_prefill_pool     (α = 1.3)
//     - SLO gate:     ttft_deflect <= τ_TTFT of the ARRIVING request
//     Otherwise send it to the prefill pool.
//
// Two details are easy to get wrong and both are load-bearing:
//   - The TBT budget uses the STRICTEST TBT target among the decode node's CURRENT
//     RESIDENTS, not the arriving request's class. The constraint protects
//     incumbents, not the newcomer.
//   - A decode node is ineligible if it already has any prefill in flight
//     (ResidentPrefillTokens > 0 || PrefillTokensAhead > 0). At most one deflected
//     prefill per decode node at a time.
//
// ─── Simulation result ───────────────────────────────────────────────────────
// Worst-cell regret 0.0217 homogeneous / 0.1050 heterogeneous, versus the focal
// arm's 0.0100 / 0.0031 (sim_results/main/confirmation_result.json). Paired delta
// focal-minus-Kairos across all 18 cells: +0.0179 goodput, 95% CI
// [+0.0104, +0.0253]. Kairos is the strongest published comparator on the
// homogeneous fleet and degrades on the heterogeneous one — its TBT-safety
// constraint is hardware-aware through θ_i, but its deflection decision has no
// notion that the two decoders differ in throughput.
//
// ─── Fidelity note specific to this arm ──────────────────────────────────────
// Algorithm 1 requires the engine to EXECUTE the computed chunk schedule. BLIS
// honours it (req.PrefillChunkSchedule). Real vLLM does not accept a per-request
// chunk schedule — chunked prefill is governed globally by
// --max-num-batched-tokens. So the deflected path cannot be executed faithfully;
// the schedule can only be used to DECIDE. This is the single largest declared
// deviation of this arm and it must appear in the write-up: the comparator is
// handicapped relative to its paper, in a direction that makes it look worse than
// it would with engine support. Do not report it as a clean refutation of Kairos.
package kairospaper

import (
	"math"
)

// HandlerType is the plugin `type` string in EndpointPickerConfig `plugins:`.
const HandlerType = "kairos-paper-handler"

// DefaultChunkCandidates is the paper's profiled chunk set, descending.
// Algorithm 1 consumes a descending list and takes the largest safe entry.
//
// TRANSLATE: confirm against kairosPaperChunkCandidates in sim/edpp_kairos.go on
// the pinned fork commit before freezing — the arm's behaviour is sensitive to
// this list, and a silently different set makes the comparator non-comparable to
// the sim result quoted above.
var DefaultChunkCandidates = []float64{2048, 1024, 512, 256, 128}

// Params is the JSON block under this plugin's `parameters:`.
type Params struct {
	// Alpha is the deflection margin gate. 1.3 in the campaign
	// (--kairos-alpha 1.3): deflect only if the decode-node TTFT is within 1.3x
	// the prefill-pool TTFT.
	Alpha float64 `json:"alpha"`
	// Beta scales the resident TBT target into a per-step budget. 1.0 in paper
	// mode (--kairos-beta 1.0). The earlier "adapted" identity used 0.5; that is a
	// DIFFERENT arm and is not transferred.
	Beta float64 `json:"beta"`

	// ChunkCandidates descending; entries above ChunkCap are dropped.
	ChunkCandidates []float64 `json:"chunkCandidates"`
	// ChunkCap = vLLM --max-num-batched-tokens.
	ChunkCap float64 `json:"chunkCap"`

	// TauTTFTUs is the ARRIVING request's TTFT target — the SLO gate.
	TauTTFTUs float64 `json:"tauTtftUs"`
	// TauITLUsByClass supplies resident TBT targets. The strictest among a node's
	// residents becomes that node's budget.
	TauITLUsByClass map[string]float64 `json:"tauItlUsByClass"`
	DefaultTauITLUs float64            `json:"defaultTauItlUs"`

	MaxSteps int `json:"maxSteps"` // 4096 in the sim; a runaway guard, not a tunable

	// Per-GPU coefficients — the same block as the other arms. Kairos uses this
	// simulator's trained-physics coefficients too (edpp_kairos.go header).
	Coeffs           map[string]Coeffs `json:"coeffs"`
	DefaultGPUType   string            `json:"defaultGpuType"`
	GPUTypeAttribute string            `json:"gpuTypeAttribute"`
	GPUTypeByLabel   map[string]string `json:"gpuTypeByLabel"`
}

// Coeffs mirrors the shared latency-law coefficients.
// TRANSLATE: reference the shared type from causal_slo_externality.go.
type Coeffs struct {
	AlphaD float64 `json:"alphaUs"`
	AlphaP float64 `json:"alphaPUs"`
	C0     float64 `json:"c0UsPerReq"`
	C1     float64 `json:"c1UsPerToken"`
	CPf    float64 `json:"cPfUsPerToken"`
	CAttn  float64 `json:"cAttnUsPerUnit"`
}

func (c Coeffs) tIterDecode(bDec int, kv, sPf int64) float64 {
	return c.AlphaD + c.C0*float64(bDec) + c.C1*float64(kv) + c.CPf*float64(sPf)
}

// stepPrefill is the prefill-node step time for a chunk of chi tokens attending
// over a resident context of k tokens (edpp_kairos.go kairosStepPrefill).
func (c Coeffs) stepPrefill(chi, k float64) float64 {
	return c.AlphaP + c.CPf*chi + c.CAttn*chi*(k+chi/2)
}

// discreteCandidates filters the paper's profiled set to what the engine can
// execute. If the engine is configured below the smallest profiled point, fall
// back to the cap itself — explicit and deterministic, matching the sim.
func discreteCandidates(all []float64, chunkCap float64) []float64 {
	out := make([]float64, 0, len(all))
	for _, candidate := range all {
		if chunkCap <= 0 || candidate <= chunkCap {
			out = append(out, candidate)
		}
	}
	if len(out) == 0 && chunkCap > 0 {
		out = append(out, chunkCap)
	}
	return out
}

// DeflectTTFT implements Algorithm 1's greedy LargestSafe search on one decode
// node: repeatedly take the largest candidate chunk whose step time fits the TBT
// budget, accumulating elapsed time until the prompt is consumed.
//
// Returns ok=false when no candidate fits the budget at some step (the node
// cannot host this prefill safely) or the step cap is hit. A false here makes the
// node ineligible; it does not fall back to a smaller-than-profiled chunk.
//
// The final chunk is shortened to the remaining prompt tokens — the executor
// cannot process padding beyond the prompt.
func (h *Handler) DeflectTTFT(c Coeffs, bDec int, kv int64, tokens, tbt float64, candidates []float64) (elapsed float64, schedule []float64, ok bool) {
	if tokens <= 0 {
		return 0, nil, true
	}
	if len(candidates) == 0 {
		return 0, nil, false
	}
	maxSteps := h.params.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 4096
	}
	var done float64
	schedule = make([]float64, 0, int(math.Ceil(tokens/candidates[len(candidates)-1])))
	for step := 0; step < maxSteps && done < tokens; step++ {
		// The base is the decode node's CURRENT iteration cost with the prefill's
		// progress folded into KV. Deflected prefill runs alongside decode, so the
		// resident batch pays it — that is precisely what the TBT budget bounds.
		base := c.tIterDecode(bDec, kv+int64(done), 0)
		remaining := tokens - done
		chosen := 0.0
		for _, candidate := range candidates {
			chi := math.Min(candidate, remaining)
			stepTime := base + c.CPf*chi + c.CAttn*chi*(done+chi/2)
			if stepTime <= tbt {
				chosen = chi
				break // descending list => first fit is the largest safe chunk
			}
		}
		if chosen <= 0 {
			return 0, nil, false
		}
		elapsed += base + c.CPf*chosen + c.CAttn*chosen*(done+chosen/2)
		done += chosen
		schedule = append(schedule, chosen)
	}
	if done < tokens {
		return 0, nil, false
	}
	return elapsed, schedule, true
}

// PrefillPoolTTFT is printed Equation 1: queue wait + prefill execution, over the
// prefill pool. No KV-transfer term and no decode-admission term — the paper's
// estimator omits both, and adding them would make this the "adapted" identity
// rather than the published one.
//
// Kairos does not account for prefix-cache residency: Algorithm 1 takes the full
// prompt length. Do not substitute the uncached suffix here even though the other
// arms use it.
func (h *Handler) PrefillPoolTTFT(req *arrival, prefills []instState) (bestTTFT float64, bestPrefillKey string) {
	tokens := float64(req.InputLen)
	if tokens <= 0 {
		return math.Inf(1), ""
	}
	bestTTFT = math.Inf(1)
	for _, ps := range prefills {
		theta := h.coeffsFor(ps.GPUType)
		chi := h.params.ChunkCap
		if chi <= 0 || chi > tokens {
			chi = tokens
		}
		// PrefillTokensAhead is the actual remaining prompt-token total queued
		// ahead of this arrival on that node.
		//
		// TRANSLATE: no vLLM metric exposes this. Derive it from the EPP's shadow
		// resident table (sum of remaining prompt tokens for requests it routed to
		// this prefill node that have not reached first token). WaitingQueueSize x
		// average-prompt-length is the crude fallback the sim explicitly moved away
		// from — if you use it, declare it.
		sumL := float64(ps.PrefillTokensAhead)
		queueWait := 0.0
		if sumL > 0 {
			queueWait = (sumL / chi) * theta.stepPrefill(chi, sumL/2)
		}
		exec := 0.0
		for done := 0.0; done < tokens; done += chi {
			stepChi := math.Min(chi, tokens-done)
			exec += theta.stepPrefill(stepChi, done)
		}
		if ttft := queueWait + exec; ttft < bestTTFT {
			bestTTFT, bestPrefillKey = ttft, ps.Key
		}
	}
	return bestTTFT, bestPrefillKey
}

// residentTBTTarget is the strictest TBT target among the decode node's current
// residents — the incumbents the safety constraint protects. The arriving
// request's class is used ONLY when the node has no decode residents.
func (h *Handler) residentTBTTarget(st instState, arrivingClass string) float64 {
	residents := h.residents.snapshot(st.Key)
	strictest := math.Inf(1)
	for _, r := range residents {
		tau, ok := h.params.TauITLUsByClass[r.SLOClass]
		if !ok {
			tau = h.params.DefaultTauITLUs
		}
		if tau > 0 && tau < strictest {
			strictest = tau
		}
	}
	if math.IsInf(strictest, 1) {
		if tau, ok := h.params.TauITLUsByClass[arrivingClass]; ok {
			return tau
		}
		return h.params.DefaultTauITLUs
	}
	return strictest
}

// Decision is the arm's output. Note the polarity discussed in the package
// header: Deflect==true means prefill runs ON the decode node.
type Decision struct {
	Deflect       bool
	DecodeKey     string    // set when Deflect
	PrefillKey    string    // set when !Deflect
	ChunkSchedule []float64 // advisory only on real vLLM — see package header
	DeflectTTFTUs float64
	PrefillTTFTUs float64
	MarginPassed  bool
	SLOGatePassed bool
	Skip          string // "empty-prompt" | "no-prefill-path" | ""
}

// Decide implements decideKairosPaper.
func (h *Handler) Decide(req *arrival, decodes, prefills []instState) Decision {
	if req.InputLen == 0 {
		return Decision{Skip: "empty-prompt"}
	}
	candidates := discreteCandidates(h.params.ChunkCandidates, h.params.ChunkCap)
	ttftPrefill, prefillKey := h.PrefillPoolTTFT(req, prefills)

	bestTTFT := math.Inf(1)
	bestDecode := ""
	var bestSchedule []float64
	for _, ds := range decodes {
		// At most one deflected prefill in flight per decode node.
		if ds.SPf > 0 || ds.PrefillTokensAhead > 0 {
			continue
		}
		tbtBudget := h.params.Beta * h.residentTBTTarget(ds, req.SLOClass)
		theta := h.coeffsFor(ds.GPUType)
		t, schedule, ok := h.DeflectTTFT(theta, ds.BDec, ds.KV, float64(req.InputLen), tbtBudget, candidates)
		if !ok {
			continue
		}
		if t < bestTTFT {
			bestTTFT, bestDecode, bestSchedule = t, ds.Key, schedule
		}
	}

	marginPassed := bestDecode != "" && bestTTFT <= h.params.Alpha*ttftPrefill
	sloPassed := bestTTFT <= h.params.TauTTFTUs
	if marginPassed && sloPassed {
		return Decision{
			Deflect:       true,
			DecodeKey:     bestDecode,
			ChunkSchedule: bestSchedule,
			DeflectTTFTUs: bestTTFT,
			PrefillTTFTUs: ttftPrefill,
			MarginPassed:  true,
			SLOGatePassed: true,
		}
	}
	if prefillKey == "" || math.IsInf(ttftPrefill, 1) {
		return Decision{
			DeflectTTFTUs: bestTTFT,
			PrefillTTFTUs: ttftPrefill,
			MarginPassed:  marginPassed,
			SLOGatePassed: sloPassed,
			Skip:          "no-prefill-path",
		}
	}
	return Decision{
		PrefillKey:    prefillKey,
		DeflectTTFTUs: bestTTFT,
		PrefillTTFTUs: ttftPrefill,
		MarginPassed:  marginPassed,
		SLOGatePassed: sloPassed,
	}
}

// ─── Shared-type references ──────────────────────────────────────────────────
//
// TRANSLATE: collapse into the shared types from causal_slo_externality.go.
// PrefillTokensAhead is the one field this arm needs that the others do not.

type arrival struct {
	RequestID string
	InputLen  int
	SLOClass  string
}

type instState struct {
	Key     string
	GPUType string
	BDec    int
	KV      int64
	SPf     int64
	Queue   int
	// PrefillTokensAhead: remaining prompt tokens queued ahead on this node.
	// TRANSLATE: derive from the shadow resident table; no vLLM metric exists.
	PrefillTokensAhead int64
}

type resident struct {
	SLOClass string
}

type residentTable interface {
	snapshot(endpointKey string) []*resident
}

type Handler struct {
	params    Params
	residents residentTable
}

func (h *Handler) coeffsFor(gpuType string) Coeffs {
	if c, ok := h.params.Coeffs[gpuType]; ok {
		return c
	}
	return h.params.Coeffs[h.params.DefaultGPUType]
}
