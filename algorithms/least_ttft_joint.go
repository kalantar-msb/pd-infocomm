// Package leastttftjoint is the sim2real ALGORITHM SOURCE for the INFOCOM 2027
// study's least-projected-TTFT comparator arm. It is a faithful reference port of
//
//	sim/edpp.go  jointCandidateTTFT   (--edpp-rule least-ttft --edpp-joint
//	                                   --edpp-ttft-overlap-aware)
//
// (fork vishakha-ramani/inference-sim @ 871b169b, branch infocom-implementation).
//
// ─── What this policy does ───────────────────────────────────────────────────
// The same joint (decode, prefill) enumeration and the same rollout as the focal
// arm, with the objective replaced by the arrival's OWN projected time-to-first-
// token. Nothing else: no resident externality, no capacity drift, no SLO virtual
// queues, no transfer penalty beyond the transfer latency already inside the
// disaggregated TTFT.
//
//	score(d, p) = projected TTFT of the arrival under (d, p)      -> argmin
//
// ─── Why this arm exists ─────────────────────────────────────────────────────
// It isolates the value of the EXTERNALITY term. Both arms share the latency law,
// the rollout, the joint enumeration, and per-candidate coefficients — so this is
// a hardware-AWARE greedy-selfish baseline, not a strawman. The BLIS source calls
// it "the fair least-TTFT the reviewer asks for" for exactly that reason.
//
// Simulation result (sim_results/main/confirmation_result.json): worst-cell regret
// 0.0542 homogeneous / 0.0475 heterogeneous, against the focal arm's 0.0100 /
// 0.0031. Paired delta focal-minus-this across all 18 cells: +0.0187 goodput,
// 95% CI [+0.0142, +0.0231]. So on simulated traffic the externality term is worth
// ~1.9 goodput points and the interval excludes zero.
//
// ─── Implementation note ─────────────────────────────────────────────────────
// This arm SHARES the focal arm's machinery. In the translated plugin it should be
// the same Go package and the same container image, selected by config — the
// difference is one objective function. Emitting a second independent
// implementation would risk the two arms diverging in the rollout, which would
// confound the comparison the arm exists to make.
//
// The shared pieces (Coeffs, SLO, instState, residentTable, tAdm,
// projectedLocalTTFT, projectedDisaggTTFT, chunkTerms, cXferUs, gpuTypeFor) all
// live in causal_slo_externality.go and are referenced here by name rather than
// duplicated.
package leastttftjoint

import (
	"math"
)

// HandlerType is the plugin `type` string in EndpointPickerConfig `plugins:`.
//
// Like the focal arm this is a ProfileHandler, not a Scorer — the objective is
// still evaluated over the CROSS PRODUCT of decode and prefill candidates, so
// llm-d's decode-first disagg-profile-handler cannot express it either. See
// causal_slo_externality.go's HandlerType comment for the joint-enumeration
// TRANSLATE question, which applies verbatim.
const HandlerType = "least-ttft-joint-handler"

// Params is the JSON block under this plugin's `parameters:`.
//
// The τ targets are still required even though the objective does not reference
// an SLO value: the admission estimator's remaining-steps model is τ-derived, so
// dropping them would change the rollout and break the like-for-like comparison
// with the focal arm.
type Params struct {
	// OverlapAware corresponds to --edpp-ttft-overlap-aware: charge the arrival's
	// prefill chunks against the decode batch's iteration time when prefill is
	// local. Always true in the campaign.
	OverlapAware bool `json:"overlapAware"`

	TauTTFTUs float64 `json:"tauTtftUs"`
	TauITLUs  float64 `json:"tauItlUs"`

	// Rollout + latency-law configuration. IDENTICAL to the focal arm's block,
	// including the per-GPU coeffs map — see config.md "Latency-law coefficients".
	// TRANSLATE: share one Params type across both arms rather than copying, so a
	// coefficient or chunk-budget edit cannot land in one arm and not the other.
	TAdmEstimator           string  `json:"tadmEstimator"`
	XferSizeAware           bool    `json:"xferSizeAware"`
	ChunkTokens             int     `json:"chunkTokens"`
	BlockSize               int     `json:"blockSize"`
	MaxNumSeqs              int     `json:"maxNumSeqs"`
	CXferUsPerToken         float64 `json:"cXferUsPerToken"`
	OutputTokenProcessingUs float64 `json:"outputTokenProcessingUs"`

	Coeffs           map[string]coeffsRef `json:"coeffs"`
	DefaultGPUType   string               `json:"defaultGpuType"`
	GPUTypeAttribute string               `json:"gpuTypeAttribute"`
	GPUTypeByLabel   map[string]string    `json:"gpuTypeByLabel"`
}

// coeffsRef mirrors causalsloexternality.Coeffs. TRANSLATE: replace with a direct
// reference to the shared type; it is duplicated here only so this reference file
// reads standalone.
type coeffsRef struct {
	AlphaD float64 `json:"alphaUs"`
	AlphaP float64 `json:"alphaPUs"`
	C0     float64 `json:"c0UsPerReq"`
	C1     float64 `json:"c1UsPerToken"`
	CPf    float64 `json:"cPfUsPerToken"`
	CAttn  float64 `json:"cAttnUsPerUnit"`
}

func (c coeffsRef) tIterDecode(bDec int, kv, sPf int64) float64 {
	return c.AlphaD + c.C0*float64(bDec) + c.C1*float64(kv) + c.CPf*float64(sPf)
}

func (c coeffsRef) tIterPrefill(sPf int64) float64 {
	return c.AlphaP + c.CPf*float64(sPf)
}

func (c coeffsRef) wp(ap, ar int) float64 {
	a, r := float64(ap), float64(ar)
	return c.CPf*a + c.CAttn*a*(r-a/2.0)
}

// CandidateTTFT is the objective: the arrival's own forward TTFT (µs) for one
// candidate — local (p == nil) or disaggregated (decode on d, prefill on p).
//
// The arithmetic must reproduce the focal arm's tHatLocal / tHatDisagg EXACTLY.
// BLIS asserts this as an invariant (INV-6, "the arithmetic reproduces exactly
// jointCandidateCost's tHatLocal/tHatDisagg"), and it is what makes the paired
// comparison attributable to the externality term alone. TRANSLATE: add a unit
// test that scores the same (request, snapshot) through both arms and asserts the
// projected TTFTs are bit-identical.
func (h *Handler) CandidateTTFT(req *arrivalRef, d instStateRef, p *instStateRef) float64 {
	thetaD := h.coeffsFor(d.GPUType)
	residents := h.residents.snapshot(d.Key)

	tIterD := thetaD.tIterDecode(d.BDec, d.KV, d.SPf)
	tIterFirstDecode := thetaD.tIterDecode(d.BDec+1, d.KV+int64(req.InputLen), d.SPf)
	tAdmD := h.tAdm(d, thetaD, residents)

	if p == nil {
		// ── local: prefill+decode co-resident on d, so prefill uses d's θ_i ──
		ap := req.UncachedSuffixTokens
		nChunks, _ := h.chunkTerms(ap)
		wpLocal := thetaD.wp(maxInt(ap, 0), req.InputLen)
		if !h.params.OverlapAware {
			// Without overlap awareness the prefill chunks do not ride the decode
			// batch's iteration time. Campaign always sets OverlapAware=true; this
			// branch exists only to make the flag's meaning explicit.
			return tAdmD + wpLocal + h.params.OutputTokenProcessingUs
		}
		return tAdmD + nChunks*tIterD + wpLocal + h.params.OutputTokenProcessingUs
	}

	// ── disagg: decode on d, prefill on p, each under its OWN θ_i ──
	thetaP := h.coeffsFor(p.GPUType)
	ap := req.UncachedSuffixTokens
	nChunksP, _ := h.chunkTerms(ap)
	wpP := thetaP.wp(maxInt(ap, 0), req.InputLen)
	tIterP := thetaP.tIterPrefill(p.SPf)
	tAdmP := h.tAdm(*p, thetaP, h.residents.snapshot(p.Key))

	prefillCompletionUs := tAdmP + nChunksP*tIterP + wpP
	remoteLeadUs := prefillCompletionUs + h.cXferUs(req)
	// Concurrent clocks from the routing instant — the max, not the sum. Same
	// validated paper form as the focal arm.
	decodeJoinUs := math.Max(remoteLeadUs, tAdmD)
	return decodeJoinUs + tIterFirstDecode + h.params.OutputTokenProcessingUs
}

// PickJoint returns the joint candidate minimizing the arrival's projected TTFT.
// Ties break on (decode key, prefill key) for determinism, matching the focal arm.
func (h *Handler) PickJoint(req *arrivalRef, decodes, prefills []instStateRef) (bestDecode, bestPrefill string, bestTTFT float64) {
	bestTTFT = math.Inf(1)
	for _, d := range decodes {
		type cand struct {
			dk, pk string
			ttft   float64
		}
		cands := []cand{{d.Key, "", h.CandidateTTFT(req, d, nil)}}
		for i := range prefills {
			cands = append(cands, cand{d.Key, prefills[i].Key, h.CandidateTTFT(req, d, &prefills[i])})
		}
		for _, c := range cands {
			if c.ttft < bestTTFT ||
				(c.ttft == bestTTFT && (c.dk < bestDecode ||
					(c.dk == bestDecode && c.pk < bestPrefill))) {
				bestDecode, bestPrefill, bestTTFT = c.dk, c.pk, c.ttft
			}
		}
	}
	return bestDecode, bestPrefill, bestTTFT
}

// ─── Shared-type references ──────────────────────────────────────────────────
//
// TRANSLATE: every declaration below is a placeholder for the corresponding type
// in causal_slo_externality.go. Collapse them — do not maintain two copies.

type arrivalRef struct {
	RequestID            string
	InputLen             int
	UncachedSuffixTokens int
	NOutEstimate         float64
	SLOClass             string
}

type instStateRef struct {
	Key     string
	GPUType string
	BDec    int
	KV      int64
	SPf     int64
	Queue   int
}

type residentTableRef interface {
	snapshot(endpointKey string) []*residentRef
}

type residentRef struct {
	ArrivalUs      int64
	FirstTokenUs   int64
	FirstTokenSet  bool
	InputLen       int
	TokensStreamed int
	NOutEstimate   float64
	SLOClass       string
}

type Handler struct {
	params    Params
	residents residentTableRef
}

func (h *Handler) coeffsFor(gpuType string) coeffsRef { panic("TRANSLATE: share with focal arm") }
func (h *Handler) tAdm(instStateRef, coeffsRef, []*residentRef) float64 {
	panic("TRANSLATE: share with focal arm")
}
func (h *Handler) chunkTerms(ap int) (float64, float64) { panic("TRANSLATE: share with focal arm") }
func (h *Handler) cXferUs(*arrivalRef) float64          { panic("TRANSLATE: share with focal arm") }

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
