package metrics

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// LNC — Likelihood–NLI Composite. A per-dimension, reference-free
// scorer built from the signals of three earlier experiments, all
// computed with small open models:
//
//	CCM  (Qwen2.5-1.5B, source in context, K=8 perturbations)
//	     logp_tok, logp_sum, ulogp_tok, order margin, negation margin,
//	     z-margin
//	NLIG (DeBERTa-v3-large NLI, SummaC-ZS construction)
//	     mean and min over summary sentences of max_w P(e) − P(c)
//	SPL  (same LM, per-token)
//	     mean log-prob at copy splice points and inside copied spans
//
// Each dimension's score is a ridge combination of the standardised
// features:
//
//	score_d = Σ_f  w_{d,f} · (f − μ_f) / σ_f
//
// μ, σ and w are fitted on the development half of SummEval only (first
// 50 articles, ridge α = 10 on the standardised human-score ranks) and
// frozen in a calibration file; the held-out last 50 are never used for
// fitting. The ridge (rather than picking the best single signal or
// pair per dimension) was chosen by 5-fold article-level cross-
// validation inside the development half.
//
// Prior art: learned combinations of existing metrics calibrated on
// human ratings (MetaMetrics, Winata et al., ICLR 2025); the NLI
// component is SummaC-ZS; the likelihood components follow BARTScore
// s→h / FFLM; the sentence-order contrast follows DeltaScore. LNC's
// claim is empirical — which cheap, reference-free signals carry each
// SummEval dimension once extractiveness is partialled out — not a new
// primitive.

// LNCFeatures lists the feature names in calibration order.
var LNCFeatures = []string{
	"logp_tok", "logp_sum", "ulogp_tok", "order", "negation", "zmargin",
	"nlig", "nlig_min", "spl", "spl_inside",
}

// LNCFeatureVector extracts the features of one candidate.
func LNCFeatureVector(c CCMResult, n NLIResult, s SPLResult) []float64 {
	neg, _ := c.FamilyMargin(CCMNegation)
	ulogp := 0.0
	if c.Tokens > 0 {
		ulogp = c.Uncond / float64(c.Tokens)
	}
	return []float64{
		c.LogPPerToken(), c.Cond, ulogp, c.OrderMargin(), neg, c.CoherenceScore(),
		n.Score(), n.MinScore(), s.Score(), s.InsideScore(),
	}
}

// LNCCalibration is the frozen development-split fit.
type LNCCalibration struct {
	Features []string             `json:"features"`
	Mu       []float64            `json:"mu"`
	Sigma    []float64            `json:"sigma"`
	Alpha    float64              `json:"alpha"`
	Weights  map[string][]float64 `json:"weights"` // dimension → per-feature weight
	FitOn    string               `json:"fit_on"`
	// Debiased marks a fit whose ridge target was the human-score rank
	// with the copy-rate rank projected out (see FitLNC).
	Debiased bool `json:"debiased"`
}

// Score applies the calibration for one dimension.
func (cal *LNCCalibration) Score(dim string, f []float64) (float64, error) {
	w, ok := cal.Weights[dim]
	if !ok {
		return 0, fmt.Errorf("lnc: no weights for dimension %q", dim)
	}
	if len(f) != len(w) {
		return 0, fmt.Errorf("lnc: %d features, calibration has %d", len(f), len(w))
	}
	var s float64
	for i := range f {
		s += w[i] * (f[i] - cal.Mu[i]) / cal.Sigma[i]
	}
	return s, nil
}

// FitLNC fits the calibration on development rows: X[i] is the feature
// vector of sample i, human[dim][i] its rating. When copyRate is non-nil
// the ridge target is the standardised human rank with the standardised
// copy-rate rank projected out, so the weights reward what the humans
// see beyond extractiveness; copy rate is used ONLY in fitting, never as
// a feature, so scoring new data needs nothing extra.
func FitLNC(X [][]float64, human map[string][]float64, copyRate []float64, alpha float64, fitOn string) (*LNCCalibration, error) {
	if len(X) == 0 {
		return nil, fmt.Errorf("lnc: no development rows")
	}
	p := len(X[0])
	cal := &LNCCalibration{
		Features: LNCFeatures, Mu: make([]float64, p), Sigma: make([]float64, p),
		Alpha: alpha, Weights: map[string][]float64{}, FitOn: fitOn,
		Debiased: copyRate != nil,
	}
	var q []float64
	if copyRate != nil {
		q = standardise(averageRanks(copyRate))
	}
	n := float64(len(X))
	for j := 0; j < p; j++ {
		for _, r := range X {
			cal.Mu[j] += r[j]
		}
		cal.Mu[j] /= n
		for _, r := range X {
			cal.Sigma[j] += (r[j] - cal.Mu[j]) * (r[j] - cal.Mu[j])
		}
		cal.Sigma[j] = math.Sqrt(cal.Sigma[j] / n)
		if cal.Sigma[j] == 0 {
			return nil, fmt.Errorf("lnc: feature %s is constant on the fit split", LNCFeatures[j])
		}
	}
	Z := make([][]float64, len(X))
	for i, r := range X {
		Z[i] = make([]float64, p)
		for j := range r {
			Z[i][j] = (r[j] - cal.Mu[j]) / cal.Sigma[j]
		}
	}
	for dim, h := range human {
		y := standardise(averageRanks(h))
		if q != nil {
			var yq, qq float64
			for i := range y {
				yq += y[i] * q[i]
				qq += q[i] * q[i]
			}
			for i := range y {
				y[i] -= yq / qq * q[i]
			}
		}
		// (ZᵀZ + αI) w = Zᵀy
		A := make([][]float64, p)
		b := make([]float64, p)
		for a := 0; a < p; a++ {
			A[a] = make([]float64, p)
			for c := 0; c < p; c++ {
				for i := range Z {
					A[a][c] += Z[i][a] * Z[i][c]
				}
			}
			A[a][a] += alpha
			for i := range Z {
				b[a] += Z[i][a] * y[i]
			}
		}
		w, err := solveLinear(A, b)
		if err != nil {
			return nil, fmt.Errorf("lnc: fit %s: %w", dim, err)
		}
		cal.Weights[dim] = w
	}
	return cal, nil
}

// averageRanks returns 1-based ranks with ties sharing their mean rank
// (scipy.stats.rankdata "average").
func averageRanks(v []float64) []float64 {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return v[idx[a]] < v[idx[b]] })
	r := make([]float64, len(v))
	for i := 0; i < len(idx); {
		j := i
		for j+1 < len(idx) && v[idx[j+1]] == v[idx[i]] {
			j++
		}
		avg := float64(i+j)/2 + 1
		for k := i; k <= j; k++ {
			r[idx[k]] = avg
		}
		i = j + 1
	}
	return r
}

func standardise(v []float64) []float64 {
	var m, s float64
	for _, x := range v {
		m += x
	}
	m /= float64(len(v))
	for _, x := range v {
		s += (x - m) * (x - m)
	}
	s = math.Sqrt(s / float64(len(v)))
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = (x - m) / s
	}
	return out
}

// solveLinear solves A x = b by Gaussian elimination with partial
// pivoting (A is small and symmetric positive definite here).
func solveLinear(A [][]float64, b []float64) ([]float64, error) {
	n := len(b)
	M := make([][]float64, n)
	for i := range M {
		M[i] = append(append([]float64{}, A[i]...), b[i])
	}
	for c := 0; c < n; c++ {
		piv := c
		for r := c + 1; r < n; r++ {
			if math.Abs(M[r][c]) > math.Abs(M[piv][c]) {
				piv = r
			}
		}
		if math.Abs(M[piv][c]) < 1e-12 {
			return nil, fmt.Errorf("singular system")
		}
		M[c], M[piv] = M[piv], M[c]
		for r := c + 1; r < n; r++ {
			f := M[r][c] / M[c][c]
			for k := c; k <= n; k++ {
				M[r][k] -= f * M[c][k]
			}
		}
	}
	x := make([]float64, n)
	for r := n - 1; r >= 0; r-- {
		s := M[r][n]
		for k := r + 1; k < n; k++ {
			s -= M[r][k] * x[k]
		}
		x[r] = s / M[r][r]
	}
	return x, nil
}

// CopyRate is the fraction of the candidate's token bigrams that occur
// in the source, tokenised exactly like cmd/confound (lowercase,
// . , ! ? ; : ( ) " ' split off). Used only to fit a debiased LNC.
func CopyRate(source, candidate string) float64 {
	tok := func(s string) []string {
		s = strings.ToLower(s)
		for _, p := range []string{".", ",", "!", "?", ";", ":", "(", ")", "\"", "'"} {
			s = strings.ReplaceAll(s, p, " "+p+" ")
		}
		return strings.Fields(s)
	}
	src := map[string]bool{}
	ts := tok(source)
	for i := 0; i+1 < len(ts); i++ {
		src[ts[i]+" "+ts[i+1]] = true
	}
	tc := tok(candidate)
	if len(tc) < 2 {
		return 0
	}
	hit := 0
	for i := 0; i+1 < len(tc); i++ {
		if src[tc[i]+" "+tc[i+1]] {
			hit++
		}
	}
	return float64(hit) / float64(len(tc)-1)
}
