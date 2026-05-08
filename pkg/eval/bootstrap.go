// Bootstrap routines for confidence intervals (single-metric) and
// paired comparisons (metric-vs-baseline), plus the multi-run
// aggregator that computes mean±std across N independent
// Correlation values (used for stochastic LLM-judge metrics).
//
// All resampling is at the article level: SummEval's 16 candidates
// per article are not independent, so the cluster bootstrap is the
// canonical fix. The same article-level grouping powers PairedBootstrap.
package eval

import (
	"math"
	"math/rand/v2"
	"sort"
)

type CI struct {
	Low  float64 `json:"low"`
	High float64 `json:"high"`
}

// BootstrapCI computes a 95% confidence interval by resampling.
//
// For "summary" level: resamples documents (SummEval pairs within a doc
// are not independent — all 16 systems summarize the same article).
// For "system" level: resamples systems.
//
// Returns zero CI{} if there's insufficient data to bootstrap.
func BootstrapCI(samples []Sample, metric, human []float64, fn CorrelationFunc, n int, level string, seed uint64) CI {
	if n < 2 || len(samples) != len(metric) || len(samples) != len(human) {
		return CI{}
	}

	rng := rand.New(rand.NewPCG(seed, seed^0xdeadbeef))
	values := make([]float64, 0, n)

	switch level {
	case "system":
		xAgg, yAgg := aggregateBySystem(samples, metric, human)
		k := len(xAgg)
		if k < 2 {
			return CI{}
		}
		xs := make([]float64, k)
		ys := make([]float64, k)
		for b := 0; b < n; b++ {
			for i := 0; i < k; i++ {
				j := rng.IntN(k)
				xs[i], ys[i] = xAgg[j], yAgg[j]
			}
			values = append(values, fn(xs, ys))
		}

	default: // "summary" — resample documents
		groups := groupByDocument(samples)
		docIDs := make([]string, 0, len(groups))
		for id := range groups {
			docIDs = append(docIDs, id)
		}
		sort.Strings(docIDs)

		if len(docIDs) < 2 {
			return CI{}
		}

		xs := make([]float64, 0, len(samples))
		ys := make([]float64, 0, len(samples))

		for b := 0; b < n; b++ {
			xs = xs[:0]
			ys = ys[:0]
			for i := 0; i < len(docIDs); i++ {
				pickedID := docIDs[rng.IntN(len(docIDs))]
				for _, idx := range groups[pickedID] {
					xs = append(xs, metric[idx])
					ys = append(ys, human[idx])
				}
			}
			if len(xs) >= 2 {
				values = append(values, fn(xs, ys))
			}
		}
	}

	return percentileCI(values, 0.025, 0.975)
}

// bootstrapOrNil returns nil if bootstrap cannot be computed (insufficient
// data), so that omitempty drops the field from JSON.
func bootstrapOrNil(samples []Sample, metric, human []float64, fn CorrelationFunc, n int, level string, seed uint64) *CI {
	ci := BootstrapCI(samples, metric, human, fn, n, level, seed)
	if ci.Low == 0 && ci.High == 0 {
		// Distinguish "bootstrap failed / insufficient data" from legitimate
		// zero-centered CI by checking: if both are exactly 0, it's a guard
		// return. A legitimate CI would have at least microscopic jitter.
		return nil
	}
	return &ci
}

type PairedComparison struct {
	DeltaMean float64 `json:"delta_mean"`
	DeltaCI   CI      `json:"delta_ci"`
	PValue    float64 `json:"p_value"`
	N         int     `json:"n"`
}

// PairedBootstrap resamples documents and on each resample computes
// both metrics' correlation against human judgment, recording the
// delta. Pairing eliminates the shared document-difficulty noise
// that would inflate the apparent uncertainty of an unpaired
// comparison. Returns the mean delta, a 95% CI for the delta, and
// a two-sided p-value.
func PairedBootstrap(samples []Sample, scoresA, scoresB, human []float64,
	fn CorrelationFunc, n int, seed uint64) PairedComparison {

	if n < 2 {
		return PairedComparison{}
	}
	groups := groupByDocument(samples)
	docIDs := make([]string, 0, len(groups))
	for id := range groups {
		docIDs = append(docIDs, id)
	}
	sort.Strings(docIDs)
	if len(docIDs) < 2 {
		return PairedComparison{}
	}

	rng := rand.New(rand.NewPCG(seed, seed^0xcafebabe))
	deltas := make([]float64, 0, n)

	xa := make([]float64, 0, len(samples))
	xb := make([]float64, 0, len(samples))
	yy := make([]float64, 0, len(samples))

	for b := 0; b < n; b++ {
		xa = xa[:0]
		xb = xb[:0]
		yy = yy[:0]
		for i := 0; i < len(docIDs); i++ {
			pickedID := docIDs[rng.IntN(len(docIDs))]
			for _, idx := range groups[pickedID] {
				xa = append(xa, scoresA[idx])
				xb = append(xb, scoresB[idx])
				yy = append(yy, human[idx])
			}
		}
		if len(xa) < 2 {
			continue
		}
		deltas = append(deltas, fn(xa, yy)-fn(xb, yy))
	}

	if len(deltas) < 2 {
		return PairedComparison{}
	}

	sort.Float64s(deltas)
	mean := 0.0
	below, above := 0, 0
	for _, d := range deltas {
		mean += d
		if d <= 0 {
			below++
		}
		if d >= 0 {
			above++
		}
	}
	mean /= float64(len(deltas))

	p := 2.0 * math.Min(float64(below), float64(above)) / float64(len(deltas))
	if p > 1 {
		p = 1
	}

	return PairedComparison{
		DeltaMean: mean,
		DeltaCI:   percentileCI(deltas, 0.025, 0.975),
		PValue:    p,
		N:         len(deltas),
	}
}

// groupByDocument indexes []Sample by DocumentID, returning the
// positions of each document's samples in the original slice. Used
// by BootstrapCI's "summary" level branch and by PairedBootstrap to
// resample at the article level rather than the sample level.
func groupByDocument(samples []Sample) map[string][]int {
	out := make(map[string][]int, len(samples)/16)
	for i, s := range samples {
		out[s.DocumentID] = append(out[s.DocumentID], i)
	}
	return out
}

// percentileCI returns the [lo, hi] percentile bracket of values as
// a CI. Used as the final step of every bootstrap routine here.
func percentileCI(values []float64, lo, hi float64) CI {
	if len(values) < 2 {
		return CI{}
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	return CI{
		Low:  percentile(sorted, lo),
		High: percentile(sorted, hi),
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	pos := p * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return sorted[lo]
	}
	frac := pos - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

// AggregateRuns computes per-dimension mean and std of each
// correlation coefficient across N independent Correlation values
// (e.g., from N independent G-Eval runs at temperature>0). Std is
// the unbiased sample standard deviation (Bessel's correction);
// for n=1 it is set to 0.
func AggregateRuns(corrs []Correlation) []DimensionRunStats {
	if len(corrs) == 0 {
		return nil
	}
	dimNames := []string{"coherence", "consistency", "fluency", "relevance"}
	out := make([]DimensionRunStats, 0, len(dimNames))
	for _, name := range dimNames {
		var sp, pe, ke []float64
		for _, c := range corrs {
			for _, d := range c.Dimensions {
				if d.Name != name {
					continue
				}
				sp = append(sp, d.Spearman)
				pe = append(pe, d.Pearson)
				ke = append(ke, d.KendallTau)
			}
		}
		out = append(out, DimensionRunStats{
			Name:     name,
			Spearman: meanStd(sp),
			Pearson:  meanStd(pe),
			Kendall:  meanStd(ke),
		})
	}
	return out
}

func meanStd(xs []float64) RunStats {
	n := len(xs)
	if n == 0 {
		return RunStats{}
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	mean := sum / float64(n)
	if n < 2 {
		return RunStats{Mean: mean}
	}
	var ss float64
	for _, x := range xs {
		d := x - mean
		ss += d * d
	}
	return RunStats{Mean: mean, Std: math.Sqrt(ss / float64(n-1))}
}
