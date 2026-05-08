package eval

import (
	"math"
	"math/rand/v2"
	"sort"
)

type Correlation struct {
	Dimensions []Dimension `json:"dimensions"`
}

type Dimension struct {
	Name         string  `json:"name"`
	Spearman     float64 `json:"spearman"`
	Pearson      float64 `json:"pearson"`
	KendallTau   float64 `json:"kendall_tau"`
	SpearmanCI   *CI     `json:"spearman_ci,omitempty"`
	PearsonCI    *CI     `json:"pearson_ci,omitempty"`
	KendallTauCI *CI     `json:"kendall_tau_ci,omitempty"`
}

type CI struct {
	Low  float64 `json:"low"`
	High float64 `json:"high"`
}

type CorrelationOptions struct {
	// Bootstrap: when > 0, compute 95% CI by resampling at the DOCUMENT level
	// (summary-level) or SYSTEM level (system-level), this many times.
	// Standard in meta-evaluation is 1000 resamples.
	Bootstrap int

	// Level: "summary" (default) or "system".
	Level string

	// Seed for the bootstrap RNG. Zero picks a fixed default for reproducibility.
	Seed uint64
}

func NewCorrelation(samples []Sample, scores []float64) Correlation {
	return NewCorrelationWith(samples, scores, CorrelationOptions{})
}

func NewCorrelationWith(samples []Sample, scores []float64, opts CorrelationOptions) Correlation {
	level := opts.Level
	if level == "" {
		level = "summary"
	}

	dimNames := []string{"coherence", "consistency", "fluency", "relevance"}
	getDim := map[string]func(Sample) float64{
		"coherence":   func(s Sample) float64 { return s.Coherence },
		"consistency": func(s Sample) float64 { return s.Consistency },
		"fluency":     func(s Sample) float64 { return s.Fluency },
		"relevance":   func(s Sample) float64 { return s.Relevance },
	}

	out := Correlation{Dimensions: make([]Dimension, len(dimNames))}
	for i, name := range dimNames {
		human := make([]float64, len(samples))
		for j, s := range samples {
			human[j] = getDim[name](s)
		}

		x, y := scores, human
		if level == "system" {
			x, y = aggregateBySystem(samples, scores, human)
		}

		d := Dimension{
			Name:       name,
			Spearman:   Spearman(x, y),
			Pearson:    Pearson(x, y),
			KendallTau: KendallTau(x, y),
		}

		if opts.Bootstrap > 0 {
			seed := opts.Seed
			if seed == 0 {
				seed = 42
			}
			d.SpearmanCI = bootstrapOrNil(samples, scores, human, Spearman, opts.Bootstrap, level, seed)
			d.PearsonCI = bootstrapOrNil(samples, scores, human, Pearson, opts.Bootstrap, level, seed+1)
			d.KendallTauCI = bootstrapOrNil(samples, scores, human, KendallTau, opts.Bootstrap, level, seed+2)
		}

		out.Dimensions[i] = d
	}
	return out
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

func Pearson(x, y []float64) float64 {
	n := len(x)
	if n != len(y) || n < 2 {
		return 0
	}
	var sx, sy, sxy, sx2, sy2 float64
	for i := range x {
		sx += x[i]
		sy += y[i]
		sxy += x[i] * y[i]
		sx2 += x[i] * x[i]
		sy2 += y[i] * y[i]
	}
	nf := float64(n)
	num := nf*sxy - sx*sy
	den := math.Sqrt((nf*sx2 - sx*sx) * (nf*sy2 - sy*sy))
	if den == 0 {
		return 0
	}
	return num / den
}

func Spearman(x, y []float64) float64 {
	return Pearson(ranks(x), ranks(y))
}

func KendallTau(x, y []float64) float64 {
	n := len(x)
	if n != len(y) || n < 2 {
		return 0
	}
	var concordant, discordant, tiesX, tiesY int
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			dx, dy := x[i]-x[j], y[i]-y[j]
			switch {
			case dx*dy > 0:
				concordant++
			case dx*dy < 0:
				discordant++
			default:
				if dx == 0 {
					tiesX++
				}
				if dy == 0 {
					tiesY++
				}
			}
		}
	}
	n0 := n * (n - 1) / 2
	den := math.Sqrt(float64(n0-tiesX) * float64(n0-tiesY))
	if den == 0 {
		return 0
	}
	return float64(concordant-discordant) / den
}

type CorrelationFunc func(x, y []float64) float64

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

func aggregateBySystem(samples []Sample, metric, human []float64) ([]float64, []float64) {
	type acc struct {
		metricSum, humanSum float64
		count               int
	}
	bySystem := map[int]*acc{}
	for i, s := range samples {
		a, ok := bySystem[s.SystemID]
		if !ok {
			a = &acc{}
			bySystem[s.SystemID] = a
		}
		a.metricSum += metric[i]
		a.humanSum += human[i]
		a.count++
	}

	ids := make([]int, 0, len(bySystem))
	for id := range bySystem {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	xs := make([]float64, len(ids))
	ys := make([]float64, len(ids))
	for i, id := range ids {
		a := bySystem[id]
		n := float64(a.count)
		xs[i] = a.metricSum / n
		ys[i] = a.humanSum / n
	}
	return xs, ys
}

func groupByDocument(samples []Sample) map[string][]int {
	out := make(map[string][]int, len(samples)/16)
	for i, s := range samples {
		out[s.DocumentID] = append(out[s.DocumentID], i)
	}
	return out
}

type PairedComparison struct {
	DeltaMean float64 `json:"delta_mean"`
	DeltaCI   CI      `json:"delta_ci"`
	PValue    float64 `json:"p_value"`
	N         int     `json:"n"`
}

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

func ranks(vals []float64) []float64 {
	n := len(vals)
	type iv struct {
		v float64
		i int
	}
	s := make([]iv, n)
	for i, v := range vals {
		s[i] = iv{v, i}
	}
	sort.Slice(s, func(a, b int) bool { return s[a].v < s[b].v })

	r := make([]float64, n)
	for i := 0; i < n; {
		j := i + 1
		for j < n && s[j].v == s[i].v {
			j++
		}
		avg := float64(i+j+1) / 2.0
		for k := i; k < j; k++ {
			r[s[k].i] = avg
		}
		i = j
	}
	return r
}
