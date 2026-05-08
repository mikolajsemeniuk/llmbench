// Package eval provides the Correlation orchestrator that turns a
// vector of metric scores plus a sample list into per-dimension
// summary-level or system-level correlations against human judgment,
// optionally with bootstrap confidence intervals. The pure
// rank/linear correlation primitives (Pearson, Spearman, Kendall)
// also live in this file because they have no dependency on the
// SummEval Sample structure and are conceptually upstream of the
// orchestration; the resampling routines that produce the optional
// CIs live in bootstrap.go.
package eval

import (
	"math"
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

// aggregateBySystem groups (metric, human) score pairs by SystemID
// and replaces each system's contribution with the per-system mean.
// Used both by NewCorrelationWith for system-level correlation and
// by BootstrapCI for the system-level resampling branch.
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

// CorrelationFunc is the signature shared by the three correlation
// coefficients below. It is the type the bootstrap routines accept,
// so any rank- or linear-correlation function can be plugged in.
type CorrelationFunc func(x, y []float64) float64

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

// ranks returns the average-ranked positions of vals (ties get the
// mean of their tied positions). Used by Spearman, which is Pearson
// applied to ranks.
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
