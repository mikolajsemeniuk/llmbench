package llmbench

import (
	"math"
)

func NewCorrelation(samples []Sample, scores []float64) Correlation {
	dims := []struct {
		name string
		vals func(Sample) float64
	}{
		{"coherence", func(s Sample) float64 { return s.Coherence }},
		{"consistency", func(s Sample) float64 { return s.Consistency }},
		{"fluency", func(s Sample) float64 { return s.Fluency }},
		{"relevance", func(s Sample) float64 { return s.Relevance }},
	}

	human := make([]float64, len(samples))
	out := Correlation{Dimensions: make([]Dimension, len(dims))}
	for i, d := range dims {
		for j, s := range samples {
			human[j] = d.vals(s)
		}
		out.Dimensions[i] = Dimension{
			Name:       d.name,
			Spearman:   SpearmanCorrelation(scores, human),
			Pearson:    PearsonCorrelation(scores, human),
			KendallTau: KendallTauCorrelation(scores, human),
		}
	}

	return out
}

type Correlation struct {
	Dimensions []Dimension
}

type Dimension struct {
	Name       string
	Spearman   float64
	Pearson    float64
	KendallTau float64
}

func PearsonCorrelation(x, y []float64) float64 {
	n := len(x)
	if n != len(y) || n == 0 {
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

func SpearmanCorrelation(x, y []float64) float64 {
	ranks := func(vals []float64) []float64 {
		n := len(vals)
		type iv struct {
			v float64
			i int
		}

		s := make([]iv, n)
		for i, v := range vals {
			s[i] = iv{v, i}
		}

		for i := 1; i < n; i++ {
			for j := i; j > 0 && s[j].v < s[j-1].v; j-- {
				s[j], s[j-1] = s[j-1], s[j]
			}
		}

		ranks := make([]float64, n)
		for i := 0; i < n; {
			j := i + 1
			for j < n && s[j].v == s[i].v {
				j++
			}

			avg := float64(i+j+1) / 2.0
			for k := i; k < j; k++ {
				ranks[s[k].i] = avg
			}

			i = j
		}

		return ranks
	}

	return PearsonCorrelation(ranks(x), ranks(y))
}

func KendallTauCorrelation(x, y []float64) float64 {
	n := len(x)
	if n != len(y) || n < 2 {
		return 0
	}
	var concordant, discordant, tiesX, tiesY int
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			dx := x[i] - x[j]
			dy := y[i] - y[j]
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
