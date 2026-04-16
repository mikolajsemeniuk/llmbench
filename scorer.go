package llmbench

import (
	"context"
	"slices"
)

type Metric func(reference, candidate string) float64
type Norm func([]float64) float64

func Max(in []float64) float64 {
	if len(in) == 0 {
		return 0
	}

	return slices.Max(in)
}

func Mean(in []float64) float64 {
	if len(in) == 0 {
		return 0
	}

	var out float64
	for _, v := range in {
		out += v
	}

	return out / float64(len(in))
}

func Score(ctx context.Context, in []Sample, m Metric, n Norm) []float64 {
	out := make([]float64, len(in))
	for i, s := range in {
		refs := make([]float64, len(s.References))
		for j, ref := range s.References {
			refs[j] = m(ref, s.Candidate)
		}

		out[i] = n(refs)
	}

	return out
}
