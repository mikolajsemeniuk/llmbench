package llmbench

import (
	"slices"
)

var Aggregators = map[string]Norm{
	"max":  Max,
	"mean": Mean,
}

type Norm func([]float64) float64

func Max(in []float64) float64 {
	if len(in) == 0 {
		return 0
	}

	return slices.Max(in)
}

func Mean(in []float64) float64 {
	var out float64
	for _, v := range in {
		out += v
	}

	return out / float64(len(in))
}
