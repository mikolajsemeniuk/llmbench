// Score normalisation strategies for metrics whose raw output spans
// an unbounded or per-document-variable range. A Norm collapses a
// single document's per-pair raw scores into a scalar reference value
// that the metric can then be normalised against (e.g., dividing by
// the per-document max so scores fall in [0, 1] regardless of the
// metric's native scale). Currently used by the BARTScore-style
// reference-conditional baselines; LGS does not need a Norm because
// its score is already in [-1, 1] structurally.
package eval

import "slices"

const DefaultNormName = "max"

var DefaultNorm = Max
var Norms = map[string]Norm{"max": Max, "mean": Mean}

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
