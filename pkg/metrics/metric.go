package metrics

import "context"

// Scorer is the common shape implemented by all model-backed scorers.
// GEval and UniEval take an extra dimension argument and intentionally do not
// satisfy this interface.
type Scorer interface {
	Score(ctx context.Context, reference, candidate string) (float64, error)
}
