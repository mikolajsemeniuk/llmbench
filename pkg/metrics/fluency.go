package metrics

import (
	"context"
	"fmt"
)

// Fluency computes a candidate-only fluency score from unconditioned
// GPT-2 perplexity via the /fluency endpoint of cmd/modelsrv. Unlike
// GPTScore (candidate conditioned on a reference) or LGS/NLI (which
// look at the source), this signal never sees the source or a
// reference at all -- it can only reward text that reads like fluent
// English, so it structurally cannot reward copying. Proposed in
// improvements.txt section 6 as the fluency component of a
// per-dimension composite metric.
type Fluency struct {
	Server *ModelServer
}

func NewFluency(host string) *Fluency {
	return &Fluency{Server: NewModelServer(host)}
}

func (f *Fluency) Score(ctx context.Context, candidate string) (float64, error) {
	in := modelServerRequest{Candidate: candidate}
	res, err := f.Server.post(ctx, "/fluency", in)
	if err != nil {
		return 0, fmt.Errorf("fluency: %w", err)
	}
	return res.Score, nil
}
