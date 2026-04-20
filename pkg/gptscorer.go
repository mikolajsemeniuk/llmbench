package llmbench

import (
	"context"
	"fmt"
)

type GPTScorer struct {
	Server *ModelServer
}

func NewGPTScorer(host string) *GPTScorer {
	return &GPTScorer{Server: NewModelServer(host)}
}

func (g *GPTScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	in := modelServerRequest{
		Reference: reference,
		Candidate: candidate,
	}
	res, err := g.Server.post(ctx, "/gptscore", in)
	if err != nil {
		return 0, fmt.Errorf("gptscore: %w", err)
	}

	return res.Score, nil
}
