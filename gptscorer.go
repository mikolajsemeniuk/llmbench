package llmbench

import (
	"context"
	"fmt"
)

// GPTScorer computes GPTScore via the model server.
//
// GPTScore (Fu et al., 2023) measures the conditional log-probability
// of generating the candidate text given the reference as context.
// Score = (1/m) * Σ log P(candidate_token_i | tokens_<i, reference)
//
// Higher score = the model considers the candidate a more natural/likely
// continuation of the reference context. This captures fluency and
// semantic fidelity without explicit comparison.
//
// Requires a generative model (GPT-2) running in the model server.
type GPTScorer struct {
	Server *ModelServer
}

func NewGPTScorer(host string) *GPTScorer {
	return &GPTScorer{Server: NewModelServer(host)}
}

func (g *GPTScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	resp, err := g.Server.post(ctx, "/gptscore", modelServerRequest{
		Reference: reference,
		Candidate: candidate,
	})
	if err != nil {
		return 0, fmt.Errorf("gptscore: %w", err)
	}
	return resp.Score, nil
}
