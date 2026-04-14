package llmbench

import (
	"context"
	"fmt"
	"os"
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
	ctx    context.Context
	Server *ModelServer
}

func NewGPTScorer(ctx context.Context, host string) *GPTScorer {
	return &GPTScorer{ctx: ctx, Server: NewModelServer(host)}
}

func (g *GPTScorer) score(ctx context.Context, reference, candidate string) (float64, error) {
	resp, err := g.Server.post(ctx, "/gptscore", modelServerRequest{
		Reference: reference,
		Candidate: candidate,
	})
	if err != nil {
		return 0, fmt.Errorf("gptscore: %w", err)
	}
	return resp.Score, nil
}

// Score implements Scorer.
func (g *GPTScorer) Score(entries []Entry) (ScoreOutput, error) {
	var out ScoreOutput
	total := 0
	for _, e := range entries {
		total += len(e.MachineSummaries)
	}
	done := 0
	for _, e := range entries {
		for mi, mach := range e.MachineSummaries {
			best := 0.0
			for _, ref := range e.HumanSummaries {
				s, err := g.score(g.ctx, ref, mach)
				if err != nil {
					return ScoreOutput{}, err
				}
				if s > best {
					best = s
				}
			}
			out.Scores = append(out.Scores, best)
			out.Relevance = append(out.Relevance, e.Relevance[mi])
			out.Coherence = append(out.Coherence, e.Coherence[mi])
			out.Fluency = append(out.Fluency, e.Fluency[mi])
			out.Consistency = append(out.Consistency, e.Consistency[mi])
			done++
			fmt.Fprintf(os.Stderr, "\r[GPTScore] %d/%d", done, total)
		}
	}
	return out, nil
}
