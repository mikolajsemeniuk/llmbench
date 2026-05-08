package metrics

import (
	"context"
	"fmt"
)

// UniEvalScorer wraps the canonical-style UniEval Boolean QA evaluator
// (Zhong et al. 2022). Each instance is bound to a specific dimension;
// `cmd/unieval` runs one dimension per invocation. The simplified
// implementation in `cmd/modelsrv` uses a single yes/no probability
// rather than the boosting/calibration of the original paper.
//
// Canonical UniEval prompts condition on different fields per dimension:
//   - coherence/consistency: source document
//   - relevance:             reference summary
//   - fluency:               candidate only
//
// Score takes (reference, source, candidate); the server picks the
// right fields based on the configured dimension.
type UniEvalScorer struct {
	Server    *ModelServer
	Dimension string
}

func NewUniEvalScorer(host, dimension string) *UniEvalScorer {
	if dimension == "" {
		dimension = "coherence"
	}

	return &UniEvalScorer{Server: NewModelServer(host), Dimension: dimension}
}

func (u *UniEvalScorer) Score(ctx context.Context, reference, source, candidate string) (float64, error) {
	in := modelServerRequest{
		Reference: reference,
		Candidate: candidate,
		Source:    source,
		Dimension: u.Dimension,
	}
	res, err := u.Server.post(ctx, "/unieval", in)
	if err != nil {
		return 0, fmt.Errorf("unieval: %w", err)
	}

	return res.Score, nil
}
