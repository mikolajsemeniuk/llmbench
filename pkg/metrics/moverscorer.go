package metrics

import (
	"context"
	"fmt"
)

// MoverScorer computes Word Mover's Distance over contextual token
// embeddings (Zhao et al. 2019). The simplified implementation in
// `cmd/modelsrv` uses uniform token weights — the canonical paper
// uses IDF + p-mean weighting. See the Methodology section of the
// paper for caveats.
type MoverScorer struct {
	Server *ModelServer
}

func NewMoverScorer(host string) *MoverScorer {
	return &MoverScorer{Server: NewModelServer(host)}
}

func (m *MoverScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	res, err := m.Server.post(ctx, "/moverscore", modelServerRequest{
		Reference: reference,
		Candidate: candidate,
	})
	if err != nil {
		return 0, fmt.Errorf("moverscore: %w", err)
	}
	return res.Score, nil
}
