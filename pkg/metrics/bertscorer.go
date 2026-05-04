package metrics

import (
	"context"
	"fmt"
)

// BERTScorer computes canonical token-level BERTScore F1 (Zhang et al.
// 2020). Backed by the Python /bertscore endpoint of `cmd/modelsrv`,
// which uses the `bert_score` library with roberta-large by default.
type BERTScorer struct {
	Server *ModelServer
}

func NewBERTScorer(host string) *BERTScorer {
	return &BERTScorer{Server: NewModelServer(host)}
}

func (b *BERTScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	res, err := b.Server.post(ctx, "/bertscore", modelServerRequest{
		Reference: reference,
		Candidate: candidate,
	})
	if err != nil {
		return 0, fmt.Errorf("bertscore: %w", err)
	}
	return res.Score, nil
}
