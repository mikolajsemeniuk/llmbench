package metrics

import (
	"context"
	"fmt"
)

// BARTScorer implements canonical BARTScore (Yuan et al. 2021):
// log P(reference | candidate) using a seq2seq model (BART).
//
// We delegate to the Python model server which uses facebook/bart-large-cnn.
// This requires the /bartscore endpoint on the model server.
type BARTScorer struct {
	Server *ModelServer
}

func NewBARTScorer(host string) *BARTScorer {
	return &BARTScorer{Server: NewModelServer(host)}
}

// Score returns the BARTScore: log P(reference | candidate) averaged
// per token. Higher (less negative) means the candidate better predicts
// the reference. Typically in [-10, 0] range.
func (b *BARTScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	in := modelServerRequest{
		Reference: reference,
		Candidate: candidate,
	}
	res, err := b.Server.post(ctx, "/bartscore", in)
	if err != nil {
		return 0, fmt.Errorf("bartscore: %w", err)
	}

	return res.Score, nil
}
