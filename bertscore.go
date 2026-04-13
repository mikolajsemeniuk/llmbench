package llmbench

import (
	"context"
	"fmt"
	"math"
)

type BERTScorer struct {
	Provider *Ollama
	Model    string
}

func NewBERTScorer(host, model string) *BERTScorer {
	return &BERTScorer{
		Provider: NewOllama(host),
		Model:    model,
	}
}

func (b *BERTScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	ref, err := b.Provider.Embed(ctx, EmbedInput{Model: b.Model, Input: reference})
	if err != nil {
		return 0, fmt.Errorf("bertscore: embed reference: %w", err)
	}

	cand, err := b.Provider.Embed(ctx, EmbedInput{Model: b.Model, Input: candidate})
	if err != nil {
		return 0, fmt.Errorf("bertscore: embed candidate: %w", err)
	}

	return cosineSimilarity(ref.Embeddings[0], cand.Embeddings[0]), nil
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}

	return dot / denom
}
