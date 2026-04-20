package llmbench

import (
	"context"
	"fmt"
	"math"
)

type EmbeddingScorer struct {
	ctx      context.Context
	Provider *Ollama
	Model    string
}

func NewEmbeddingScorer(ctx context.Context, host, model string) *EmbeddingScorer {
	return &EmbeddingScorer{
		ctx:      ctx,
		Provider: NewOllama(host),
		Model:    model,
	}
}

func (e *EmbeddingScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	ref, err := e.Provider.Embed(ctx, EmbedInput{Model: e.Model, Input: reference})
	if err != nil {
		return 0, fmt.Errorf("embedding: embed reference: %w", err)
	}
	cand, err := e.Provider.Embed(ctx, EmbedInput{Model: e.Model, Input: candidate})
	if err != nil {
		return 0, fmt.Errorf("embedding: embed candidate: %w", err)
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
