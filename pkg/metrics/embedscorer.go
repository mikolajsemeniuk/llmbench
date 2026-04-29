package metrics

import (
	"context"
	"fmt"
	"math"
)

// EmbeddingScorer computes cosine similarity between embeddings of
// reference and candidate as a single-shot semantic similarity baseline.
// Compare with SMART-Model which embeds at sentence level.
type EmbeddingScorer struct {
	Provider *Ollama
	Model    string
}

func NewEmbeddingScorer(host, model string) *EmbeddingScorer {
	return &EmbeddingScorer{
		Provider: NewOllama(host),
		Model:    model,
	}
}

// Score returns cosine similarity between the embeddings of reference and
// candidate. Result is in [-1, 1] but for typical embedding models on
// natural language, almost always in [0.5, 1.0] (the "ceiling effect").
func (e *EmbeddingScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	ref, err := e.Provider.Embed(ctx, EmbedInput{Model: e.Model, Input: reference})
	if err != nil {
		return 0, fmt.Errorf("embedding: embed reference: %w", err)
	}
	if len(ref.Embeddings) == 0 {
		return 0, fmt.Errorf("embedding: empty embedding for reference")
	}

	cand, err := e.Provider.Embed(ctx, EmbedInput{Model: e.Model, Input: candidate})
	if err != nil {
		return 0, fmt.Errorf("embedding: embed candidate: %w", err)
	}
	if len(cand.Embeddings) == 0 {
		return 0, fmt.Errorf("embedding: empty embedding for candidate")
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
