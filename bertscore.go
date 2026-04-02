package llmbench

import (
	"context"
	"fmt"
	"math"
)

// BERTScore computes a simplified BERTScore by embedding both texts with
// the given Ollama embedding model and returning cosine similarity.
//
// The canonical BERTScore computes token-level pairwise cosine similarity and
// then greedy-matches tokens. This simplified version uses sentence-level
// embeddings, which is a common practical approximation and sufficient for
// our evaluation framework. The difference is discussed in the paper's
// methodology section.
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

// cosineSimilarity returns the cosine similarity of two vectors.
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
