package llmbench

import (
	"context"
	"fmt"
	"math"
	"os"
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

// score computes cosine similarity between sentence-level embeddings
// of reference and candidate.
func (e *EmbeddingScorer) score(ctx context.Context, reference, candidate string) (float64, error) {
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

// maxScore returns the best cosine similarity of candidate against all references.
func (e *EmbeddingScorer) maxScore(ctx context.Context, references []string, candidate string) (float64, error) {
	best := 0.0
	for _, ref := range references {
		s, err := e.score(ctx, ref, candidate)
		if err != nil {
			return 0, err
		}
		if s > best {
			best = s
		}
	}
	return best, nil
}

// Score implements Scorer.
func (e *EmbeddingScorer) Score(entries []Entry) (ScoreOutput, error) {
	var out ScoreOutput
	total := 0
	for _, en := range entries {
		total += len(en.MachineSummaries)
	}
	done := 0
	for _, en := range entries {
		for mi, mach := range en.MachineSummaries {
			s, err := e.maxScore(e.ctx, en.HumanSummaries, mach)
			if err != nil {
				return ScoreOutput{}, err
			}
			out.Scores = append(out.Scores, s)
			out.Relevance = append(out.Relevance, en.Relevance[mi])
			out.Coherence = append(out.Coherence, en.Coherence[mi])
			out.Fluency = append(out.Fluency, en.Fluency[mi])
			out.Consistency = append(out.Consistency, en.Consistency[mi])
			done++
			fmt.Fprintf(os.Stderr, "\r[EmbedScorer] %d/%d", done, total)
		}
	}
	return out, nil
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
