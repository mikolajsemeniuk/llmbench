package llmbench

import (
	"context"
	"fmt"
)

// SMARTModelScorer computes the SMART metric using embedding cosine similarity
// as the sentence matching function instead of ROUGE-L.
type SMARTModelScorer struct {
	ctx      context.Context
	Provider *Ollama
	Model    string
}

func NewSMARTModelScorer(ctx context.Context, host, model string) *SMARTModelScorer {
	return &SMARTModelScorer{
		ctx:      ctx,
		Provider: NewOllama(host),
		Model:    model,
	}
}

// score computes SMART with model-based sentence matching.
func (s *SMARTModelScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	refSents := splitSentences(reference)
	candSents := splitSentences(candidate)

	if len(refSents) == 0 || len(candSents) == 0 {
		return 0, nil
	}

	refEmbeds, err := s.embedAll(ctx, refSents)
	if err != nil {
		return 0, fmt.Errorf("smart-model: embed reference: %w", err)
	}
	candEmbeds, err := s.embedAll(ctx, candSents)
	if err != nil {
		return 0, fmt.Errorf("smart-model: embed candidate: %w", err)
	}

	// Precision: for each candidate sentence, find best matching reference sentence.
	var precisionSum float64
	for _, ce := range candEmbeds {
		best := 0.0
		for _, re := range refEmbeds {
			if sim := cosineSimilarity(ce, re); sim > best {
				best = sim
			}
		}
		precisionSum += best
	}
	precision := precisionSum / float64(len(candEmbeds))

	// Recall: for each reference sentence, find best matching candidate sentence.
	var recallSum float64
	for _, re := range refEmbeds {
		best := 0.0
		for _, ce := range candEmbeds {
			if sim := cosineSimilarity(re, ce); sim > best {
				best = sim
			}
		}
		recallSum += best
	}
	recall := recallSum / float64(len(refEmbeds))

	if precision+recall == 0 {
		return 0, nil
	}
	return 2 * precision * recall / (precision + recall), nil
}

func (s *SMARTModelScorer) embedAll(ctx context.Context, sentences []string) ([][]float64, error) {
	embeds := make([][]float64, len(sentences))
	for i, sent := range sentences {
		resp, err := s.Provider.Embed(ctx, EmbedInput{Model: s.Model, Input: sent})
		if err != nil {
			return nil, err
		}
		embeds[i] = resp.Embeddings[0]
	}
	return embeds, nil
}
