package llmbench

import (
	"context"
	"fmt"
	"strings"
	"unicode"
)

// SentenceCoverageScorer computes semantic coverage at the sentence level.
//
// Inspired by SMART (Amplayo et al., ICLR 2023), this metric treats sentences
// as basic units of matching instead of tokens. For each sentence in the
// reference, it finds the best-matching sentence in the candidate (and vice
// versa), producing recall and precision scores that are combined into an F1.
//
// This approach captures information coverage better than token-level metrics
// like BERTScore, particularly when the candidate omits or adds information
// relative to the reference.
type SentenceCoverageScorer struct {
	Provider *Ollama
	Model    string
}

func NewSentenceCoverageScorer(host, model string) *SentenceCoverageScorer {
	return &SentenceCoverageScorer{
		Provider: NewOllama(host),
		Model:    model,
	}
}

// Score computes the sentence-level coverage F1 between reference and candidate.
func (s *SentenceCoverageScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	refSents := splitSentences(reference)
	candSents := splitSentences(candidate)

	if len(refSents) == 0 || len(candSents) == 0 {
		// Fallback to full-text cosine if we can't split into sentences
		return s.fullTextSimilarity(ctx, reference, candidate)
	}

	refEmbeds, err := s.embedAll(ctx, refSents)
	if err != nil {
		return 0, fmt.Errorf("sentence-coverage: embed reference: %w", err)
	}

	candEmbeds, err := s.embedAll(ctx, candSents)
	if err != nil {
		return 0, fmt.Errorf("sentence-coverage: embed candidate: %w", err)
	}

	// Recall: for each reference sentence, find best match in candidate
	recall := greedyMatchScore(refEmbeds, candEmbeds)

	// Precision: for each candidate sentence, find best match in reference
	precision := greedyMatchScore(candEmbeds, refEmbeds)

	if recall+precision == 0 {
		return 0, nil
	}
	return 2 * precision * recall / (precision + recall), nil
}

// RecallOnly returns just the recall component — useful as a standalone signal
// measuring how much of the reference information the candidate covers.
func (s *SentenceCoverageScorer) RecallOnly(ctx context.Context, reference, candidate string) (float64, error) {
	refSents := splitSentences(reference)
	candSents := splitSentences(candidate)

	if len(refSents) == 0 || len(candSents) == 0 {
		return s.fullTextSimilarity(ctx, reference, candidate)
	}

	refEmbeds, err := s.embedAll(ctx, refSents)
	if err != nil {
		return 0, err
	}

	candEmbeds, err := s.embedAll(ctx, candSents)
	if err != nil {
		return 0, err
	}

	return greedyMatchScore(refEmbeds, candEmbeds), nil
}

func (s *SentenceCoverageScorer) embedAll(ctx context.Context, sentences []string) ([][]float64, error) {
	embeds := make([][]float64, len(sentences))
	for i, sent := range sentences {
		out, err := s.Provider.Embed(ctx, EmbedInput{Model: s.Model, Input: sent})
		if err != nil {
			return nil, fmt.Errorf("embed sentence %d: %w", i, err)
		}
		if len(out.Embeddings) == 0 {
			return nil, fmt.Errorf("empty embedding for sentence %d", i)
		}
		embeds[i] = out.Embeddings[0]
	}
	return embeds, nil
}

func (s *SentenceCoverageScorer) fullTextSimilarity(ctx context.Context, a, b string) (float64, error) {
	embA, err := s.Provider.Embed(ctx, EmbedInput{Model: s.Model, Input: a})
	if err != nil {
		return 0, err
	}
	embB, err := s.Provider.Embed(ctx, EmbedInput{Model: s.Model, Input: b})
	if err != nil {
		return 0, err
	}
	return cosineSimilarity(embA.Embeddings[0], embB.Embeddings[0]), nil
}

// greedyMatchScore computes for each embedding in `from`, the maximum cosine
// similarity with any embedding in `to`, then returns the average.
// This is analogous to BERTScore's greedy matching but at sentence level.
func greedyMatchScore(from, to [][]float64) float64 {
	var total float64
	for _, f := range from {
		best := -1.0
		for _, t := range to {
			sim := cosineSimilarity(f, t)
			if sim > best {
				best = sim
			}
		}
		if best < 0 {
			best = 0
		}
		total += best
	}
	return total / float64(len(from))
}

// splitSentences splits text into sentences using punctuation boundaries.
// Handles common abbreviations and edge cases for technical text.
func splitSentences(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		current.WriteRune(runes[i])

		if runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
			// Look ahead: is the next non-space char uppercase or end of text?
			// This helps avoid splitting on abbreviations like "e.g." or "i.e."
			j := i + 1
			for j < len(runes) && runes[j] == ' ' {
				j++
			}

			isEnd := j >= len(runes)
			isNewSentence := !isEnd && unicode.IsUpper(runes[j])

			if isEnd || isNewSentence {
				trimmed := strings.TrimSpace(current.String())
				if len(trimmed) > 3 { // skip fragments shorter than 3 chars
					sentences = append(sentences, trimmed)
				}
				current.Reset()
			}
		}
	}

	// Don't forget trailing text without terminal punctuation
	if trimmed := strings.TrimSpace(current.String()); len(trimmed) > 3 {
		sentences = append(sentences, trimmed)
	}

	// If splitting produced only 1 sentence, return nil to trigger full-text fallback
	if len(sentences) <= 1 {
		return nil
	}

	return sentences
}
