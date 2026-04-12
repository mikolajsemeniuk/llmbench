package llmbench

import (
	"context"
	"fmt"
)

// HybridScorer combines multiple evaluation signals into a single score.
//
// Architecture (4 signals):
//   - Embedding similarity (sentence-level cosine via Ollama)
//   - Cross-encoder relevance (bidirectional, via local reranker)
//   - Sentence-level coverage F1 (SMART-inspired)
//   - Lexical overlap (ROUGE-L as lightweight baseline signal)
//
// The key insight is that each signal captures a different aspect of quality:
//   - Embeddings capture semantic proximity
//   - Cross-encoder captures deep token-level interactions
//   - Sentence coverage captures information completeness
//   - ROUGE-L captures surface-level fidelity
//
// Combining them yields better correlation with human judgments than any
// single signal, while remaining 10-50x faster than LLM-as-Judge.
type HybridScorer struct {
	Embed        *BERTScorer
	CrossEncoder *CrossEncoderScorer
	SentCoverage *SentenceCoverageScorer

	// Weights for each signal (should sum to 1.0).
	WEmbed   float64
	WCross   float64
	WSentCov float64
	WRougeL  float64
}

// DefaultHybridWeights returns weights that work well as a starting point.
// These can be optimized on a dataset with human annotations.
func DefaultHybridWeights() (wEmbed, wCross, wSentCov, wRougeL float64) {
	return 0.20, 0.35, 0.35, 0.10
}

func NewHybridScorer(ollamaHost, embedModel, rerankerHost string) *HybridScorer {
	wE, wC, wS, wR := DefaultHybridWeights()
	return &HybridScorer{
		Embed:        NewBERTScorer(ollamaHost, embedModel),
		CrossEncoder: NewCrossEncoderScorer(rerankerHost),
		SentCoverage: NewSentenceCoverageScorer(ollamaHost, embedModel),
		WEmbed:       wE,
		WCross:       wC,
		WSentCov:     wS,
		WRougeL:      wR,
	}
}

// Score computes the hybrid metric for a (question, reference, candidate) triple.
// The question parameter is currently unused but reserved for future
// question-relevance signal (Signal 4 in the paper design).
func (h *HybridScorer) Score(ctx context.Context, question, reference, candidate string) (float64, error) {
	// Signal 1: Sentence-level embedding cosine similarity
	embedScore, err := h.Embed.Score(ctx, reference, candidate)
	if err != nil {
		return 0, fmt.Errorf("hybrid: embed: %w", err)
	}

	// Signal 2: Cross-encoder bidirectional relevance
	crossScore, err := h.CrossEncoder.BidirectionalScore(ctx, reference, candidate)
	if err != nil {
		return 0, fmt.Errorf("hybrid: cross-encoder: %w", err)
	}

	// Signal 3: Sentence-level coverage F1
	sentScore, err := h.SentCoverage.Score(ctx, reference, candidate)
	if err != nil {
		return 0, fmt.Errorf("hybrid: sentence-coverage: %w", err)
	}

	// Signal 4: Lexical overlap (free, no API call)
	rougeScore := ROUGEL(reference, candidate)

	score := h.WEmbed*embedScore +
		h.WCross*crossScore +
		h.WSentCov*sentScore +
		h.WRougeL*rougeScore

	return score, nil
}

// ScoreDetailed returns the composite score along with all individual signals.
// Useful for analysis and debugging.
func (h *HybridScorer) ScoreDetailed(ctx context.Context, question, reference, candidate string) (HybridDetail, error) {
	embedScore, err := h.Embed.Score(ctx, reference, candidate)
	if err != nil {
		return HybridDetail{}, fmt.Errorf("hybrid: embed: %w", err)
	}

	crossScore, err := h.CrossEncoder.BidirectionalScore(ctx, reference, candidate)
	if err != nil {
		return HybridDetail{}, fmt.Errorf("hybrid: cross-encoder: %w", err)
	}

	sentScore, err := h.SentCoverage.Score(ctx, reference, candidate)
	if err != nil {
		return HybridDetail{}, fmt.Errorf("hybrid: sentence-coverage: %w", err)
	}

	rougeScore := ROUGEL(reference, candidate)

	composite := h.WEmbed*embedScore +
		h.WCross*crossScore +
		h.WSentCov*sentScore +
		h.WRougeL*rougeScore

	return HybridDetail{
		Composite: composite,
		Embed:     embedScore,
		CrossEnc:  crossScore,
		SentCov:   sentScore,
		RougeL:    rougeScore,
	}, nil
}

// HybridDetail holds the composite score and all individual signal scores.
type HybridDetail struct {
	Composite float64 `json:"composite"`
	Embed     float64 `json:"embed"`
	CrossEnc  float64 `json:"cross_enc"`
	SentCov   float64 `json:"sent_cov"`
	RougeL    float64 `json:"rouge_l"`
}

// ──────────────────────────────────────────────────────────────────────────
// Ablation variants — each omits one signal to measure its contribution.
// These are essential for the paper's ablation study section.
// ──────────────────────────────────────────────────────────────────────────

// HybridNoCrossScorer is an ablation variant without the cross-encoder signal.
// Useful when the reranker is unavailable or for ablation studies.
type HybridNoCrossScorer struct {
	Embed        *BERTScorer
	SentCoverage *SentenceCoverageScorer
}

func NewHybridNoCrossScorer(ollamaHost, embedModel string) *HybridNoCrossScorer {
	return &HybridNoCrossScorer{
		Embed:        NewBERTScorer(ollamaHost, embedModel),
		SentCoverage: NewSentenceCoverageScorer(ollamaHost, embedModel),
	}
}

func (h *HybridNoCrossScorer) Score(ctx context.Context, question, reference, candidate string) (float64, error) {
	embedScore, err := h.Embed.Score(ctx, reference, candidate)
	if err != nil {
		return 0, err
	}

	sentScore, err := h.SentCoverage.Score(ctx, reference, candidate)
	if err != nil {
		return 0, err
	}

	rougeScore := ROUGEL(reference, candidate)

	// Redistribute cross-encoder weight proportionally
	return 0.30*embedScore + 0.55*sentScore + 0.15*rougeScore, nil
}
