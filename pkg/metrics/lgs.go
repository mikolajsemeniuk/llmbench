package metrics

import (
	"context"
	"fmt"
	"math"
)

// LGS — reference-free, embedding-only summary-quality metric. For source
// document D and candidate summary C split into sentences:
//
//	w(i)  = exp(−λ · i / n)                    // exponential lead-bias prior
//	score = mean over c_j of  max_i  w(i) · cos(emb(c_j), emb(s_i))
//
// where i is the source-sentence index (0 = lead, n = source length) and
// j ranges over candidate sentences. Each candidate sentence is anchored
// to its single best-matching source sentence after the cosine has been
// weighted by source position. The candidate-side mean and source-side
// argmax give a localised "is each summary sentence grounded in some
// salient part of the source?" signal — penalising hallucination on
// the candidate side and incorporating the well-documented lead bias of
// CNN/DailyMail-style news on the source side.
//
// λ is the only hyperparameter and is selected on a held-out development
// split (see cmd/lgs and Makefile). λ=0 disables the prior and reproduces
// position-agnostic mean-of-max recall.
//
// Differences from the closest reference-based baselines:
//   - BERTScore: token-level F1 against a reference summary. LGS is
//     sentence-level, source-grounded, reference-free.
//   - SMART-Model: reference-based sentence matching. LGS uses the
//     source as the grounding target and adds the lead-bias prior.
//   - EmbedScorer: a single whole-text cosine. LGS exploits sentence
//     granularity and source-position weighting.
type LGS struct {
	Embedder   *Ollama
	EmbedModel string

	// LeadBiasLambda imposes an exponential position prior on source
	// sentences when computing recall: each cosine cos(c_j, s_i) is
	// multiplied by w(i) = exp(−λ · i / n) before the argmax. λ=0
	// disables the prior. λ>0 down-weights later source sentences.
	// Selected on the dev split.
	LeadBiasLambda float64

	// MinSentenceLen drops sentences shorter than this many runes
	// after trimming. Avoids degenerate inputs ("Yes.", "OK.").
	MinSentenceLen int
}

func NewLGS(host, embedModel string) *LGS {
	return &LGS{
		Embedder:       NewOllama(host),
		EmbedModel:     embedModel,
		MinSentenceLen: 4,
		LeadBiasLambda: 0.0, // overridden by cmd/lgs with the dev-selected λ*
	}
}

// LGSDiagnostics carries per-call internals so cmd/lgs can produce
// run-level statistics without re-running the pipeline. Score is the
// final scalar returned for correlation; Recall is the same value
// (kept as an alias for callers that read the decomposition).
type LGSDiagnostics struct {
	NumSourceSents    int
	NumCandidateSents int
	Recall            float64
	Score             float64
}

func (b *LGS) Score(ctx context.Context, source, candidate string) (float64, error) {
	score, _, err := b.ScoreDetailed(ctx, source, candidate)
	return score, err
}

// ScoreWithSourceEmbeddings reuses precomputed source embeddings (one
// per source sentence). cmd/lgs caches these per DocumentID so the 16
// candidates summarising the same article share the embedding cost.
func (b *LGS) ScoreWithSourceEmbeddings(ctx context.Context,
	sourceSents []string, sourceEmbs [][]float64,
	candidate string,
) (float64, LGSDiagnostics, error) {
	candSents := filterShortSentences(splitSentences(candidate), b.MinSentenceLen)
	if len(sourceSents) == 0 || len(candSents) == 0 {
		return 0, LGSDiagnostics{}, nil
	}
	candEmbs, err := b.embedAll(ctx, candSents)
	if err != nil {
		return 0, LGSDiagnostics{}, fmt.Errorf("lgs: embed candidate: %w", err)
	}
	return b.score(sourceSents, sourceEmbs, candSents, candEmbs)
}

func (b *LGS) ScoreDetailed(ctx context.Context, source, candidate string) (float64, LGSDiagnostics, error) {
	sourceSents := filterShortSentences(splitSentences(source), b.MinSentenceLen)
	candSents := filterShortSentences(splitSentences(candidate), b.MinSentenceLen)
	if len(sourceSents) == 0 || len(candSents) == 0 {
		return 0, LGSDiagnostics{}, nil
	}
	sourceEmbs, err := b.embedAll(ctx, sourceSents)
	if err != nil {
		return 0, LGSDiagnostics{}, fmt.Errorf("lgs: embed source: %w", err)
	}
	candEmbs, err := b.embedAll(ctx, candSents)
	if err != nil {
		return 0, LGSDiagnostics{}, fmt.Errorf("lgs: embed candidate: %w", err)
	}
	return b.score(sourceSents, sourceEmbs, candSents, candEmbs)
}

func (b *LGS) score(
	sourceSents []string, sourceEmbs [][]float64,
	candSents []string, candEmbs [][]float64,
) (float64, LGSDiagnostics, error) {

	// Pre-compute source-position weights once. λ=0 degenerates to
	// all-ones so we skip the multiplication entirely.
	var posWeights []float64
	if b.LeadBiasLambda > 0 {
		n := len(sourceEmbs)
		posWeights = make([]float64, n)
		for i := range n {
			posWeights[i] = math.Exp(-b.LeadBiasLambda * float64(i) / float64(n))
		}
	}

	var recall float64
	for _, ce := range candEmbs {
		best := -2.0
		for i, se := range sourceEmbs {
			c := cosineSimilarity(ce, se)
			if posWeights != nil {
				c *= posWeights[i]
			}
			if c > best {
				best = c
			}
		}
		recall += best
	}
	recall /= float64(len(candEmbs))
	if recall < 0 {
		recall = 0
	}

	return recall, LGSDiagnostics{
		NumSourceSents:    len(sourceSents),
		NumCandidateSents: len(candSents),
		Recall:            recall,
		Score:             recall,
	}, nil
}

// EmbedSentences exposes embedAll for cmd/lgs so per-document caching
// of source-sentence embeddings is possible.
func (b *LGS) EmbedSentences(ctx context.Context, sents []string) ([][]float64, error) {
	return b.embedAll(ctx, sents)
}

// SplitSentencesForLGS exposes the package-private splitter so cmd/lgs
// can produce the same filtered list of source sentences that the
// metric uses internally.
func SplitSentencesForLGS(text string) []string {
	return splitSentences(text)
}

func (b *LGS) embedAll(ctx context.Context, sentences []string) ([][]float64, error) {
	embeds := make([][]float64, len(sentences))
	for i, sent := range sentences {
		resp, err := b.Embedder.Embed(ctx, EmbedInput{Model: b.EmbedModel, Input: sent})
		if err != nil {
			return nil, err
		}
		if len(resp.Embeddings) == 0 {
			return nil, fmt.Errorf("lgs: empty embedding for sentence %d", i)
		}
		embeds[i] = resp.Embeddings[0]
	}
	return embeds, nil
}

func filterShortSentences(sents []string, minLen int) []string {
	out := sents[:0]
	for _, s := range sents {
		if len([]rune(s)) >= minLen {
			out = append(out, s)
		}
	}
	return out
}
