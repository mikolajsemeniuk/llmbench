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

// LGSInput is the per-call input. The caller pre-computes source
// sentence splits and embeddings once per DocumentID and passes them
// in for every candidate of that article.
type LGSInput struct {
	SourceSents []string
	SourceEmbs  [][]float64
	Candidate   string
}

// LGSOutput carries the per-call result.
type LGSOutput struct {
	Score float64
}

func NewLGS(host, embedModel string) *LGS {
	return &LGS{
		Embedder:       NewOllama(host),
		EmbedModel:     embedModel,
		MinSentenceLen: 4,
		LeadBiasLambda: 0.0, // overridden by cmd/lgs with the dev-selected λ*
	}
}

// Score computes LGS for a single (source, candidate) pair. Source
// embeddings are reused across the 16 candidates of an article so the
// caller is responsible for caching them per DocumentID.
func (b *LGS) Score(ctx context.Context, in LGSInput) (LGSOutput, error) {
	candSents := filterShortSentences(SplitSentences(in.Candidate), b.MinSentenceLen)
	if len(in.SourceSents) == 0 || len(candSents) == 0 {
		return LGSOutput{}, nil
	}

	candEmbs, err := b.EmbedSentences(ctx, candSents)
	if err != nil {
		return LGSOutput{}, fmt.Errorf("lgs: embed candidate: %w", err)
	}

	var weights []float64
	if b.LeadBiasLambda > 0 {
		n := len(in.SourceEmbs)
		weights = make([]float64, n)
		for i := range n {
			weights[i] = math.Exp(-b.LeadBiasLambda * float64(i) / float64(n))
		}
	}

	var recall float64
	for _, ce := range candEmbs {
		best := -2.0
		for i, se := range in.SourceEmbs {
			c := cosineSimilarity(ce, se)
			if weights != nil {
				c *= weights[i]
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

	return LGSOutput{Score: recall}, nil
}

// EmbedSentences embeds sentences one at a time using the configured
// Ollama embedder. cmd/lgs uses it directly to embed source sentences
// once per DocumentID; Score uses it internally for the candidate.
func (b *LGS) EmbedSentences(ctx context.Context, sentences []string) ([][]float64, error) {
	embeds := make([][]float64, len(sentences))
	for i, sent := range sentences {
		res, err := b.Embedder.Embed(ctx, EmbedInput{Model: b.EmbedModel, Input: sent})
		if err != nil {
			return nil, err
		}
		if len(res.Embeddings) == 0 {
			return nil, fmt.Errorf("lgs: empty embedding for sentence %d", i)
		}
		embeds[i] = res.Embeddings[0]
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
