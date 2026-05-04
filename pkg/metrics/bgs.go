package metrics

import (
	"context"
	"fmt"
	"math"
	"sort"
)

// BGS — Bidirectional Grounding Score.
//
// Reference-free, embedding-only summary-quality metric. For source D
// and candidate summary C:
//
//	Recall    R = mean over c_j of  max_i  cos(emb(c_j), emb(s_i))
//	Precision P = mean over s_i ∈ salient(D) of  max_j  cos(emb(s_i), emb(c_j))
//	BGS       = F1(P, R) = 2·P·R / (P + R)
//
// Recall is the standard "is each summary sentence anchored in some
// source sentence?" — penalises hallucination. Precision asks "is each
// salient source sentence covered by some summary sentence?" — penalises
// omission of important content.
//
// Salience filter on the source side is necessary because summaries are
// short by design: requiring coverage of *every* source sentence would
// floor precision for any normal summary. We pick a salient core via
// degree centrality on the source's pairwise cosine graph (sentences
// most similar to the rest of the document), which is a cheap proxy
// for extractive importance — the same intuition behind TextRank /
// LexRank without their iterative PageRank step.
//
// Differences from existing baselines in the repo:
//   - BERTScore: token-level F1 against a *reference* summary
//     (reference-based). BGS is sentence-level and reference-free.
//   - SMART-Model: reference-based sentence matching with a different
//     soft-alignment aggregation. BGS uses simple max-pool both sides.
//   - EmbedScorer: a single whole-text cosine with no sentence
//     structure. BGS exploits sentence granularity and salience.
type BGS struct {
	Embedder   *Ollama
	EmbedModel string

	// SalienceTopFrac is the fraction of source sentences that count
	// as the "salient core" used in the precision side. Default 0.30
	// (top 30% by degree centrality). Range (0, 1]. Set to 1.0 to
	// disable salience filtering (precision over all source sentences).
	SalienceTopFrac float64

	// SalienceMin is a floor on the size of the salient core. Avoids
	// degenerate single-sentence cores on very short documents.
	SalienceMin int

	// Beta controls the recall-vs-precision weighting of F_β:
	//   F_β = (1+β²)·P·R / (β²·P + R)
	// β = 1 is harmonic mean (F1, default). β > 1 weights recall
	// more (β² times more); β < 1 weights precision more. SummEval
	// shows recall as the stronger individual signal, so β=2 is
	// the natural ablation candidate.
	Beta float64

	// RecallOnly disables the precision side entirely: the score is
	// just the mean candidate→max-source cosine. Used as the bottom
	// row of the ablation table — "what does the metric look like
	// if we drop the bidirectional half?".
	RecallOnly bool

	// MinSentenceLen drops sentences shorter than this many runes
	// after trimming. Avoids degenerate inputs ("Yes.", "OK.").
	MinSentenceLen int
}

func NewBGS(host, embedModel string) *BGS {
	return &BGS{
		Embedder:        NewOllama(host),
		EmbedModel:      embedModel,
		SalienceTopFrac: 0.30,
		SalienceMin:     3,
		MinSentenceLen:  4,
		// Beta=2 chosen empirically: the ablation sweep over β ∈ {1, 2, 3}
		// at default salience showed β=2 gives the best balance — coh
		// ρ=.373 ties BERTScore .377 while preserving the +.019 lift on
		// rel ρ=.356 (vs Grounding-only .337) that motivates the
		// precision side. β=1 (F1) over-weights precision and drags coh
		// down to .341; β=3 squeezes a touch more from coh (.377) but
		// loses on rel (.348). β=2 is also the conventional recall-biased
		// F-measure in IR literature.
		Beta: 2.0,
	}
}

// BGSDiagnostics carries per-call internals so cmd/bgs can produce
// run-level statistics without re-running the pipeline. The FScore
// field is the F_β combined score (β from BGS.Beta); when β=1 it
// reduces to the harmonic mean.
type BGSDiagnostics struct {
	NumSourceSents    int
	NumCandidateSents int
	NumSalientSents   int
	Precision         float64
	Recall            float64
	FScore            float64
}

func (b *BGS) Score(ctx context.Context, source, candidate string) (float64, error) {
	score, _, err := b.ScoreDetailed(ctx, source, candidate)
	return score, err
}

// ScoreWithSourceEmbeddings reuses precomputed source embeddings (one
// per source sentence). cmd/bgs caches these per DocumentID so the 16
// candidates summarising the same article share the embedding cost.
func (b *BGS) ScoreWithSourceEmbeddings(ctx context.Context,
	sourceSents []string, sourceEmbs [][]float64,
	candidate string,
) (float64, BGSDiagnostics, error) {
	candSents := filterShortSentences(splitSentences(candidate), b.MinSentenceLen)
	if len(sourceSents) == 0 || len(candSents) == 0 {
		return 0, BGSDiagnostics{}, nil
	}
	candEmbs, err := b.embedAll(ctx, candSents)
	if err != nil {
		return 0, BGSDiagnostics{}, fmt.Errorf("bgs: embed candidate: %w", err)
	}
	return b.score(sourceSents, sourceEmbs, candSents, candEmbs)
}

func (b *BGS) ScoreDetailed(ctx context.Context, source, candidate string) (float64, BGSDiagnostics, error) {
	sourceSents := filterShortSentences(splitSentences(source), b.MinSentenceLen)
	candSents := filterShortSentences(splitSentences(candidate), b.MinSentenceLen)
	if len(sourceSents) == 0 || len(candSents) == 0 {
		return 0, BGSDiagnostics{}, nil
	}
	sourceEmbs, err := b.embedAll(ctx, sourceSents)
	if err != nil {
		return 0, BGSDiagnostics{}, fmt.Errorf("bgs: embed source: %w", err)
	}
	candEmbs, err := b.embedAll(ctx, candSents)
	if err != nil {
		return 0, BGSDiagnostics{}, fmt.Errorf("bgs: embed candidate: %w", err)
	}
	return b.score(sourceSents, sourceEmbs, candSents, candEmbs)
}

func (b *BGS) score(
	sourceSents []string, sourceEmbs [][]float64,
	candSents []string, candEmbs [][]float64,
) (float64, BGSDiagnostics, error) {

	// Recall: every candidate sentence pulled toward its best source match
	// across the FULL source (no salience filter — the candidate is allowed
	// to ground itself in any source sentence, salient or not).
	var recall float64
	for _, ce := range candEmbs {
		best := -2.0
		for _, se := range sourceEmbs {
			if c := cosineSimilarity(ce, se); c > best {
				best = c
			}
		}
		recall += best
	}
	recall /= float64(len(candEmbs))
	if recall < 0 {
		recall = 0
	}

	// Recall-only mode skips precision and salience selection entirely.
	// The score is the recall component as-is. Used as an ablation
	// "no precision side" reference row.
	if b.RecallOnly {
		return recall, BGSDiagnostics{
			NumSourceSents:    len(sourceSents),
			NumCandidateSents: len(candSents),
			NumSalientSents:   0,
			Precision:         0,
			Recall:            recall,
			FScore:            recall,
		}, nil
	}

	salient := selectSalientIndices(sourceEmbs, b.SalienceTopFrac, b.SalienceMin)

	// Precision: every salient source sentence pulled toward its best
	// candidate match. A candidate that misses many salient core sentences
	// gets a low precision even if all of its sentences happen to be
	// well-grounded (high recall) — this is what catches omission errors.
	var precision float64
	for _, idx := range salient {
		best := -2.0
		for _, ce := range candEmbs {
			if c := cosineSimilarity(sourceEmbs[idx], ce); c > best {
				best = c
			}
		}
		precision += best
	}
	precision /= float64(len(salient))

	// Clamp negatives to 0 before harmonic mean. Cosine similarity is in
	// [-1, 1] but in practice with sentence embeddings on natural English
	// it sits in [0.3, 0.95]. The clamp is defensive — it only fires on
	// pathological inputs and prevents F1 from being undefined.
	if precision < 0 {
		precision = 0
	}

	beta := b.Beta
	if beta <= 0 {
		beta = 1.0
	}
	var fbeta float64
	denom := beta*beta*precision + recall
	if denom > 0 {
		fbeta = (1 + beta*beta) * precision * recall / denom
	}

	return fbeta, BGSDiagnostics{
		NumSourceSents:    len(sourceSents),
		NumCandidateSents: len(candSents),
		NumSalientSents:   len(salient),
		Precision:         precision,
		Recall:            recall,
		FScore:            fbeta,
	}, nil
}

// EmbedSentences exposes embedAll for cmd/bgs so per-document caching
// of source-sentence embeddings is possible.
func (b *BGS) EmbedSentences(ctx context.Context, sents []string) ([][]float64, error) {
	return b.embedAll(ctx, sents)
}

// SplitSentencesForBGS exposes the package-private splitter so cmd/bgs
// can produce the same filtered list of source sentences that the
// metric uses internally.
func SplitSentencesForBGS(text string) []string {
	return splitSentences(text)
}

func (b *BGS) embedAll(ctx context.Context, sentences []string) ([][]float64, error) {
	embeds := make([][]float64, len(sentences))
	for i, sent := range sentences {
		resp, err := b.Embedder.Embed(ctx, EmbedInput{Model: b.EmbedModel, Input: sent})
		if err != nil {
			return nil, err
		}
		if len(resp.Embeddings) == 0 {
			return nil, fmt.Errorf("bgs: empty embedding for sentence %d", i)
		}
		embeds[i] = resp.Embeddings[0]
	}
	return embeds, nil
}

// selectSalientIndices picks the top-k source-sentence indices by degree
// centrality on the pairwise cosine graph. Degree centrality of s_i is
// the sum of cos(s_i, s_j) for j ≠ i — sentences most similar to the
// rest of the document. Returns indices in ascending order.
//
// k = max(SalienceMin, ceil(SalienceTopFrac · n)), clamped to n.
// When k == n the whole source counts as salient (no filter).
func selectSalientIndices(sourceEmbs [][]float64, fraction float64, minK int) []int {
	n := len(sourceEmbs)
	if n == 0 {
		return nil
	}

	k := int(math.Ceil(float64(n) * fraction))
	k = max(k, minK)
	k = min(k, n)

	if k == n {
		out := make([]int, n)
		for i := range out {
			out[i] = i
		}
		return out
	}

	score := make([]float64, n)
	for i := range n {
		for j := i + 1; j < n; j++ {
			c := cosineSimilarity(sourceEmbs[i], sourceEmbs[j])
			score[i] += c
			score[j] += c
		}
	}

	type indexed struct {
		i int
		s float64
	}
	ranked := make([]indexed, n)
	for i := range n {
		ranked[i] = indexed{i: i, s: score[i]}
	}
	sort.Slice(ranked, func(a, b int) bool { return ranked[a].s > ranked[b].s })

	out := make([]int, k)
	for i := range k {
		out[i] = ranked[i].i
	}
	sort.Ints(out)
	return out
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
