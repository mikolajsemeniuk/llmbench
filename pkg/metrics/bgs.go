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
// and candidate summary C, the canonical formulation combines a
// sentence-level grounding term with an anchor-coverage term:
//
//	Recall    R    = mean over c_j of  max_i  cos(emb(c_j), emb(s_i))
//	Anchor a(j)    = argmax_i cos(emb(c_j), emb(s_i))
//	Coverage C    = |{distinct a(j)}| / m
//	BGS       = R · C^α
//
// Recall is the standard "is each summary sentence anchored in some
// source sentence?" — penalises hallucination. Coverage asks "do the
// summary sentences spread across the source, or do they all collapse
// onto a single fact?" — penalises within-summary redundancy and
// rewards information breadth, in the spirit of Maximal Marginal
// Relevance (Carbonell & Goldstein 1998) and sub-modular summarisation
// (Lin & Bilmes 2011), but evaluated on the candidate summary rather
// than used as a selection objective.
//
// Legacy mode (LegacyPrecision=true) preserves the original
// bidirectional formulation with a precision side over a salient core
// of source sentences (top-k% by degree centrality) combined via F_β.
// Empirically this trades coverage of three SummEval dimensions
// (coh/con/flu) for a small gain on relevance and is reported as an
// ablation rather than the canonical configuration.
//
// Differences from existing baselines in the repo:
//   - BERTScore: token-level F1 against a *reference* summary
//     (reference-based). BGS is sentence-level and reference-free.
//   - SMART-Model: reference-based sentence matching with a different
//     soft-alignment aggregation. BGS uses simple max-pool plus the
//     anchor-coverage term, with the source as the grounding target.
//   - EmbedScorer: a single whole-text cosine with no sentence
//     structure. BGS exploits sentence granularity and anchor diversity.
type BGS struct {
	Embedder   *Ollama
	EmbedModel string

	// LeadBiasLambda imposes an exponential position prior on source
	// sentences when computing recall. Each cosine cos(c_j, s_i) is
	// multiplied by w(i) = exp(−λ · i / n) before the top-k aggregation,
	// where i is the source-sentence index (0 = lead) and n is the
	// source length. λ=0 (default) disables the prior and reproduces
	// position-agnostic recall. λ>0 down-weights later sentences,
	// reflecting the well-documented lead bias of CNN/DailyMail-style
	// news. Selected on the dev split.
	LeadBiasLambda float64

	// RecallTopK selects how many top-cosine source sentences contribute
	// to each candidate sentence's recall score. The recall component is
	//   recall = mean_j  (1/k) · Σ_{top-k cos(c_j, s_i)}
	// k=1 (default) reduces to the standard mean-of-max formulation
	// (each candidate sentence pulled toward its single best source
	// match). k>1 averages the top-k cosines, giving credit to
	// candidates supported by multiple source sentences (paraphrastic
	// fusion, distributed grounding). When k exceeds the source size,
	// it is clipped to the source size. Selected on the dev split.
	RecallTopK int

	// CoverageAlpha is the exponent α in the canonical scoring rule
	//   score = recall · coverage^α · diversity^γ
	// where coverage = |{distinct argmax-source-anchors}| / m, with
	// m the number of candidate sentences. α=0 disables the coverage
	// term (coverage^0 = 1). α>0 rewards summaries whose sentences
	// anchor to distinct source sentences, penalising collapsed /
	// redundant summaries that paraphrase the same fact m times.
	CoverageAlpha float64

	// RedundancyGamma is the exponent γ in the canonical scoring rule
	//   score = recall · coverage^α · diversity^γ
	// where diversity = 1 − mean_pairwise_cos(c_j, c_k) computed over
	// all j<k pairs of candidate sentences. γ=0 disables the diversity
	// term (diversity^0 = 1). γ>0 penalises candidates whose sentences
	// are mutually similar (paraphrastic redundancy). Distinct from
	// CoverageAlpha: redundancy operates on the candidate alone (no
	// source involvement); coverage operates on where in the source
	// the candidate sentences anchor. Both can be active at once.
	//
	// The canonical α and γ values are selected on a held-out dev
	// split (cmd/bgs runs with -doc-split first50, sweep over α and
	// γ grids); the chosen values are then evaluated on the test
	// split (-doc-split last50).
	RedundancyGamma float64

	// LegacyPrecision enables the original BGS-F_β path: a precision
	// side over salient-core source sentences combined with recall via
	// F_β. Kept for the ablation comparison in paper/ablation.tex.
	// When false (default), the canonical recall · coverage^α formula
	// is used.
	LegacyPrecision bool

	// SalienceTopFrac is the fraction of source sentences that count
	// as the "salient core" used in the LEGACY precision side. Default
	// 0.30 (top 30% by degree centrality). Range (0, 1]. Set to 1.0 to
	// disable salience filtering (precision over all source sentences).
	// Has no effect unless LegacyPrecision=true.
	SalienceTopFrac float64

	// SalienceMin is a floor on the size of the legacy salient core.
	// Avoids degenerate single-sentence cores on very short documents.
	// Has no effect unless LegacyPrecision=true.
	SalienceMin int

	// Beta controls the recall-vs-precision weighting in the LEGACY
	// F_β path: F_β = (1+β²)·P·R / (β²·P + R). β=1 is the harmonic
	// mean (F1); β>1 weights recall more. Has no effect unless
	// LegacyPrecision=true.
	Beta float64

	// RecallOnly forces the score to be the recall component only,
	// bypassing both the coverage term and the legacy precision side.
	// Equivalent to CoverageAlpha=0 with LegacyPrecision=false but
	// preserved as an explicit flag for the ablation row that
	// originally reported "no second component".
	RecallOnly bool

	// MinSentenceLen drops sentences shorter than this many runes
	// after trimming. Avoids degenerate inputs ("Yes.", "OK.").
	MinSentenceLen int
}

func NewBGS(host, embedModel string) *BGS {
	return &BGS{
		Embedder:       NewOllama(host),
		EmbedModel:     embedModel,
		MinSentenceLen: 4,
		// Canonical defaults: pure recall (k=1, α=γ=0, λ=0). cmd/bgs
		// sets the values selected on the held-out dev split.
		RecallTopK:      1,
		CoverageAlpha:   0.0,
		RedundancyGamma: 0.0,
		LeadBiasLambda:  0.0,
		// Legacy precision-side defaults (only consulted when
		// LegacyPrecision=true). Preserved so the legacy ablation
		// rows in paper/ablation.tex remain reproducible.
		SalienceTopFrac: 0.30,
		SalienceMin:     3,
		Beta:            2.0,
	}
}

// BGSDiagnostics carries per-call internals so cmd/bgs can produce
// run-level statistics without re-running the pipeline. FScore is the
// final scalar returned for correlation; the rest are decomposition
// terms (recall, coverage, diversity, optional precision side from
// the legacy path).
type BGSDiagnostics struct {
	NumSourceSents    int
	NumCandidateSents int
	NumSalientSents   int
	DistinctAnchors   int
	Recall            float64
	Coverage          float64
	Diversity         float64
	Precision         float64
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

	// Recall (top-k mean, optional lead-bias prior) + anchors. For
	// each candidate sentence we compute weighted cosines to every
	// source sentence, then average the top-k largest. k=1 reproduces
	// the original mean-of-max formulation; λ=0 reproduces the
	// position-agnostic source weighting. The anchor (argmax index)
	// is recorded separately for the downstream coverage term.
	k := b.RecallTopK
	if k < 1 {
		k = 1
	}
	if k > len(sourceEmbs) {
		k = len(sourceEmbs)
	}

	// Pre-compute the source-position weights once per source. λ=0
	// degenerates to all-ones, which we skip multiplying.
	var posWeights []float64
	useLeadBias := b.LeadBiasLambda > 0
	if useLeadBias {
		n := len(sourceEmbs)
		posWeights = make([]float64, n)
		for i := range n {
			posWeights[i] = math.Exp(-b.LeadBiasLambda * float64(i) / float64(n))
		}
	}

	var recall float64
	anchors := make([]int, len(candEmbs))
	cosBuf := make([]float64, len(sourceEmbs))
	for j, ce := range candEmbs {
		best := -2.0
		bestIdx := 0
		for i, se := range sourceEmbs {
			c := cosineSimilarity(ce, se)
			if useLeadBias {
				c *= posWeights[i]
			}
			cosBuf[i] = c
			if c > best {
				best = c
				bestIdx = i
			}
		}
		anchors[j] = bestIdx

		// Top-k mean. For k=1 this equals best (the argmax cosine);
		// for larger k we sort the cosine slice descending and average
		// the first k entries.
		if k == 1 {
			recall += best
			continue
		}
		sortDescending(cosBuf)
		var sum float64
		for i := 0; i < k; i++ {
			sum += cosBuf[i]
		}
		recall += sum / float64(k)
	}
	recall /= float64(len(candEmbs))
	if recall < 0 {
		recall = 0
	}

	// Anchor coverage: fraction of distinct source-sentence anchors among
	// candidate sentences. Range [1/m, 1]. A redundant summary whose
	// sentences all paraphrase the same source fact collapses to a single
	// anchor → coverage = 1/m. A diverse summary covering m distinct
	// source facts → coverage = 1.
	seen := make(map[int]struct{}, len(anchors))
	for _, a := range anchors {
		seen[a] = struct{}{}
	}
	coverage := float64(len(seen)) / float64(len(candEmbs))

	// Within-summary diversity: 1 − mean pairwise cosine among
	// candidate sentences. Single-sentence candidates have no pairs
	// and default to diversity=1 (no penalty). Range typically
	// [0.05, 0.7] for natural English summaries. Captures
	// paraphrastic redundancy that anchor coverage misses (two
	// candidates can map to different source anchors yet still
	// say similar things).
	diversity := 1.0
	if len(candEmbs) >= 2 {
		var sumPair float64
		nPair := 0
		for j := 0; j < len(candEmbs); j++ {
			for k := j + 1; k < len(candEmbs); k++ {
				sumPair += cosineSimilarity(candEmbs[j], candEmbs[k])
				nPair++
			}
		}
		diversity = 1.0 - sumPair/float64(nPair)
		if diversity < 0 {
			diversity = 0
		}
	}

	baseDiag := BGSDiagnostics{
		NumSourceSents:    len(sourceSents),
		NumCandidateSents: len(candSents),
		DistinctAnchors:   len(seen),
		Recall:            recall,
		Coverage:          coverage,
		Diversity:         diversity,
	}

	// Recall-only mode bypasses both coverage and the legacy precision
	// side. Kept as an explicit flag so the original "no second
	// component" ablation row stays reproducible.
	if b.RecallOnly {
		baseDiag.FScore = recall
		return recall, baseDiag, nil
	}

	// Legacy mode: original BGS-F_β with salient-core precision side.
	// Preserved for the ablation table that motivated the redesign.
	if b.LegacyPrecision {
		salient := selectSalientIndices(sourceEmbs, b.SalienceTopFrac, b.SalienceMin)

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

		baseDiag.NumSalientSents = len(salient)
		baseDiag.Precision = precision
		baseDiag.FScore = fbeta
		return fbeta, baseDiag, nil
	}

	// Canonical mode: recall · coverage^α · diversity^γ. With α=γ=0
	// this reduces to recall-only (any value to the zeroth power is 1).
	// α>0 enables anchor coverage; γ>0 enables within-summary
	// diversity; both can be active simultaneously.
	alpha := b.CoverageAlpha
	if alpha < 0 {
		alpha = 0
	}
	gamma := b.RedundancyGamma
	if gamma < 0 {
		gamma = 0
	}
	score := recall * math.Pow(coverage, alpha) * math.Pow(diversity, gamma)

	baseDiag.FScore = score
	return score, baseDiag, nil
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

// sortDescending sorts a float64 slice in descending order in place.
// Wraps stdlib's ascending sort with a reverse view to avoid an
// allocation that the closure-based sort.Slice would incur.
func sortDescending(xs []float64) {
	sort.Sort(sort.Reverse(sort.Float64Slice(xs)))
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
