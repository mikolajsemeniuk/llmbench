// cmd/bgs runs Bidirectional Grounding Score (BGS), our reference-free
// embedding-only metric, on SummEval.
//
// Canonical formulation:
//
//	score = recall · coverage^α · diversity^γ
//	  recall    = mean_j max_i cos(c_j, s_i)
//	  anchor a(j) = argmax_i cos(c_j, s_i)
//	  coverage  = |{distinct a(j)}| / m
//	  diversity = 1 − mean_{j<k} cos(c_j, c_k)   (within-summary)
//
// α and γ are selected on a held-out development split
// (-doc-split first50) and the chosen values are then evaluated on
// the test split (-doc-split last50) or on the full set
// (-doc-split all). With α=γ=0 the formula reduces to recall.
//
// Per-document caching: SummEval has 16 candidates per article. We
// embed the source's sentences ONCE per DocumentID and reuse them
// across the 16 candidates that share the same source.
//
// Ablation flags:
//
//	-coverage-alpha 0       → coverage term off (default)
//	-coverage-alpha 1.0     → balanced anchor-coverage penalty
//	-redundancy-gamma 0     → diversity term off (default)
//	-redundancy-gamma 1.0   → balanced within-summary diversity penalty
//	-recall-only            → explicit recall-only mode (same as α=γ=0)
//	-legacy-precision       → original BGS-F_β with salient-core precision
//	-doc-split first50      → first 50 articles (dev)
//	-doc-split last50       → last 50 articles (test, held-out)
//	-doc-split all          → full SummEval (1600 samples)
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
	"github.com/mikolajsemeniuk/llmbench/pkg/metrics"
	"github.com/schollz/progressbar/v3"
)

var (
	input           string
	output          string
	embedHost       string
	embedModel      string
	recallTopK      int
	leadBiasLambda  float64
	coverageAlpha   float64
	redundancyGamma float64
	legacyPrecision bool
	salienceFrac    float64
	salienceMin     int
	beta            float64
	recallOnly      bool
	minSentLen      int
	docSplit        string
	n               int
	bootstrap       int
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "output/bgs.json", "write results to file")
	flag.StringVar(&embedHost, "embed-host", "http://localhost:11434", "Ollama host (sentence embeddings)")
	flag.StringVar(&embedModel, "embed-model", "nomic-embed-text", "Ollama embedding model")
	flag.IntVar(&recallTopK, "recall-top-k", 1, "k in recall = mean_j (1/k)·Σ top-k cos(c_j, s_i); k=1 → mean-of-max")
	flag.Float64Var(&leadBiasLambda, "lead-bias-lambda", 0.0, "λ in source weight w(i)=exp(−λ·i/n); 0 disables, larger = stronger lead bias")
	flag.Float64Var(&coverageAlpha, "coverage-alpha", 0.0, "α in score = recall · coverage^α · diversity^γ; 0 disables coverage term")
	flag.Float64Var(&redundancyGamma, "redundancy-gamma", 0.0, "γ in score = recall · coverage^α · diversity^γ; 0 disables diversity term")
	flag.BoolVar(&legacyPrecision, "legacy-precision", false, "use legacy BGS-F_β path (recall + salience-core precision); kept for the ablation row")
	flag.Float64Var(&salienceFrac, "salience-frac", 0.30, "(legacy only) fraction of source sentences in salient core")
	flag.IntVar(&salienceMin, "salience-min", 3, "(legacy only) floor on salient-core size for short documents")
	flag.Float64Var(&beta, "beta", 2.0, "(legacy only) F_β recall-vs-precision weighting; β>1 favours recall")
	flag.BoolVar(&recallOnly, "recall-only", false, "explicit recall-only mode (equivalent to -coverage-alpha 0)")
	flag.IntVar(&minSentLen, "min-sent-len", 4, "drop sentences shorter than this many runes")
	flag.StringVar(&docSplit, "doc-split", "all", "article-level split: all|first50|last50 (held-out hyperparameter selection)")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	if salienceFrac <= 0 || salienceFrac > 1 {
		log.Fatalf("-salience-frac must be in (0, 1], got %v", salienceFrac)
	}
	if beta <= 0 {
		log.Fatalf("-beta must be > 0, got %v", beta)
	}
	if coverageAlpha < 0 {
		log.Fatalf("-coverage-alpha must be ≥ 0, got %v", coverageAlpha)
	}
	if redundancyGamma < 0 {
		log.Fatalf("-redundancy-gamma must be ≥ 0, got %v", redundancyGamma)
	}
	if recallTopK < 1 {
		log.Fatalf("-recall-top-k must be ≥ 1, got %d", recallTopK)
	}
	if leadBiasLambda < 0 {
		log.Fatalf("-lead-bias-lambda must be ≥ 0, got %v", leadBiasLambda)
	}
	switch docSplit {
	case "all", "first50", "last50":
	default:
		log.Fatalf("-doc-split must be all|first50|last50, got %q", docSplit)
	}

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	fsys := os.DirFS(filepath.Dir(input))
	path := filepath.Base(input)
	if input == "" {
		fsys = dataset.Summeval
		path = dataset.SummevalDefaultPath
	}

	samples, err := eval.NewDataset(fsys, path, n)
	if err != nil {
		log.Fatal(err)
	}
	samples = applyDocSplit(samples, docSplit)

	scorer := metrics.NewBGS(embedHost, embedModel)
	scorer.MinSentenceLen = minSentLen
	scorer.RecallTopK = recallTopK
	scorer.LeadBiasLambda = leadBiasLambda
	scorer.CoverageAlpha = coverageAlpha
	scorer.RedundancyGamma = redundancyGamma
	scorer.LegacyPrecision = legacyPrecision
	scorer.SalienceTopFrac = salienceFrac
	scorer.SalienceMin = salienceMin
	scorer.Beta = beta
	scorer.RecallOnly = recallOnly

	switch {
	case recallOnly:
		log.Printf("BGS: recall-only mode (top-k=%d)", recallTopK)
	case legacyPrecision:
		log.Printf("BGS: legacy precision-side mode (top-k=%d, salience top-frac=%.2f, min=%d, beta=%.2f)",
			recallTopK, salienceFrac, salienceMin, beta)
	default:
		log.Printf("BGS: canonical mode (top-k=%d, λ=%.3f lead-bias, α=%.3f coverage, γ=%.3f diversity), split=%s, %d samples",
			recallTopK, leadBiasLambda, coverageAlpha, redundancyGamma, docSplit, len(samples))
	}

	type cached struct {
		sents []string
		embs  [][]float64
	}
	cache := make(map[string]*cached)

	bar := progressbar.NewOptions(
		len(samples),
		progressbar.OptionSetDescription("bgs"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	scores := make([]float64, len(samples))
	entries := make([]eval.Score, len(samples))

	var sumR, sumCov, sumDiv, sumScore float64
	var sumDistinct int

	start := time.Now()

	for i, s := range samples {
		c, ok := cache[s.DocumentID]
		if !ok {
			sents := filteredSplit(s.Document, minSentLen)
			if len(sents) == 0 {
				log.Fatalf("sample %s: source has no usable sentences", s.ID)
			}
			embs, err := scorer.EmbedSentences(ctx, sents)
			if err != nil {
				log.Fatalf("sample %s: embed source: %v", s.ID, err)
			}
			c = &cached{sents: sents, embs: embs}
			cache[s.DocumentID] = c
		}

		score, diag, err := scorer.ScoreWithSourceEmbeddings(ctx, c.sents, c.embs, s.Candidate)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}

		scores[i] = score
		entries[i] = eval.Score{SampleID: s.ID, Value: score}

		sumR += diag.Recall
		sumCov += diag.Coverage
		sumDiv += diag.Diversity
		sumScore += diag.FScore
		sumDistinct += diag.DistinctAnchors

		bar.Add(1)
	}

	elapsed := time.Since(start)
	N := float64(len(samples))
	log.Printf("BGS: %d samples in %.1fs — mean R=%.3f, cov=%.3f, div=%.3f, score=%.3f, distinct anchors=%.1f",
		len(samples), elapsed.Seconds(),
		sumR/N, sumCov/N, sumDiv/N, sumScore/N, float64(sumDistinct)/N)

	var norm string
	switch {
	case recallOnly:
		norm = fmt.Sprintf("recall_only=true,top_k=%d", recallTopK)
	case legacyPrecision:
		norm = fmt.Sprintf("legacy_precision=true,salience=%.2f,beta=%.2f,top_k=%d", salienceFrac, beta, recallTopK)
	default:
		norm = fmt.Sprintf("top_k=%d,lead_lambda=%.3f,coverage_alpha=%.3f,redundancy_gamma=%.3f,split=%s",
			recallTopK, leadBiasLambda, coverageAlpha, redundancyGamma, docSplit)
	}
	report := eval.Report{
		Metric:     "bgs",
		Norm:       norm,
		Samples:    len(samples),
		RuntimeSec: elapsed.Seconds(),
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Scores:     entries,
		SummaryLevel: eval.NewCorrelationWith(samples, scores, eval.CorrelationOptions{
			Bootstrap: bootstrap,
			Level:     "summary",
		}),
		SystemLevel: eval.NewCorrelationWith(samples, scores, eval.CorrelationOptions{
			Bootstrap: bootstrap,
			Level:     "system",
		}),
	}
	if err := eval.NewReport(output, report); err != nil {
		log.Fatal(err)
	}
}

// filteredSplit splits a document into sentences and drops degenerate ones,
// matching the filter the metric applies internally.
func filteredSplit(text string, minLen int) []string {
	raw := metrics.SplitSentencesForBGS(text)
	out := raw[:0]
	for _, s := range raw {
		if len([]rune(s)) >= minLen {
			out = append(out, s)
		}
	}
	return out
}

// applyDocSplit slices the loaded sample list into a document-level
// development or test half. SummEval has 16 candidates per article so
// the boundary always lands cleanly between articles. The split is by
// dataset order, which matches eval.NewDataset's JSONL emission order
// (deterministic across runs).
func applyDocSplit(samples []eval.Sample, split string) []eval.Sample {
	if split == "all" {
		return samples
	}
	docs := uniqueDocsInOrder(samples)
	if len(docs) < 100 {
		log.Fatalf("doc-split %q expects ≥100 documents, got %d", split, len(docs))
	}
	var keep map[string]struct{}
	switch split {
	case "first50":
		keep = setOf(docs[:50])
	case "last50":
		keep = setOf(docs[len(docs)-50:])
	default:
		log.Fatalf("invalid doc-split: %s", split)
	}
	out := samples[:0]
	for _, s := range samples {
		if _, ok := keep[s.DocumentID]; ok {
			out = append(out, s)
		}
	}
	return out
}

func uniqueDocsInOrder(samples []eval.Sample) []string {
	seen := make(map[string]struct{})
	var docs []string
	for _, s := range samples {
		if _, ok := seen[s.DocumentID]; ok {
			continue
		}
		seen[s.DocumentID] = struct{}{}
		docs = append(docs, s.DocumentID)
	}
	return docs
}

func setOf(ids []string) map[string]struct{} {
	out := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		out[id] = struct{}{}
	}
	return out
}
