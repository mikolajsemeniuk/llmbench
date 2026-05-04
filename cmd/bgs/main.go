// cmd/bgs runs Bidirectional Grounding Score (BGS), our reference-free
// embedding-only metric, on SummEval. BGS = F1 of:
//
//   - Recall: each candidate sentence's max cosine to any source sentence
//   - Precision: each salient-core source sentence's max cosine to any
//     candidate sentence
//
// The salient core is the top-k% source sentences by degree centrality.
//
// Per-document caching: SummEval has 16 candidates per article. We
// embed the source's sentences ONCE per DocumentID and reuse them
// across the 16 candidates that share the same source.
//
// Ablation flags (used by `make benchmark-ablation` to populate
// ablation/*.json):
//
//	-salience-frac 0.30   → default (top-30% as salient core)
//	-salience-frac 1.00   → no salience filter (precision over all sources)
//	-salience-frac 0.10   → very tight salient core
//	-beta 1               → F1 (precision and recall equally weighted)
//	-beta 2               → recall-biased F (canonical, β² = 4× weight on R)
//	-recall-only          → drop precision side entirely (R alone)
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
	input        string
	output       string
	embedHost    string
	embedModel   string
	salienceFrac float64
	salienceMin  int
	beta         float64
	recallOnly   bool
	minSentLen   int
	n            int
	bootstrap    int
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "output/bgs.json", "write results to file")
	flag.StringVar(&embedHost, "embed-host", "http://localhost:11434", "Ollama host (sentence embeddings)")
	flag.StringVar(&embedModel, "embed-model", "nomic-embed-text", "Ollama embedding model")
	flag.Float64Var(&salienceFrac, "salience-frac", 0.30, "fraction of source sentences in salient core (precision side); 1.0 = no filter")
	flag.IntVar(&salienceMin, "salience-min", 3, "floor on salient-core size for short documents")
	flag.Float64Var(&beta, "beta", 2.0, "F_β recall-vs-precision weighting; β>1 favours recall (β=2 default from sweep, β=1 → F1)")
	flag.BoolVar(&recallOnly, "recall-only", false, "skip precision side entirely; score = recall (ablation row in paper/ablation.tex)")
	flag.IntVar(&minSentLen, "min-sent-len", 4, "drop sentences shorter than this many runes")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	if salienceFrac <= 0 || salienceFrac > 1 {
		log.Fatalf("-salience-frac must be in (0, 1], got %v", salienceFrac)
	}
	if beta <= 0 {
		log.Fatalf("-beta must be > 0, got %v", beta)
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

	scorer := metrics.NewBGS(embedHost, embedModel)
	scorer.SalienceTopFrac = salienceFrac
	scorer.SalienceMin = salienceMin
	scorer.MinSentenceLen = minSentLen
	scorer.Beta = beta
	scorer.RecallOnly = recallOnly

	if recallOnly {
		log.Printf("BGS: recall-only mode (precision side disabled)")
	} else {
		log.Printf("BGS: salience top-frac=%.2f, min=%d, beta=%.2f", salienceFrac, salienceMin, beta)
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

	var sumP, sumR, sumF1 float64
	var sumSalient int

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

		sumP += diag.Precision
		sumR += diag.Recall
		sumF1 += diag.FScore
		sumSalient += diag.NumSalientSents

		bar.Add(1)
	}

	elapsed := time.Since(start)
	N := float64(len(samples))
	log.Printf("BGS: %d samples in %.1fs — mean P=%.3f, R=%.3f, F1=%.3f, mean salient core=%.1f sents",
		len(samples), elapsed.Seconds(),
		sumP/N, sumR/N, sumF1/N, float64(sumSalient)/N)

	norm := fmt.Sprintf("salience=%.2f,beta=%.2f", salienceFrac, beta)
	if recallOnly {
		norm = "recall_only=true"
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
