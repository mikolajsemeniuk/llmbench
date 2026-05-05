// cmd/lgs runs LGS, our reference-free embedding-only summary-quality
// metric, on SummEval. The metric is
//
//	w(i)  = exp(−λ · i / n)
//	score = mean over c_j of  max_i  w(i) · cos(emb(c_j), emb(s_i))
//
// where λ is selected on a held-out development split (-doc-split first50)
// and the chosen value is then evaluated on the test split
// (-doc-split last50) or on the full set (-doc-split all). λ=0 disables
// the lead-bias prior and reproduces position-agnostic mean-of-max recall.
//
// Per-document caching: SummEval has 16 candidates per article. We embed
// the source's sentences ONCE per DocumentID and reuse them across the
// 16 candidates that share the same source.
//
// Flags:
//
//	-lead-bias-lambda   λ in w(i)=exp(−λ·i/n); 0 disables, λ*=0.5 is canonical
//	-doc-split          first50 (dev) | last50 (test) | all (full SummEval)
//	-min-sent-len       drop sentences shorter than this many runes
//	-bootstrap          bootstrap resamples for 95% CI
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
	input          string
	output         string
	embedHost      string
	embedModel     string
	leadBiasLambda float64
	minSentLen     int
	docSplit       string
	n              int
	bootstrap      int
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "output/lgs.json", "write results to file")
	flag.StringVar(&embedHost, "embed-host", "http://localhost:11434", "Ollama host (sentence embeddings)")
	flag.StringVar(&embedModel, "embed-model", "nomic-embed-text", "Ollama embedding model")
	flag.Float64Var(&leadBiasLambda, "lead-bias-lambda", 0.5, "λ in source weight w(i)=exp(−λ·i/n); 0 disables the prior")
	flag.IntVar(&minSentLen, "min-sent-len", 4, "drop sentences shorter than this many runes")
	flag.StringVar(&docSplit, "doc-split", "all", "article-level split: all|first50|last50 (held-out hyperparameter selection)")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

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

	scorer := metrics.NewLGS(embedHost, embedModel)
	scorer.MinSentenceLen = minSentLen
	scorer.LeadBiasLambda = leadBiasLambda

	log.Printf("LGS: λ=%.3f lead-bias, split=%s, %d samples", leadBiasLambda, docSplit, len(samples))

	type cached struct {
		sents []string
		embs  [][]float64
	}
	cache := make(map[string]*cached)

	bar := progressbar.NewOptions(
		len(samples),
		progressbar.OptionSetDescription("lgs"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	scores := make([]float64, len(samples))
	entries := make([]eval.Score, len(samples))

	var sumScore float64

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

		score, _, err := scorer.ScoreWithSourceEmbeddings(ctx, c.sents, c.embs, s.Candidate)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}

		scores[i] = score
		entries[i] = eval.Score{SampleID: s.ID, Value: score}
		sumScore += score

		bar.Add(1)
	}

	elapsed := time.Since(start)
	N := float64(len(samples))
	log.Printf("LGS: %d samples in %.1fs — mean score=%.3f",
		len(samples), elapsed.Seconds(), sumScore/N)

	norm := fmt.Sprintf("lead_lambda=%.3f,split=%s,embed_model=%s", leadBiasLambda, docSplit, embedModel)
	report := eval.Report{
		Metric:     "lgs",
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

// filteredSplit splits a document into sentences and drops degenerate
// ones, matching the filter the metric applies internally.
func filteredSplit(text string, minLen int) []string {
	raw := metrics.SplitSentencesForLGS(text)
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
