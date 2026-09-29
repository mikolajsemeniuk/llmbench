// cmd/lgs runs LGS, our reference-free embedding-only summary-quality
// metric, on SummEval. The metric is
//
//	w(i)  = exp(−λ · i / n)
//	score = mean over c_j of  max_i  w(i) · cos(emb(c_j), emb(s_i))
//
// with λ = 0.5, the value selected on the SummEval development half in
// the earlier LGS study. λ=0 disables the lead-bias prior and reproduces
// position-agnostic mean-of-max recall.
//
// Per-document caching: SummEval has 16 candidates per article. We embed
// the source's sentences ONCE per DocumentID and reuse them across the
// 16 candidates that share the same source.
//
// Flags:
//
//	-lead-bias-lambda   λ in w(i)=exp(−λ·i/n); 0 disables, λ*=0.5 is canonical
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
	datasetName    string
	output         string
	embedHost      string
	embedModel     string
	leadBiasLambda float64
	minSentLen     int
	n              int
	bootstrap      int
)

func main() {
	flag.StringVar(&datasetName, "dataset", dataset.Default, "embedded corpus in pkg/dataset: summeval|frank_cnndm|frank_xsum|rose_cnndm")
	flag.StringVar(&output, "output", "", "report path (default: output/<dataset>/lgs.json)")
	flag.StringVar(&embedHost, "embed-host", "http://localhost:11434", "Ollama host (sentence embeddings)")
	flag.StringVar(&embedModel, "embed-model", "nomic-embed-text", "Ollama embedding model")
	flag.Float64Var(&leadBiasLambda, "lead-bias-lambda", 0.5, "λ in source weight w(i)=exp(−λ·i/n); 0 disables the prior")
	flag.IntVar(&minSentLen, "min-sent-len", 4, "drop sentences shorter than this many runes")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()
	if output == "" {
		output = filepath.Join(eval.OutputDir(datasetName), "lgs.json")
	}

	if leadBiasLambda < 0 {
		log.Fatalf("-lead-bias-lambda must be ≥ 0, got %v", leadBiasLambda)
	}

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	samples, err := eval.LoadDataset(datasetName, n)
	if err != nil {
		log.Fatal(err)
	}

	scorer := metrics.NewLGS(embedHost, embedModel)
	scorer.MinSentenceLen = minSentLen
	scorer.LeadBiasLambda = leadBiasLambda

	log.Printf("LGS: λ=%.3f lead-bias, %s, %d samples", leadBiasLambda, datasetName, len(samples))

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

		out, err := scorer.Score(ctx, metrics.LGSInput{
			SourceSents: c.sents,
			SourceEmbs:  c.embs,
			Candidate:   s.Candidate,
		})
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}

		scores[i] = out.Score
		entries[i] = eval.Score{SampleID: s.ID, Value: out.Score}
		sumScore += out.Score

		bar.Add(1)
	}

	elapsed := time.Since(start)
	N := float64(len(samples))
	log.Printf("LGS: %d samples in %.1fs — mean score=%.3f",
		len(samples), elapsed.Seconds(), sumScore/N)

	norm := fmt.Sprintf("lead_lambda=%.3f,embed_model=%s", leadBiasLambda, embedModel)
	report := eval.Report{
		Metric:     "lgs",
		Norm:       norm,
		Samples:    len(samples),
		RuntimeSec: elapsed.Seconds(),
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Scores:     entries,
		SummaryLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
			Bootstrap: bootstrap,
			Level:     "summary",
		}),
		SystemLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
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
	raw := metrics.SplitSentences(text)
	out := raw[:0]
	for _, s := range raw {
		if len([]rune(s)) >= minLen {
			out = append(out, s)
		}
	}
	return out
}
