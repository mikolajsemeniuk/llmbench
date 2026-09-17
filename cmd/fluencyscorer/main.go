// cmd/fluencyscorer scores every candidate's fluency from unconditioned
// GPT-2 perplexity (no source, no reference) via cmd/modelsrv's
// /fluency endpoint. One forward pass per sample -- much cheaper than
// LGS's or NLI's O(n*m) sentence-pair grid, since there is no source
// to compare against.
//
// Part of the go/no-go composite-metric probe (improvements.txt
// section 6): a signal that structurally cannot see the source cannot
// reward copying, which is the mechanism behind every cosine/lexical/
// positional metric's confound with extractiveness.
//
// Usage:
//
//	go run ./cmd/fluencyscorer -output output/fluency.json
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

func main() {
	var input, output, server string
	var n, bootstrap int
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "output/fluency.json", "write results to file")
	flag.StringVar(&server, "server", "http://localhost:9200", "model server URL, must have /fluency loaded")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

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

	scorer := metrics.NewFluency(server)

	bar := progressbar.NewOptions(
		len(samples),
		progressbar.OptionSetDescription("fluency"),
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
		score, err := scorer.Score(ctx, s.Candidate)
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
	log.Printf("Fluency (unconditioned perplexity): %d samples in %.1fs — mean score=%.3f",
		len(samples), elapsed.Seconds(), sumScore/N)

	report := eval.Report{
		Metric:     "fluencyppl",
		Norm:       "model=gpt2,unconditioned=true",
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
	fmt.Println("done")
}
