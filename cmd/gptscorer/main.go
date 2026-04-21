package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	llmbench "github.com/mikolajsemeniuk/llmbench/pkg"
	"github.com/schollz/progressbar/v3"
)

var (
	input     string
	output    string
	server    string
	n         int
	bootstrap int
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "output/gptscore.json", "write results to file instead of stdout")
	flag.StringVar(&server, "server", "http://localhost:9200", "model server host")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	fsys := os.DirFS(filepath.Dir(input))
	path := filepath.Base(input)
	if input == "" {
		fsys = llmbench.SummevalDataset
		path = llmbench.DefaultDatasetPath
	}
	samples, err := llmbench.NewDataset(fsys, path, n)
	if err != nil {
		log.Fatal(err)
	}

	bar := progressbar.NewOptions(
		len(samples),
		progressbar.OptionSetDescription("gptscore"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	scorer := llmbench.NewGPTScorer(server)

	start := time.Now()
	scores := make([]float64, len(samples))
	entries := make([]llmbench.Score, len(samples))

	for i, s := range samples {
		v, err := scorer.Score(ctx, s.Document, s.Candidate)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}
		scores[i] = v
		entries[i] = llmbench.Score{SampleID: s.ID, Value: v}
		bar.Add(1)
	}
	elapsed := time.Since(start)

	report := llmbench.Report{
		Metric:     "gptscore",
		Norm:       "none",
		Samples:    len(samples),
		RuntimeSec: elapsed.Seconds(),
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Scores:     entries,
		SummaryLevel: llmbench.NewCorrelationWith(samples, scores, llmbench.CorrelationOptions{
			Bootstrap: bootstrap,
			Level:     "summary",
		}),
		SystemLevel: llmbench.NewCorrelationWith(samples, scores, llmbench.CorrelationOptions{
			Bootstrap: bootstrap,
			Level:     "system",
		}),
	}
	if err := llmbench.NewReport(output, report); err != nil {
		log.Fatal(err)
	}
}
