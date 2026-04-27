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
	norm      string
	n         int
	bootstrap int
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "output/rouge.json", "write results to file instead of stdout")
	flag.StringVar(&norm, "norm", "max", "reference aggregation: max|mean")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()
	_ = ctx

	fn, ok := llmbench.Aggregators[norm]
	if !ok {
		log.Fatalf("unknown norm %q (available: max, mean)", norm)
	}

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
		progressbar.OptionSetDescription("rouge"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	start := time.Now()
	scores := make([]float64, len(samples))
	entries := make([]llmbench.Score, len(samples))
	references := make([]float64, 0, 16)

	for i, s := range samples {
		references = references[:0]
		for _, ref := range s.References {
			references = append(references, llmbench.ROUGEL(ref, s.Candidate))
		}
		scores[i] = fn(references)
		entries[i] = llmbench.Score{SampleID: s.ID, Value: scores[i]}
		bar.Add(1)
	}
	elapsed := time.Since(start)

	report := llmbench.Report{
		Metric:     "rouge",
		Norm:       norm,
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
