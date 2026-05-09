package main

import (
	"context"
	"flag"
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
	input     string
	output    string
	host      string
	model     string
	n         int
	norm      string
	bootstrap int
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "output/smartmodel.json", "write results to file instead of stdout")
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL (port 11434)")
	flag.StringVar(&model, "model", "nomic-embed-text", "embedding model for SMART-Model")
	flag.StringVar(&norm, "norm", eval.DefaultNormName, "reference aggregation: max|mean")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95% CI (0 = disabled)")
	flag.Parse()

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	fn, ok := eval.Norms[norm]
	if !ok {
		fn = eval.DefaultNorm
	}

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

	bar := progressbar.NewOptions(
		len(samples),
		progressbar.OptionSetDescription("smartmodel"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	scorer := metrics.NewSMARTModelScorer(host, model)

	start := time.Now()
	scores := make([]float64, len(samples))
	entries := make([]eval.Score, len(samples))
	references := make([]float64, 0, 16)

	for i, s := range samples {
		references = references[:0]
		for _, ref := range s.References {
			v, err := scorer.Score(ctx, ref, s.Candidate)
			if err != nil {
				log.Fatalf("sample %s: %v", s.ID, err)
			}
			references = append(references, v)
		}
		scores[i] = fn(references)
		entries[i] = eval.Score{SampleID: s.ID, Value: scores[i]}
		bar.Add(1)
	}
	elapsed := time.Since(start)

	report := eval.Report{
		Metric:     "smartmodel",
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
