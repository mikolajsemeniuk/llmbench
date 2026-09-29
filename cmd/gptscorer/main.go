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
	datasetName string
	output      string
	server      string
	n           int
	bootstrap   int
)

func main() {
	flag.StringVar(&datasetName, "dataset", dataset.Default, "embedded corpus in pkg/dataset: summeval|frank_cnndm|frank_xsum|rose_cnndm")
	flag.StringVar(&output, "output", "", "report path (default: output/<dataset>/gptscore.json)")
	flag.StringVar(&server, "server", "http://localhost:9200", "model server URL (port 9200)")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95% CI (0 = disabled)")
	flag.Parse()
	if output == "" {
		output = filepath.Join(eval.OutputDir(datasetName), "gptscore.json")
	}

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	samples, err := eval.LoadDataset(datasetName, n)
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

	scorer := metrics.NewGPTScorer(server)

	start := time.Now()
	scores := make([]float64, len(samples))
	entries := make([]eval.Score, len(samples))

	for i, s := range samples {
		v, err := scorer.Score(ctx, s.Document, s.Candidate)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}
		scores[i] = v
		entries[i] = eval.Score{SampleID: s.ID, Value: v}
		bar.Add(1)
	}
	elapsed := time.Since(start)

	report := eval.Report{
		Metric:     "gptscore",
		Norm:       "none",
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
