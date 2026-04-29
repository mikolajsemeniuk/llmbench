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

	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
	"github.com/mikolajsemeniuk/llmbench/pkg/metrics"
	"github.com/schollz/progressbar/v3"
)

var (
	input     string
	output    string
	host      string
	model     string
	dimension string
	n         int
	bootstrap int
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "", "write results to file (default: output/geval_<dim>.json)")
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL (port 11434)")
	flag.StringVar(&model, "model", "qwen2.5:7b-instruct-q4_K_M", "judge model for G-Eval")
	flag.StringVar(&dimension, "dimension", "coherence", "SummEval dimension: coherence|consistency|fluency|relevance")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95% CI (0 = disabled)")
	flag.Parse()

	if _, ok := metrics.GEvalDimensions[dimension]; !ok {
		log.Fatalf("unknown dimension %q (available: coherence, consistency, fluency, relevance)", dimension)
	}
	if output == "" {
		output = fmt.Sprintf("output/geval_%s.json", dimension)
	}

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	fsys := os.DirFS(filepath.Dir(input))
	path := filepath.Base(input)
	if input == "" {
		fsys = eval.SummevalDataset
		path = eval.DefaultDatasetPath
	}
	samples, err := eval.NewDataset(fsys, path, n)
	if err != nil {
		log.Fatal(err)
	}

	bar := progressbar.NewOptions(
		len(samples),
		progressbar.OptionSetDescription(fmt.Sprintf("geval-%s", dimension)),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	geval := metrics.NewGEval(host, model)

	start := time.Now()
	scores := make([]float64, len(samples))
	entries := make([]eval.Score, len(samples))

	for i, s := range samples {
		v, err := geval.Score(ctx, dimension, s.Document, s.Candidate)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}
		scores[i] = v
		entries[i] = eval.Score{SampleID: s.ID, Value: v}
		bar.Add(1)
	}
	elapsed := time.Since(start)

	report := eval.Report{
		Metric:     fmt.Sprintf("geval_%s", dimension),
		Norm:       "none",
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
