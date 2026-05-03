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
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL")
	flag.StringVar(&model, "model", "qwen2.5:7b-instruct-q4_K_M", "judge model for G-Eval")
	flag.StringVar(&dimension, "dimension", "coherence", "SummEval dimension: coherence|consistency|fluency|relevance")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
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
	fallbackCount := 0
	var fallbackExamples []string

	for i, s := range samples {
		res, err := geval.ScoreDetailed(ctx, dimension, s.Document, s.Candidate)
		if err != nil {
			// Network / Ollama errors: log, neutral fallback, continue.
			log.Printf("warning: sample %s: %v — using neutral fallback", s.ID, err)
			res.NormalizedScore = 0.5
			res.UsedFallback = true
		}
		scores[i] = res.NormalizedScore
		entries[i] = eval.Score{SampleID: s.ID, Value: res.NormalizedScore}
		if res.UsedFallback {
			fallbackCount++
			if len(fallbackExamples) < 5 {
				fallbackExamples = append(fallbackExamples,
					fmt.Sprintf("%s: %q", s.ID, res.RawResponse))
			}
		}
		bar.Add(1)
	}
	elapsed := time.Since(start)

	fallbackPct := 100.0 * float64(fallbackCount) / float64(len(samples))
	fmt.Fprintf(os.Stderr, "\nfallback rate: %d/%d (%.1f%%)\n",
		fallbackCount, len(samples), fallbackPct)
	if fallbackPct > 5.0 {
		fmt.Fprintln(os.Stderr, "WARNING: high fallback rate — model often fails to follow format")
		fmt.Fprintln(os.Stderr, "first few unparseable responses:")
		for _, ex := range fallbackExamples {
			fmt.Fprintf(os.Stderr, "  %s\n", ex)
		}
	}

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
