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
	input       string
	output      string
	host        string
	model       string
	dimension   string
	n           int
	bootstrap   int
	runs        int
	temperature float64
	baseSeed    int64
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "", "write results to file (default: output/geval_<dim>.json)")
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL")
	flag.StringVar(&model, "model", "qwen2.5:7b-instruct-q4_K_M", "judge model for G-Eval")
	flag.StringVar(&dimension, "dimension", "coherence", "SummEval dimension: coherence|consistency|fluency|relevance")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.IntVar(&runs, "runs", 1, "independent G-Eval runs to average over (>1 enables multi-run with mean±std variance reporting)")
	flag.Float64Var(&temperature, "temperature", 0, "LLM-judge temperature (0 = greedy; >0 needed for non-trivial multi-run variance)")
	flag.Int64Var(&baseSeed, "base-seed", 42, "base seed; run k uses seed = base-seed + k")
	flag.Parse()

	if runs < 1 {
		log.Fatalf("-runs must be ≥ 1 (got %d)", runs)
	}

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
		fsys = dataset.Summeval
		path = dataset.SummevalDefaultPath
	}

	samples, err := eval.NewDataset(fsys, path, n)
	if err != nil {
		log.Fatal(err)
	}

	bar := progressbar.NewOptions(
		len(samples)*runs,
		progressbar.OptionSetDescription(fmt.Sprintf("geval-%s", dimension)),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	geval := metrics.NewGEval(host, model)
	geval.Temperature = temperature

	start := time.Now()
	perRunScores := make([][]float64, runs)
	for k := range perRunScores {
		perRunScores[k] = make([]float64, len(samples))
	}
	fallbackCount := 0
	var fallbackExamples []string

	for k := 0; k < runs; k++ {
		geval.Seed = baseSeed + int64(k)
		for i, s := range samples {
			res, err := geval.ScoreDetailed(ctx, dimension, s.Document, s.Candidate)
			if err != nil {
				log.Printf("warning: run %d sample %s: %v — using neutral fallback", k, s.ID, err)
				res.NormalizedScore = 0.5
				res.UsedFallback = true
			}
			perRunScores[k][i] = res.NormalizedScore
			if res.UsedFallback {
				fallbackCount++
				if len(fallbackExamples) < 5 {
					fallbackExamples = append(fallbackExamples,
						fmt.Sprintf("run=%d %s: %q", k, s.ID, res.RawResponse))
				}
			}
			bar.Add(1)
		}
	}
	elapsed := time.Since(start)

	// Canonical scores = per-sample mean across runs. Matches the
	// original G-Eval methodology of averaging N stochastic samples.
	scores := make([]float64, len(samples))
	entries := make([]eval.Score, len(samples))
	for i, s := range samples {
		var sum float64
		for k := 0; k < runs; k++ {
			sum += perRunScores[k][i]
		}
		scores[i] = sum / float64(runs)
		entries[i] = eval.Score{SampleID: s.ID, Value: scores[i]}
	}

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

	norm := fmt.Sprintf("runs=%d,temperature=%g,base_seed=%d", runs, temperature, baseSeed)

	report := eval.Report{
		Metric:     fmt.Sprintf("geval_%s", dimension),
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

	if runs > 1 {
		summaryCorrs := make([]eval.Correlation, runs)
		systemCorrs := make([]eval.Correlation, runs)
		for k := 0; k < runs; k++ {
			summaryCorrs[k] = eval.NewCorrelationWith(samples, perRunScores[k], eval.CorrelationOptions{Level: "summary"})
			systemCorrs[k] = eval.NewCorrelationWith(samples, perRunScores[k], eval.CorrelationOptions{Level: "system"})
		}
		report.Runs = &eval.RunsAggregate{
			NRuns:       runs,
			Temperature: temperature,
			Summary:     eval.AggregateRuns(summaryCorrs),
			System:      eval.AggregateRuns(systemCorrs),
		}
	}
	if err := eval.NewReport(output, report); err != nil {
		log.Fatal(err)
	}
}
