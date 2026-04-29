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

	llmbench "github.com/mikolajsemeniuk/llmbench/pkg"
	"github.com/schollz/progressbar/v3"
)

var validDimensions = map[string]bool{
	"coherence": true, "consistency": true, "fluency": true, "relevance": true,
}

var (
	input     string
	output    string
	server    string
	dimension string
	n         int
	bootstrap int
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "", "write results to file (default: output/unieval_<dim>.json)")
	flag.StringVar(&server, "server", "http://localhost:9200", "model server host")
	flag.StringVar(&dimension, "dimension", "coherence", "SummEval dimension: coherence|consistency|fluency|relevance")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	if !validDimensions[dimension] {
		log.Fatalf("unknown dimension %q (available: coherence, consistency, fluency, relevance)", dimension)
	}
	if output == "" {
		output = fmt.Sprintf("output/unieval_%s.json", dimension)
	}

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
		progressbar.OptionSetDescription(fmt.Sprintf("unieval-%s", dimension)),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	scorer := llmbench.NewUniEvalScorer(server, dimension)

	start := time.Now()
	scores := make([]float64, len(samples))
	entries := make([]llmbench.Score, len(samples))

	for i, s := range samples {
		// UniEval is reference-based for consistency/relevance, candidate-only
		// for coherence/fluency. The Python server handles dimension-specific
		// prompting; we always pass both ref and candidate.
		v, err := scorer.Score(ctx, s.References[0], s.Candidate)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}
		scores[i] = v
		entries[i] = llmbench.Score{SampleID: s.ID, Value: v}
		bar.Add(1)
	}
	elapsed := time.Since(start)

	report := llmbench.Report{
		Metric:     fmt.Sprintf("unieval_%s", dimension),
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
