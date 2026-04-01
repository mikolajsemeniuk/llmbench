package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/mikolajsemeniuk/llmbench"
)

var (
	input  string
	output string
	host   string
	embed  string
)

func main() {
	flag.StringVar(&input, "input", "", "path to samples JSON file")
	flag.StringVar(&output, "output", "", "path to write report JSON (- for stdout)")
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL")
	flag.StringVar(&embed, "embed", "nomic-embed-text", "Ollama embedding model")
	flag.Parse()

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	samples, err := llmbench.NewSamples(input)
	if err != nil {
		log.Fatalf("load samples: %v", err)
	}

	scorer := llmbench.NewBERTScorer(host, embed)
	results := make([]llmbench.Result, len(samples))
	for i, s := range samples {
		score, err := scorer.Score(ctx, s.Reference, s.Candidate)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}

		results[i] = llmbench.Result{ID: s.ID, Score: score}
		fmt.Fprintf(os.Stderr, "[BERTScore] %s: %.4f\n", s.ID, score)
	}

	report := llmbench.NewReport("BERTScore", results)
	if err := report.WriteJSON(output); err != nil {
		log.Fatalf("write report: %v", err)
	}
}
