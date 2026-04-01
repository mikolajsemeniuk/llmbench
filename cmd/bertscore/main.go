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

func main() {
	input := flag.String("input", "", "path to samples JSON file")
	output := flag.String("output", "-", "path to write report JSON (- for stdout)")
	host := flag.String("host", "http://localhost:11434", "Ollama host URL")
	embed := flag.String("embed", "nomic-embed-text", "Ollama embedding model")
	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "usage: bertscore -input samples.json [-embed nomic-embed-text] [-host http://localhost:11434]")
		os.Exit(1)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	samples, err := llmbench.NewSamples(*input)
	if err != nil {
		log.Fatalf("load samples: %v", err)
	}

	scorer := llmbench.NewBERTScorer(*host, *embed)
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
	if err := report.WriteJSON(*output); err != nil {
		log.Fatalf("write report: %v", err)
	}
}
