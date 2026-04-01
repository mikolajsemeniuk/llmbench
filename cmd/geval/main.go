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
	judge := flag.String("judge", "qwen2.5:7b", "Ollama model used as evaluator")
	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "usage: geval -input samples.json [-judge qwen2.5:7b] [-host http://localhost:11434]")
		os.Exit(1)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	samples, err := llmbench.NewSamples(*input)
	if err != nil {
		log.Fatalf("load samples: %v", err)
	}

	eval := llmbench.NewGEval(*host, *judge)

	results := make([]llmbench.Result, len(samples))
	for i, s := range samples {
		score, err := eval.Score(ctx, s.Question, s.Reference, s.Candidate)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}
		results[i] = llmbench.Result{ID: s.ID, Score: score}
		fmt.Fprintf(os.Stderr, "[G-Eval] %s: %.2f (raw: %.1f/10)\n", s.ID, score, score*10)
	}

	report := llmbench.NewReport("G-Eval", results)
	if err := report.WriteJSON(*output); err != nil {
		log.Fatalf("write report: %v", err)
	}
}
