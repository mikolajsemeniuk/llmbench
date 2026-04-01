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
	judge  string
)

func main() {
	flag.StringVar(&input, "input", "", "path to samples JSON file")
	flag.StringVar(&output, "output", "-", "path to write report JSON (- for stdout)")
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL")
	flag.StringVar(&judge, "judge", "qwen2.5:7b", "Ollama model used as evaluator")
	flag.Parse()

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	samples, err := llmbench.NewSamples(input)
	if err != nil {
		log.Fatalf("load samples: %v", err)
	}

	eval := llmbench.NewGEval(host, judge)

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
	if err := report.WriteJSON(output); err != nil {
		log.Fatalf("write report: %v", err)
	}
}
