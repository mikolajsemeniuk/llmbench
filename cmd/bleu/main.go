package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/mikolajsemeniuk/llmbench"
)

func main() {
	input := flag.String("input", "", "path to samples JSON file")
	output := flag.String("output", "-", "path to write report JSON (- for stdout)")
	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "usage: bleu -input samples.json [-output report.json] [-model llama3.1]")
		os.Exit(1)
	}

	samples, err := llmbench.NewSamples(*input)
	if err != nil {
		log.Fatalf("load samples: %v", err)
	}

	results := make([]llmbench.Result, len(samples))
	for i, s := range samples {
		score := llmbench.BLEU(s.Reference, s.Candidate)
		results[i] = llmbench.Result{ID: s.ID, Score: score}
		fmt.Fprintf(os.Stderr, "[BLEU] %s: %.4f\n", s.ID, score)
	}

	report := llmbench.NewReport("BLEU-4", results)
	if err := report.WriteJSON(*output); err != nil {
		log.Fatalf("write report: %v", err)
	}
}
