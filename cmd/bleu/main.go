package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/mikolajsemeniuk/llmbench"
)

var (
	input  string
	output string
)

func main() {
	flag.StringVar(&input, "input", "", "path to samples JSON file")
	flag.StringVar(&output, "output", "-", "path to write report JSON (- for stdout)")
	flag.Parse()

	samples, err := llmbench.NewSamples(input)
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
	if err := report.WriteJSON(output); err != nil {
		log.Fatalf("write report: %v", err)
	}
}
