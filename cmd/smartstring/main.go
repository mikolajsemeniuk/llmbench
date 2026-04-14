package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/mikolajsemeniuk/llmbench"
)

var input string

func main() {
	flag.StringVar(&input, "input", "../../model_annotations.aligned.scored.jsonl", "path to SummEval dataset JSON/JSONL file")
	flag.Parse()

	dataset, err := llmbench.NewDataset(input)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	var scores, relevance, coherence, fluency, consistency []float64

	for _, entry := range dataset {
		for mi, machSumm := range entry.MachineSummaries {
			s := llmbench.MaxSMARTString(entry.HumanSummaries, machSumm)
			scores = append(scores, s)
			relevance = append(relevance, entry.Relevance[mi])
			coherence = append(coherence, entry.Coherence[mi])
			fluency = append(fluency, entry.Fluency[mi])
			consistency = append(consistency, entry.Consistency[mi])
		}
	}

	fmt.Fprintf(os.Stderr, "samples: %d\n\n", len(scores))
	fmt.Fprintf(os.Stderr, "SMART-String summary-level correlations:\n")
	fmt.Fprintf(os.Stderr, "%-15s %10s %10s\n", "dimension", "spearman", "pearson")
	fmt.Fprintf(os.Stderr, "%-15s %10s %10s\n", "---------", "--------", "-------")

	dims := []struct {
		name string
		vals []float64
	}{
		{"coherence", coherence},
		{"consistency", consistency},
		{"fluency", fluency},
		{"relevance", relevance},
	}

	results := make([]llmbench.Result, len(dims))
	for i, d := range dims {
		sp := llmbench.SpearmanCorrelation(scores, d.vals)
		pe := llmbench.PearsonCorrelation(scores, d.vals)
		fmt.Fprintf(os.Stderr, "%-15s %10.4f %10.4f\n", d.name, sp, pe)
		results[i] = llmbench.Result{ID: d.name, Score: sp}
	}
}
