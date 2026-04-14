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
		for i, v := range entry.MachineSummaries {
			m := llmbench.MaxBLEU(entry.HumanSummaries, v)
			scores = append(scores, m)
			relevance = append(relevance, entry.Relevance[i])
			coherence = append(coherence, entry.Coherence[i])
			fluency = append(fluency, entry.Fluency[i])
			consistency = append(consistency, entry.Consistency[i])
		}
	}

	fmt.Fprintf(os.Stderr, "samples: %d\n\n", len(scores))
	fmt.Fprintf(os.Stderr, "BLEU-4 summary-level correlations:\n")
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

	for _, d := range dims {
		sp := llmbench.SpearmanCorrelation(scores, d.vals)
		pe := llmbench.PearsonCorrelation(scores, d.vals)
		fmt.Fprintf(os.Stderr, "%-15s %10.4f %10.4f\n", d.name, sp, pe)
	}

	fmt.Println()
}
