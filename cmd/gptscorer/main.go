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
	input string
	host  string
)

func main() {
	flag.StringVar(&input, "input", "../../model_annotations.aligned.scored.jsonl", "path to SummEval dataset JSON/JSONL file")
	flag.StringVar(&host, "host", "http://localhost:9200", "model server host")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	dataset, err := llmbench.NewDataset(input)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	scorer := llmbench.NewGPTScorer(host)

	var scores, relevance, coherence, fluency, consistency []float64

	total := 0
	for _, entry := range dataset {
		total += len(entry.MachineSummaries)
	}

	done := 0
	for _, entry := range dataset {
		for mi, machSumm := range entry.MachineSummaries {
			best := 0.0
			first := true
			for _, humanSumm := range entry.HumanSummaries {
				s, err := scorer.Score(ctx, humanSumm, machSumm)
				if err != nil {
					log.Fatalf("entry %s model %d: %v", entry.ID, mi, err)
				}
				if first || s > best {
					best = s
					first = false
				}
			}
			scores = append(scores, best)
			relevance = append(relevance, entry.Relevance[mi])
			coherence = append(coherence, entry.Coherence[mi])
			fluency = append(fluency, entry.Fluency[mi])
			consistency = append(consistency, entry.Consistency[mi])

			done++
			fmt.Fprintf(os.Stderr, "\r[GPTScore] %d/%d", done, total)
		}
	}

	fmt.Fprintf(os.Stderr, "\nsamples: %d\n\n", len(scores))
	fmt.Fprintf(os.Stderr, "GPTScore summary-level correlations:\n")
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
}
