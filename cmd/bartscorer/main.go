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
	model  string
)

func main() {
	flag.StringVar(&input, "input", "../../model_annotations.aligned.scored.jsonl", "path to SummEval dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "-", "path to write report JSON (- for stdout)")
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL")
	flag.StringVar(&model, "model", "qwen2.5:3b-instruct", "Ollama generative model for log-probability scoring")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	dataset, err := llmbench.NewDataset(input)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	scorer := llmbench.NewBARTScorer(host, model)

	var scores, relevance, coherence, fluency, consistency []float64

	total := 0
	for _, entry := range dataset {
		total += len(entry.MachineSummaries)
	}

	done := 0
	for _, entry := range dataset {
		for mi, machSumm := range entry.MachineSummaries {
			s, err := scorer.MaxScore(ctx, entry.HumanSummaries, machSumm)
			if err != nil {
				log.Fatalf("entry %s model %d: %v", entry.ID, mi, err)
			}
			scores = append(scores, s)
			relevance = append(relevance, entry.Relevance[mi])
			coherence = append(coherence, entry.Coherence[mi])
			fluency = append(fluency, entry.Fluency[mi])
			consistency = append(consistency, entry.Consistency[mi])

			done++
			fmt.Fprintf(os.Stderr, "\r[BARTScore] %d/%d", done, total)
		}
	}

	fmt.Fprintf(os.Stderr, "\nsamples: %d\n\n", len(scores))
	fmt.Fprintf(os.Stderr, "BARTScore summary-level correlations:\n")
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

	report := llmbench.NewReport("BARTScore", results)
	if err := report.WriteJSON(output); err != nil {
		log.Fatalf("write report: %v", err)
	}
}
