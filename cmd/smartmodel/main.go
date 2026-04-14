package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"

	"github.com/mikolajsemeniuk/llmbench"
)

func main() {
	input := flag.String("input", "../../model_annotations.aligned.scored.jsonl", "path to SummEval dataset JSON/JSONL file")
	host := flag.String("host", "http://localhost:11434", "Ollama host URL")
	embed := flag.String("embed", "nomic-embed-text", "Ollama embedding model")
	n := flag.Int("n", 0, "entries limit (0=all)")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	var opts []llmbench.DatasetOption
	if *n > 0 {
		opts = append(opts, llmbench.WithDatasetSize(*n))
	}
	dataset, err := llmbench.NewDataset(*input, opts...)
	if err != nil {
		log.Fatal(err)
	}
	scorer := llmbench.NewSMARTModelScorer(ctx, *host, *embed)
	out, err := scorer.Score(dataset)
	if err != nil {
		log.Fatal(err)
	}
	llmbench.PrintResult("SMART-Model", out, llmbench.Correlation(out))
}
