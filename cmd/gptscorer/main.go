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
	host := flag.String("host", "http://localhost:9200", "model server host")
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

	scorer := llmbench.NewGPTScorer(ctx, *host)
	out, err := scorer.Score(dataset)
	if err != nil {
		log.Fatal(err)
	}

	llmbench.PrintResult("GPTScore", out, llmbench.Correlation(out))
}
