package main

import (
	"flag"
	"log"

	"github.com/mikolajsemeniuk/llmbench"
)

func main() {
	input := flag.String("input", "../../model_annotations.aligned.scored.jsonl", "path to SummEval dataset JSON/JSONL file")
	n := flag.Int("n", 0, "entries limit (0=all)")
	flag.Parse()

	var opts []llmbench.DatasetOption
	if *n > 0 {
		opts = append(opts, llmbench.WithDatasetSize(*n))
	}
	dataset, err := llmbench.NewDataset(*input, opts...)
	if err != nil {
		log.Fatal(err)
	}
	out, err := llmbench.NewChrFScorer().Score(dataset)
	if err != nil {
		log.Fatal(err)
	}
	llmbench.PrintResult("ChrF", out, llmbench.Correlation(out))
}
