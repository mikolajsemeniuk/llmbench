package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"path/filepath"

	"github.com/schollz/progressbar/v3"

	llmbench "github.com/mikolajsemeniuk/llmbench/pkg"
)

var (
	input  string
	output string
	n      int
	norm   string
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "", "write results to file instead of stdout")
	flag.StringVar(&norm, "norm", "max", "reference aggregation: max|mean")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	fsys := os.DirFS(filepath.Dir(input))
	path := filepath.Base(input)
	if input == "" {
		fsys = llmbench.SummevalDataset
		path = llmbench.DefaultDatasetPath
	}

	samples, err := llmbench.NewDataset(fsys, path, n)
	if err != nil {
		log.Fatal(err)
	}

	bar := progressbar.NewOptions(
		len(samples),
		progressbar.OptionSetDescription("bleu"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)
	scores := make([]float64, len(samples))
	for i, s := range samples {
		best := 0.0
		for _, ref := range s.References {
			if v := llmbench.BLEU(ref, s.Candidate); v > best {
				best = v
			}
		}

		scores[i] = best
		bar.Add(1)
	}
}
