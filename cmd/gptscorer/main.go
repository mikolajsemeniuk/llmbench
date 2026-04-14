package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/mikolajsemeniuk/llmbench"
)

var (
	input string
	host  string
)

func main() {
	flag.StringVar(&input, "input", "../../model_annotations.aligned.scored.jsonl", "path to samples JSON file")
	flag.StringVar(&host, "host", "http://localhost:9200", "model server host")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	samples, err := llmbench.NewSamples(input)
	if err != nil {
		log.Fatalf("load samples: %v", err)
	}

	scorer := llmbench.NewGPTScorer(host)
	n := len(samples)
	fmt.Fprintf(os.Stderr, "=== GPTScore (%d samples) ===\n\n", n)

	scores := make([]float64, n)
	var sum float64
	start := time.Now()

	for i, s := range samples {
		score, err := scorer.Score(ctx, s.Reference, s.Candidate)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}
		scores[i] = score
		sum += score
		fmt.Fprintf(os.Stderr, "  [%s] %.4f\n", s.ID, score)
	}

	total := time.Since(start)
	mean := sum / float64(n)
	avg := total / time.Duration(n)

	fmt.Fprintf(os.Stderr, "\nMEAN: %.4f  TOTAL: %s  AVG: %s\n", mean, total.Round(time.Millisecond), avg.Round(time.Millisecond))
}
