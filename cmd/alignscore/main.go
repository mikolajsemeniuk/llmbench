// cmd/alignscore runs AlignScore-large (pkg/metrics/alignscore.go) on
// SummEval or, with -input, on a transfer corpus in the same layout. One
// request per article. Writes a standard report.
package main

import (
	"context"
	"flag"
	"io/fs"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
	"github.com/mikolajsemeniuk/llmbench/pkg/metrics"
	"github.com/schollz/progressbar/v3"
)

var (
	input     string
	output    string
	host      string
	bootstrap int
)

func main() {
	flag.StringVar(&input, "input", "", "path to a dataset JSONL in SummEval layout (default: embedded SummEval)")
	flag.StringVar(&output, "output", "output/alignscore.json", "report path")
	flag.StringVar(&host, "host", "http://localhost:9200", "model server host")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	fsys, path := fs.FS(dataset.Summeval), dataset.SummevalDefaultPath
	if input != "" {
		fsys, path = os.DirFS(filepath.Dir(input)), filepath.Base(input)
	}
	samples, err := eval.NewDataset(fsys, path, 0)
	if err != nil {
		log.Fatal(err)
	}

	var docs []string
	byDoc := map[string][]int{}
	for i, s := range samples {
		if _, ok := byDoc[s.DocumentID]; !ok {
			docs = append(docs, s.DocumentID)
		}
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], i)
	}

	scorer := metrics.NewAlignScore(host)
	bar := progressbar.NewOptions(len(samples),
		progressbar.OptionSetDescription("alignscore"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	scores := make([]float64, len(samples))
	start := time.Now()
	for _, d := range docs {
		idx := byDoc[d]
		src := samples[idx[0]].Document
		cands := make([]string, len(idx))
		for j, i := range idx {
			cands[j] = metrics.RenderCandidate(src, samples[i].Candidate)
		}
		res, err := scorer.ScoreArticle(ctx, src, cands)
		if err != nil {
			log.Fatalf("article %s: %v", d, err)
		}
		for j, i := range idx {
			scores[i] = res[j]
		}
		bar.Add(len(idx))
	}
	elapsed := time.Since(start)
	log.Printf("AlignScore: %.1fs, %.1f ms/sample", elapsed.Seconds(), 1000*elapsed.Seconds()/float64(len(samples)))

	entries := make([]eval.Score, len(samples))
	for i, s := range samples {
		entries[i] = eval.Score{SampleID: s.ID, Value: scores[i]}
	}
	report := eval.Report{
		Metric: "alignscore", Norm: "large,nli_sp", Samples: len(samples),
		RuntimeSec: elapsed.Seconds(),
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Scores:     entries,
		SummaryLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
			Bootstrap: bootstrap, Level: "summary",
		}),
		SystemLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
			Bootstrap: bootstrap, Level: "system",
		}),
	}
	if err := eval.NewReport(output, report); err != nil {
		log.Fatal(err)
	}
}
