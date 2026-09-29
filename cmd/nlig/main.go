// cmd/nlig runs NLIG (NLI grounding, SummaC-ZS construction) on
// SummEval: every summary sentence against every 1- and 2-sentence
// source window with a DeBERTa-v3-large NLI cross-encoder served by
// cmd/modelsrv (/nli). One request per article.
//
// Outputs:
//
//	output/<dataset>/nlig.json          canonical: mean_j max_w (P(e) − P(c))
//	ablation/<dataset>/nlig_min.json    weakest sentence instead of the mean
//	ablation/<dataset>/nlig_contra.json −max contradiction
//	ablation/<dataset>/nlig_raw.json    per-sentence evidence for every candidate
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
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
	datasetName string
	host        string
	window      int
	n           int
	bootstrap   int
)

func main() {
	flag.StringVar(&datasetName, "dataset", dataset.Default, "embedded corpus in pkg/dataset: summeval|frank_cnndm|frank_xsum|rose_cnndm")
	flag.StringVar(&host, "host", "http://localhost:9200", "model server host")
	flag.IntVar(&window, "window", 2, "largest number of adjacent source sentences per premise")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	samples, err := eval.LoadDataset(datasetName, n)
	if err != nil {
		log.Fatal(err)
	}

	scorer := metrics.NewNLIG(host)
	scorer.Window = window

	var docs []string
	byDoc := map[string][]int{}
	for i, s := range samples {
		if _, ok := byDoc[s.DocumentID]; !ok {
			docs = append(docs, s.DocumentID)
		}
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], i)
	}
	log.Printf("NLIG: window=%d, %d articles, %d samples", window, len(docs), len(samples))

	bar := progressbar.NewOptions(len(samples),
		progressbar.OptionSetDescription("nlig"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	results := make([]metrics.NLIResult, len(samples))
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
			results[i] = res[j]
		}
		bar.Add(len(idx))
	}
	elapsed := time.Since(start)
	log.Printf("NLIG: %.1fs, %.1f ms/sample", elapsed.Seconds(), 1000*elapsed.Seconds()/float64(len(samples)))

	norm := fmt.Sprintf("window=%d", window)
	write := func(out, metric string, f func(metrics.NLIResult) float64) {
		scores := make([]float64, len(samples))
		entries := make([]eval.Score, len(samples))
		for i, r := range results {
			scores[i] = f(r)
			entries[i] = eval.Score{SampleID: samples[i].ID, Value: scores[i]}
		}
		report := eval.Report{
			Metric: metric, Norm: norm, Samples: len(samples),
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
		if err := eval.NewReport(out, report); err != nil {
			log.Fatal(err)
		}
	}

	write(filepath.Join(eval.OutputDir(datasetName), "nlig.json"), "nlig", metrics.NLIResult.Score)
	ablationDir := eval.AblationDir(datasetName)
	write(filepath.Join(ablationDir, "nlig_min.json"), "nlig_min", metrics.NLIResult.MinScore)
	write(filepath.Join(ablationDir, "nlig_contra.json"), "nlig_contra", metrics.NLIResult.MaxContradiction)

	type rawSample struct {
		SampleID string                `json:"sample_id"`
		Sents    []metrics.NLISentence `json:"sents"`
	}
	raw := make([]rawSample, len(samples))
	for i, r := range results {
		raw[i] = rawSample{SampleID: samples[i].ID, Sents: r.Sents}
	}
	b, err := json.Marshal(raw)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ablationDir, "nlig_raw.json"), b, 0o644); err != nil {
		log.Fatal(err)
	}
}
