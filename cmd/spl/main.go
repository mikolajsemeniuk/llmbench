// cmd/spl runs SPL (Splice-Point Likelihood, pkg/metrics/spl.go) on
// SummEval: the mean log-probability, under the CCM causal LM with the
// source in context, of the words where the candidate departs from a
// verbatim source fragment. One /tokenlogprobs request per article
// (one source pass, 16 candidates, no perturbations).
//
// Outputs:
//
//	output/spl.json          canonical: mean log P(splice word | x, prefix)
//	ablation/spl_pmi.json    mean [log P(w|x,·) − log P(w|·)] at splices
//	ablation/spl_min.json    worst splice
//	ablation/spl_inside.json mean over words INSIDE copied fragments (control)
//	ablation/spl_raw.json    per-word evidence for every candidate
package main

import (
	"context"
	"encoding/json"
	"flag"
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
	output      string
	ablationDir string
	host        string
	n           int
	bootstrap   int
)

func main() {
	flag.StringVar(&output, "output", "output/spl.json", "canonical report")
	flag.StringVar(&ablationDir, "ablation-dir", "ablation", "directory for ablation variants and the raw dump (empty = skip)")
	flag.StringVar(&host, "host", "http://localhost:9200", "model server host")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	samples, err := eval.NewDataset(dataset.Summeval, dataset.SummevalDefaultPath, n)
	if err != nil {
		log.Fatal(err)
	}
	scorer := metrics.NewSPL(host)

	var docs []string
	byDoc := map[string][]int{}
	for i, s := range samples {
		if _, ok := byDoc[s.DocumentID]; !ok {
			docs = append(docs, s.DocumentID)
		}
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], i)
	}
	log.Printf("SPL: %d articles, %d samples", len(docs), len(samples))

	bar := progressbar.NewOptions(len(samples),
		progressbar.OptionSetDescription("spl"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	results := make([]metrics.SPLResult, len(samples))
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
	log.Printf("SPL: %.1fs, %.1f ms/sample", elapsed.Seconds(), 1000*elapsed.Seconds()/float64(len(samples)))

	write := func(out, metric string, f func(metrics.SPLResult) float64) {
		scores := make([]float64, len(samples))
		entries := make([]eval.Score, len(samples))
		for i, r := range results {
			scores[i] = f(r)
			entries[i] = eval.Score{SampleID: samples[i].ID, Value: scores[i]}
		}
		report := eval.Report{
			Metric: metric, Norm: "lm=ccm", Samples: len(samples),
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

	write(output, "spl", metrics.SPLResult.Score)
	if ablationDir == "" {
		return
	}
	write(filepath.Join(ablationDir, "spl_pmi.json"), "spl_pmi", metrics.SPLResult.PMIScore)
	write(filepath.Join(ablationDir, "spl_min.json"), "spl_min", metrics.SPLResult.MinScore)
	write(filepath.Join(ablationDir, "spl_inside.json"), "spl_inside", metrics.SPLResult.InsideScore)

	type rawSample struct {
		SampleID string            `json:"sample_id"`
		Words    []metrics.SPLWord `json:"words"`
	}
	raw := make([]rawSample, len(samples))
	for i, r := range results {
		raw[i] = rawSample{SampleID: samples[i].ID, Words: r.Words}
	}
	b, err := json.Marshal(raw)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ablationDir, "spl_raw.json"), b, 0o644); err != nil {
		log.Fatal(err)
	}
}
