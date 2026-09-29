// cmd/ccm runs CCM (Copy-invariant Contrastive Margin), our
// reference-free summary-quality metric, on SummEval:
//
//	score = log P(y | x) − mean_k log P(y'_k | x)
//
// where y'_1..y'_K are local, meaning-changing perturbations of the
// candidate y (entity swap, number change, negation flip, sentence
// swap) and P is a small causal LM served by cmd/modelsrv (/ccm). See
// pkg/metrics/ccm.go for the definition and its prior art.
//
// One request is sent per article carrying all 16 candidates and their
// perturbations, so the model server encodes each source once.
//
// Besides the canonical report (output/<dataset>/ccm.json), the command
// writes the ablation variants into ablation/<dataset>/, computed from
// the SAME forward passes (no extra model calls):
//
//	ccm_logp.json      log P(y|x)/|y|, no contrast (BARTScore s→h style)
//	ccm_zmargin.json   margin / std of perturbed log-likelihoods
//	ccm_pmimargin.json conditional margin minus unconditional margin
//
// and a per-sample dump (ccm_raw.json) with every log-likelihood and
// perturbation family, for the per-family and confound analyses.
//
// No hyperparameter is tuned: K=8 and the canonical variant (plain
// conditional margin) are fixed a priori.
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
	k           int
	seed        uint64
	n           int
	bootstrap   int
)

type rawSample struct {
	SampleID   string    `json:"sample_id"`
	Cond       float64   `json:"cond"`
	Uncond     float64   `json:"uncond"`
	Tokens     int       `json:"tokens"`
	PertCond   []float64 `json:"pert_cond"`
	PertUncond []float64 `json:"pert_uncond"`
	Families   []string  `json:"families"`
	Candidate  string    `json:"candidate"`
	Perturbed  []string  `json:"perturbed"`
}

func main() {
	flag.StringVar(&datasetName, "dataset", dataset.Default, "embedded corpus in pkg/dataset: summeval|frank_cnndm|frank_xsum|rose_cnndm")
	flag.StringVar(&host, "host", "http://localhost:9200", "model server host")
	flag.IntVar(&k, "k", 8, "perturbations per candidate")
	flag.Uint64Var(&seed, "seed", 42, "perturbation sampling seed")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	if k < 1 {
		log.Fatalf("-k must be ≥ 1, got %d", k)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	samples, err := eval.LoadDataset(datasetName, n)
	if err != nil {
		log.Fatal(err)
	}

	scorer := metrics.NewCCM(host)
	scorer.K = k
	scorer.Seed = seed

	log.Printf("CCM: K=%d, seed=%d, %s, %d samples", k, seed, datasetName, len(samples))

	// Group sample indices by article, preserving order.
	var docs []string
	byDoc := map[string][]int{}
	for i, s := range samples {
		if _, ok := byDoc[s.DocumentID]; !ok {
			docs = append(docs, s.DocumentID)
		}
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], i)
	}

	bar := progressbar.NewOptions(
		len(samples),
		progressbar.OptionSetDescription("ccm"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	results := make([]metrics.CCMResult, len(samples))
	raw := make([]rawSample, len(samples))
	short := 0

	start := time.Now()

	for _, d := range docs {
		idx := byDoc[d]
		cands := make([]metrics.CCMCandidate, len(idx))
		for j, i := range idx {
			cands[j] = scorer.Prepare(samples[i].ID, samples[i].Document, samples[i].Candidate)
			if len(cands[j].Perturbations) < k {
				short++
			}
		}
		res, err := scorer.ScoreArticle(ctx, samples[idx[0]].Document, cands)
		if err != nil {
			log.Fatalf("article %s: %v", d, err)
		}
		for j, i := range idx {
			results[i] = res[j]
			pt := make([]string, len(cands[j].Perturbations))
			for p, x := range cands[j].Perturbations {
				pt[p] = x.Text
			}
			raw[i] = rawSample{
				SampleID: samples[i].ID, Cond: res[j].Cond, Uncond: res[j].Uncond,
				Tokens: res[j].Tokens, PertCond: res[j].PertCond, PertUncond: res[j].PertUncond,
				Families: res[j].Families, Candidate: cands[j].Text, Perturbed: pt,
			}
		}
		bar.Add(len(idx))
	}

	elapsed := time.Since(start)
	log.Printf("CCM: %d samples in %.1fs (%.1f ms/sample); %d candidates got fewer than K=%d perturbations",
		len(samples), elapsed.Seconds(), 1000*elapsed.Seconds()/float64(len(samples)), short, k)

	norm := fmt.Sprintf("k=%d,seed=%d", k, seed)
	write := func(out, metric string, f func(metrics.CCMResult) float64) {
		scores := make([]float64, len(samples))
		entries := make([]eval.Score, len(samples))
		for i, r := range results {
			scores[i] = f(r)
			entries[i] = eval.Score{SampleID: samples[i].ID, Value: scores[i]}
		}
		report := eval.Report{
			Metric:     metric,
			Norm:       norm,
			Samples:    len(samples),
			RuntimeSec: elapsed.Seconds(),
			Timestamp:  time.Now().UTC().Format(time.RFC3339),
			Scores:     entries,
			SummaryLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
				Bootstrap: bootstrap,
				Level:     "summary",
			}),
			SystemLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
				Bootstrap: bootstrap,
				Level:     "system",
			}),
		}
		if err := eval.NewReport(out, report); err != nil {
			log.Fatal(err)
		}
	}

	ablationDir := eval.AblationDir(datasetName)
	write(filepath.Join(eval.OutputDir(datasetName), "ccm.json"), "ccm", metrics.CCMResult.Margin)
	write(filepath.Join(ablationDir, "ccm_logp.json"), "ccm_logp", metrics.CCMResult.LogPPerToken)
	write(filepath.Join(ablationDir, "ccm_zmargin.json"), "ccm_zmargin", metrics.CCMResult.ZMargin)
	write(filepath.Join(ablationDir, "ccm_pmimargin.json"), "ccm_pmimargin", metrics.CCMResult.PMIMargin)

	f, err := os.Create(filepath.Join(ablationDir, "ccm_raw.json"))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	if err := enc.Encode(raw); err != nil {
		log.Fatal(err)
	}
}
