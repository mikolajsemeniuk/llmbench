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
// Besides the canonical report (-output), the command writes the
// ablation variants into -ablation-dir, computed from the SAME forward
// passes (no extra model calls):
//
//	ccm_logp.json      log P(y|x)/|y|, no contrast (BARTScore s→h style)
//	ccm_zmargin.json   margin / std of perturbed log-likelihoods
//	ccm_pmimargin.json conditional margin minus unconditional margin
//
// and a per-sample dump (ccm_raw.json) with every log-likelihood and
// perturbation family, for the per-family and confound analyses.
//
// It also writes CCM-D (-output-dim-dir): one dev-selected signal per
// SummEval dimension, output/ccmd_{coherence,consistency,fluency,
// relevance}.json. See pkg/metrics/ccm.go for the assignment.
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
	input        string
	output       string
	ablationDir  string
	outputDimDir string
	host         string
	k            int
	seed         uint64
	docSplit     string
	n            int
	bootstrap    int
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
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "output/ccm.json", "write canonical report to file")
	flag.StringVar(&ablationDir, "ablation-dir", "ablation", "directory for ablation variants and the raw dump (empty = skip)")
	flag.StringVar(&outputDimDir, "output-dim-dir", "output", "directory for the per-dimension CCM-D reports ccmd_<dim>.json (empty = skip)")
	flag.StringVar(&host, "host", "http://localhost:9200", "model server host")
	flag.IntVar(&k, "k", 8, "perturbations per candidate")
	flag.Uint64Var(&seed, "seed", 42, "perturbation sampling seed")
	flag.StringVar(&docSplit, "doc-split", "all", "article-level split: all|first50|last50")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	if k < 1 {
		log.Fatalf("-k must be ≥ 1, got %d", k)
	}
	switch docSplit {
	case "all", "first50", "last50":
	default:
		log.Fatalf("-doc-split must be all|first50|last50, got %q", docSplit)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	fsys := os.DirFS(filepath.Dir(input))
	path := filepath.Base(input)
	if input == "" {
		fsys = dataset.Summeval
		path = dataset.SummevalDefaultPath
	}

	samples, err := eval.NewDataset(fsys, path, n)
	if err != nil {
		log.Fatal(err)
	}
	samples = applyDocSplit(samples, docSplit)

	scorer := metrics.NewCCM(host)
	scorer.K = k
	scorer.Seed = seed

	log.Printf("CCM: K=%d, seed=%d, split=%s, %d samples", k, seed, docSplit, len(samples))

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

	norm := fmt.Sprintf("k=%d,seed=%d,split=%s", k, seed, docSplit)
	// share is the fraction of the run's wall-clock attributed to this
	// report. The four CCM-D reports come from ONE run, and the
	// cost-aware tools (cmd/confound, cmd/frontier) sum a dimensional
	// metric's reports, so each carries a quarter.
	write := func(out, metric string, f func(metrics.CCMResult) float64, share float64) {
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
			RuntimeSec: share * elapsed.Seconds(),
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

	write(output, "ccm", metrics.CCMResult.Margin, 1)

	// CCM-D: one dev-selected signal per SummEval dimension, from the
	// same forward passes. Laid out like UniEval/G-Eval
	// (<metric>_<dimension>.json) so cmd/compare and cmd/confound pick
	// the matching dimension.
	if outputDimDir != "" {
		for _, d := range []struct {
			dim string
			f   func(metrics.CCMResult) float64
		}{
			{"coherence", metrics.CCMResult.CoherenceScore},
			{"consistency", metrics.CCMResult.ConsistencyScore},
			{"fluency", metrics.CCMResult.FluencyScore},
			{"relevance", metrics.CCMResult.RelevanceScore},
		} {
			write(filepath.Join(outputDimDir, "ccmd_"+d.dim+suffixFor(docSplit)+".json"), "ccmd_"+d.dim, d.f, 0.25)
		}
	}

	if ablationDir == "" {
		return
	}
	suffix := suffixFor(docSplit)
	write(filepath.Join(ablationDir, "ccm_logp"+suffix+".json"), "ccm_logp", metrics.CCMResult.LogPPerToken, 1)
	write(filepath.Join(ablationDir, "ccm_zmargin"+suffix+".json"), "ccm_zmargin", metrics.CCMResult.ZMargin, 1)
	write(filepath.Join(ablationDir, "ccm_pmimargin"+suffix+".json"), "ccm_pmimargin", metrics.CCMResult.PMIMargin, 1)

	f, err := os.Create(filepath.Join(ablationDir, "ccm_raw"+suffix+".json"))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	if err := enc.Encode(raw); err != nil {
		log.Fatal(err)
	}
}

func suffixFor(split string) string {
	if split == "all" {
		return ""
	}
	return "_" + split
}

// applyDocSplit slices the loaded sample list into a document-level
// development or test half, by dataset order (same as cmd/lgs).
func applyDocSplit(samples []eval.Sample, split string) []eval.Sample {
	if split == "all" {
		return samples
	}
	var docs []string
	seen := map[string]bool{}
	for _, s := range samples {
		if !seen[s.DocumentID] {
			seen[s.DocumentID] = true
			docs = append(docs, s.DocumentID)
		}
	}
	if len(docs) < 100 {
		log.Fatalf("doc-split %q expects ≥100 documents, got %d", split, len(docs))
	}
	keep := map[string]bool{}
	switch split {
	case "first50":
		for _, d := range docs[:50] {
			keep[d] = true
		}
	case "last50":
		for _, d := range docs[len(docs)-50:] {
			keep[d] = true
		}
	}
	out := samples[:0]
	for _, s := range samples {
		if keep[s.DocumentID] {
			out = append(out, s)
		}
	}
	return out
}
