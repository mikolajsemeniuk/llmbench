// cmd/psc runs PSC (Paraphrased-Source Conditioning) on SummEval: every
// CCM signal is recomputed with the source replaced by a meaning-
// preserving paraphrase x̃, written once per article by an Ollama
// instruction model. Perturbations, seed and LM are identical to
// cmd/ccm, so PSC vs CCM differs ONLY in the conditioning text.
//
// Paraphrases are cached in -paraphrases (JSON, keyed by document ID,
// with the generation wall-clock per article) so a rerun scores the
// exact same x̃; the cached generation time is still charged to the
// reported runtime.
//
// Outputs:
//
//	output/pscd_<dim>.json   CCM-D recipe (pkg/metrics/ccm.go) under x̃ —
//	                         the pre-registered primary metric; each
//	                         report carries a quarter of the wall-clock
//	ablation/psc_logp.json    log P(y|x̃)/|y|
//	ablation/psc_margin.json  log P(y|x̃) − mean_k log P(y'_k|x̃)
//	ablation/psc_raw.json     per-sample dump (same layout as ccm_raw.json)
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
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
	input        string
	outputDimDir string
	ablationDir  string
	paraFile     string
	host         string
	ollamaHost   string
	paraModel    string
	k            int
	seed         uint64
	n            int
	bootstrap    int
)

type paraEntry struct {
	Text    string  `json:"text"`
	Seconds float64 `json:"seconds"`
	Overlap float64 `json:"bigram_overlap_with_source"`
}

type paraCache struct {
	Model string               `json:"model"`
	Docs  map[string]paraEntry `json:"docs"`
}

type rawSample struct {
	SampleID   string    `json:"sample_id"`
	Cond       float64   `json:"cond"`
	Uncond     float64   `json:"uncond"`
	Tokens     int       `json:"tokens"`
	PertCond   []float64 `json:"pert_cond"`
	PertUncond []float64 `json:"pert_uncond"`
	Families   []string  `json:"families"`
}

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&outputDimDir, "output-dim-dir", "output", "directory for pscd_<dim>.json")
	flag.StringVar(&ablationDir, "ablation-dir", "ablation", "directory for ablation variants and the raw dump (empty = skip)")
	flag.StringVar(&paraFile, "paraphrases", "ablation/psc_paraphrases.json", "paraphrase cache")
	flag.StringVar(&host, "host", "http://localhost:9200", "model server host (scoring LM)")
	flag.StringVar(&ollamaHost, "ollama-host", "http://localhost:11434", "Ollama host (paraphraser)")
	flag.StringVar(&paraModel, "paraphrase-model", "qwen2.5:7b-instruct-q4_K_M", "Ollama model that writes x̃")
	flag.IntVar(&k, "k", 8, "perturbations per candidate (same as cmd/ccm)")
	flag.Uint64Var(&seed, "seed", 42, "perturbation sampling seed (same as cmd/ccm)")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

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

	cache, err := loadCache(paraFile)
	if err != nil {
		log.Fatal(err)
	}
	if cache.Model != "" && cache.Model != paraModel {
		log.Fatalf("paraphrase cache %s was written by %s, not %s; delete it or pass the matching -paraphrase-model",
			paraFile, cache.Model, paraModel)
	}
	cache.Model = paraModel

	para := metrics.NewParaphraser(ollamaHost, paraModel)
	scorer := metrics.NewCCM(host)
	scorer.K = k
	scorer.Seed = seed

	var docs []string
	byDoc := map[string][]int{}
	for i, s := range samples {
		if _, ok := byDoc[s.DocumentID]; !ok {
			docs = append(docs, s.DocumentID)
		}
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], i)
	}

	log.Printf("PSC: paraphraser=%s, K=%d, seed=%d, %d articles, %d samples (%d paraphrases cached)",
		paraModel, k, seed, len(docs), len(samples), len(cache.Docs))

	bar := progressbar.NewOptions(len(samples),
		progressbar.OptionSetDescription("psc"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	results := make([]metrics.CCMResult, len(samples))
	var scoreTime, paraTime time.Duration
	var overlap float64

	for _, d := range docs {
		idx := byDoc[d]
		src := samples[idx[0]].Document

		pe, ok := cache.Docs[d]
		if !ok {
			t := time.Now()
			txt, err := para.Paraphrase(ctx, src)
			if err != nil {
				log.Fatalf("article %s: %v", d, err)
			}
			pe = paraEntry{Text: txt, Seconds: time.Since(t).Seconds(), Overlap: metrics.BigramOverlap(src, txt)}
			cache.Docs[d] = pe
			if err := saveCache(paraFile, cache); err != nil {
				log.Fatal(err)
			}
		}
		paraTime += time.Duration(pe.Seconds * float64(time.Second))
		overlap += pe.Overlap

		t := time.Now()
		cands := make([]metrics.CCMCandidate, len(idx))
		for j, i := range idx {
			cands[j] = scorer.Prepare(samples[i].ID, src, samples[i].Candidate)
		}
		res, err := scorer.ScoreArticle(ctx, pe.Text, cands)
		if err != nil {
			log.Fatalf("article %s: %v", d, err)
		}
		scoreTime += time.Since(t)
		for j, i := range idx {
			results[i] = res[j]
		}
		bar.Add(len(idx))
	}

	elapsed := paraTime + scoreTime
	N := float64(len(samples))
	log.Printf("PSC: paraphrase %.1fs + scoring %.1fs = %.1f ms/sample; mean bigram overlap x̃ vs x = %.3f",
		paraTime.Seconds(), scoreTime.Seconds(), 1000*elapsed.Seconds()/N, overlap/float64(len(docs)))

	norm := fmt.Sprintf("paraphraser=%s,k=%d,seed=%d", paraModel, k, seed)
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

	// The four pscd reports come from ONE run; cost-aware tools sum a
	// dimensional metric's reports, so each carries a quarter.
	for _, d := range []struct {
		dim string
		f   func(metrics.CCMResult) float64
	}{
		{"coherence", metrics.CCMResult.CoherenceScore},
		{"consistency", metrics.CCMResult.ConsistencyScore},
		{"fluency", metrics.CCMResult.FluencyScore},
		{"relevance", metrics.CCMResult.RelevanceScore},
	} {
		write(filepath.Join(outputDimDir, "pscd_"+d.dim+".json"), "pscd_"+d.dim, d.f, 0.25)
	}

	if ablationDir == "" {
		return
	}
	write(filepath.Join(ablationDir, "psc_logp.json"), "psc_logp", metrics.CCMResult.LogPPerToken, 1)
	write(filepath.Join(ablationDir, "psc_margin.json"), "psc_margin", metrics.CCMResult.Margin, 1)

	raw := make([]rawSample, len(samples))
	for i, r := range results {
		raw[i] = rawSample{
			SampleID: samples[i].ID, Cond: r.Cond, Uncond: r.Uncond, Tokens: r.Tokens,
			PertCond: r.PertCond, PertUncond: r.PertUncond, Families: r.Families,
		}
	}
	b, err := json.Marshal(raw)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ablationDir, "psc_raw.json"), b, 0o644); err != nil {
		log.Fatal(err)
	}
}

func loadCache(path string) (*paraCache, error) {
	c := &paraCache{Docs: map[string]paraEntry{}}
	b, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, c); err != nil {
		return nil, fmt.Errorf("paraphrase cache %s: %w", path, err)
	}
	if c.Docs == nil {
		c.Docs = map[string]paraEntry{}
	}
	return c, nil
}

func saveCache(path string, c *paraCache) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
