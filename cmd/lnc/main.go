// cmd/lnc runs LNC (Likelihood–NLI Composite, pkg/metrics/lnc.go) on
// SummEval end to end: for every article it computes the CCM signals
// (source-conditioned likelihood + K perturbations), the NLIG signals
// (sentence-level NLI grounding) and the SPL signals (splice-point
// likelihood), then applies a per-dimension ridge calibration.
//
// With -fit (default) the calibration is fitted on the development half
// ONLY (first 50 articles in dataset order), written to -calibration,
// and applied to all 1600 samples; the last 50 articles are the held-out
// test. With -fit=false an existing calibration is loaded and applied
// unchanged — how LNC is meant to be used on new data.
//
// Outputs:
//
//	output/lnc_<dim>.json        per-dimension reports (each carries a
//	                             quarter of the shared wall-clock)
//	ablation/lnc_calibration.json the frozen μ, σ, ridge weights
//	ablation/<name>_features.json per-sample feature vectors
//
// LNC-fast (make benchmark-lncf) is the same pipeline with cost cuts
// fixed a priori: the model server runs the CCM LM in bf16
// (CCM_DTYPE=bfloat16), K=4 perturbations, and NLI only against each
// summary sentence's 4 highest-overlap source windows (-nli-topk 4).
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
	calPath      string
	host         string
	fit          bool
	alpha        float64
	k            int
	seed         uint64
	bootstrap    int
	debias       bool
	featuresIn   string
	name         string
	nliTopK      int
)

var dims = []string{"coherence", "consistency", "fluency", "relevance"}

func main() {
	flag.StringVar(&input, "input", "", "path to a dataset JSONL in SummEval layout (default: embedded SummEval)")
	flag.StringVar(&outputDimDir, "output-dim-dir", "output", "directory for lnc_<dim>.json")
	flag.StringVar(&ablationDir, "ablation-dir", "ablation", "directory for the feature dump")
	flag.StringVar(&calPath, "calibration", "ablation/lnc_calibration.json", "calibration file (written with -fit, read without)")
	flag.StringVar(&host, "host", "http://localhost:9200", "model server host")
	flag.BoolVar(&fit, "fit", true, "fit the calibration on the development half (first 50 articles)")
	flag.Float64Var(&alpha, "alpha", 10, "ridge penalty (fixed a priori)")
	flag.IntVar(&k, "k", 8, "CCM perturbations per candidate")
	flag.Uint64Var(&seed, "seed", 42, "CCM perturbation seed")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.BoolVar(&debias, "debias", false, "fit on human ranks with the copy-rate rank projected out (copy rate used only for fitting)")
	flag.StringVar(&featuresIn, "features-in", "", "reuse a feature dump written by an earlier run instead of calling the model server")
	flag.StringVar(&name, "name", "lnc", "report prefix: output/<name>_<dim>.json")
	flag.IntVar(&nliTopK, "nli-topk", 0, "score each summary sentence against only its K highest-overlap source windows (0 = all)")
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
	for _, s := range samples {
		if len(docs) == 0 || docs[len(docs)-1] != s.DocumentID {
			docs = append(docs, s.DocumentID)
		}
	}

	var X [][]float64
	var msPerSample float64
	if featuresIn != "" {
		X, msPerSample, err = readFeatures(featuresIn, samples)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("LNC: reusing features from %s (%.1f ms/sample when computed)", featuresIn, msPerSample)
	} else {
		X, msPerSample = computeFeatures(ctx, samples)
	}
	elapsed := time.Duration(msPerSample * float64(len(samples)) * float64(time.Millisecond))

	var cal *metrics.LNCCalibration
	if fit {
		dev := map[string]bool{}
		for _, d := range docs[:50] {
			dev[d] = true
		}
		var Xd [][]float64
		var copyRate []float64
		human := map[string][]float64{}
		for i, s := range samples {
			if !dev[s.DocumentID] {
				continue
			}
			Xd = append(Xd, X[i])
			copyRate = append(copyRate, metrics.CopyRate(s.Document, s.Candidate))
			human["coherence"] = append(human["coherence"], s.Coherence)
			human["consistency"] = append(human["consistency"], s.Consistency)
			human["fluency"] = append(human["fluency"], s.Fluency)
			human["relevance"] = append(human["relevance"], s.Relevance)
		}
		if !debias {
			copyRate = nil
		}
		cal, err = metrics.FitLNC(Xd, human, copyRate, alpha, "SummEval first 50 articles (dataset order)")
		if err != nil {
			log.Fatal(err)
		}
		if err := writeJSON(calPath, cal); err != nil {
			log.Fatal(err)
		}
		log.Printf("LNC: calibration fitted on %d development samples → %s", len(Xd), calPath)
	} else {
		cal, err = readCalibration(calPath)
		if err != nil {
			log.Fatal(err)
		}
	}

	norm := fmt.Sprintf("alpha=%g,debiased=%v,k=%d,seed=%d,nli_topk=%d,calibration=%s", cal.Alpha, cal.Debiased, k, seed, nliTopK, cal.FitOn)
	for _, dim := range dims {
		scores := make([]float64, len(samples))
		entries := make([]eval.Score, len(samples))
		for i := range samples {
			v, err := cal.Score(dim, X[i])
			if err != nil {
				log.Fatal(err)
			}
			scores[i] = v
			entries[i] = eval.Score{SampleID: samples[i].ID, Value: v}
		}
		report := eval.Report{
			Metric: name + "_" + dim, Norm: norm, Samples: len(samples),
			// Four reports from ONE run: cost-aware tools sum a
			// dimensional metric's reports, so each carries a quarter.
			RuntimeSec: elapsed.Seconds() / 4,
			Timestamp:  time.Now().UTC().Format(time.RFC3339),
			Scores:     entries,
			SummaryLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
				Bootstrap: bootstrap, Level: "summary",
			}),
			SystemLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
				Bootstrap: bootstrap, Level: "system",
			}),
		}
		if err := eval.NewReport(filepath.Join(outputDimDir, name+"_"+dim+".json"), report); err != nil {
			log.Fatal(err)
		}
	}

	if featuresIn != "" {
		return
	}
	rows := make([]featureRow, len(samples))
	for i := range samples {
		rows[i] = featureRow{samples[i].ID, X[i]}
	}
	if err := writeJSON(filepath.Join(ablationDir, name+"_features.json"), featureDump{
		Features: metrics.LNCFeatures, MsPerSample: msPerSample, Rows: rows,
	}); err != nil {
		log.Fatal(err)
	}
}

type featureRow struct {
	SampleID string    `json:"sample_id"`
	Features []float64 `json:"features"`
}

type featureDump struct {
	Features    []string     `json:"features"`
	MsPerSample float64      `json:"ms_per_sample"`
	Rows        []featureRow `json:"rows"`
}

func readFeatures(path string, samples []eval.Sample) ([][]float64, float64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	var d featureDump
	if err := json.Unmarshal(b, &d); err != nil {
		return nil, 0, fmt.Errorf("features %s: %w", path, err)
	}
	if len(d.Rows) != len(samples) {
		return nil, 0, fmt.Errorf("features %s: %d rows, expected %d", path, len(d.Rows), len(samples))
	}
	X := make([][]float64, len(samples))
	for i, r := range d.Rows {
		if r.SampleID != samples[i].ID {
			return nil, 0, fmt.Errorf("features %s: row %d is %s, expected %s", path, i, r.SampleID, samples[i].ID)
		}
		X[i] = r.Features
	}
	return X, d.MsPerSample, nil
}

func writeJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func readCalibration(path string) (*metrics.LNCCalibration, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("calibration %s not found; run with -fit first", path)
	}
	if err != nil {
		return nil, err
	}
	var c metrics.LNCCalibration
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("calibration %s: %w", path, err)
	}
	return &c, nil
}

// computeFeatures runs CCM, NLIG and SPL for every article and returns
// the per-sample feature vectors and the measured ms/sample.
func computeFeatures(ctx context.Context, samples []eval.Sample) ([][]float64, float64) {
	ccm := metrics.NewCCM(host)
	ccm.K, ccm.Seed = k, seed
	nlig := metrics.NewNLIG(host)
	nlig.TopK = nliTopK
	spl := metrics.NewSPL(host)

	var docs []string
	byDoc := map[string][]int{}
	for i, s := range samples {
		if _, ok := byDoc[s.DocumentID]; !ok {
			docs = append(docs, s.DocumentID)
		}
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], i)
	}
	log.Printf("LNC: %d articles, %d samples", len(docs), len(samples))

	bar := progressbar.NewOptions(len(samples),
		progressbar.OptionSetDescription("lnc"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	X := make([][]float64, len(samples))
	var tCCM, tNLI, tSPL time.Duration
	for _, d := range docs {
		idx := byDoc[d]
		src := samples[idx[0]].Document

		t := time.Now()
		cands := make([]metrics.CCMCandidate, len(idx))
		rendered := make([]string, len(idx))
		for j, i := range idx {
			cands[j] = ccm.Prepare(samples[i].ID, src, samples[i].Candidate)
			rendered[j] = cands[j].Text
		}
		cr, err := ccm.ScoreArticle(ctx, src, cands)
		if err != nil {
			log.Fatalf("article %s: %v", d, err)
		}
		tCCM += time.Since(t)

		t = time.Now()
		nr, err := nlig.ScoreArticle(ctx, src, rendered)
		if err != nil {
			log.Fatalf("article %s: %v", d, err)
		}
		tNLI += time.Since(t)

		t = time.Now()
		sr, err := spl.ScoreArticle(ctx, src, rendered)
		if err != nil {
			log.Fatalf("article %s: %v", d, err)
		}
		tSPL += time.Since(t)

		for j, i := range idx {
			X[i] = metrics.LNCFeatureVector(cr[j], nr[j], sr[j])
		}
		bar.Add(len(idx))
	}
	elapsed := tCCM + tNLI + tSPL
	N := float64(len(samples))
	log.Printf("LNC: CCM %.1f + NLI %.1f + SPL %.1f ms/sample = %.1f ms/sample",
		1000*tCCM.Seconds()/N, 1000*tNLI.Seconds()/N, 1000*tSPL.Seconds()/N, 1000*elapsed.Seconds()/N)
	return X, 1000 * elapsed.Seconds() / N
}
