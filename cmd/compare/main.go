package main

// LLMBench: Compare
//
// Reads two benchmark result JSONs produced by cmd/evaluate, computes
// pairwise statistical tests (Wilcoxon rank-sum + rank-biserial r effect
// size, Bonferroni-corrected p-values), and writes a compare.json file
// suitable for cmd/report to render.
//
// No output is printed to stdout on success — only the JSON file is written.
// Errors go to stderr via log.Fatal.
//
// Usage:
//
//	go run ./cmd/compare \
//	  -a qwen.json \
//	  -b llama-pro.json \
//	  -output compare.json

import (
	"encoding/json"
	"flag"
	"log"
	"math"
	"os"
	"sort"
	"time"

	"github.com/mikolajsemeniuk/llmbench"
)

var (
	flagA      string
	flagB      string
	flagOutput string
)

func init() {
	flag.StringVar(&flagA, "a", "", "Path to first benchmark JSON (required)")
	flag.StringVar(&flagB, "b", "", "Path to second benchmark JSON (required)")
	flag.StringVar(&flagOutput, "output", "compare.json", "Output path for comparison JSON")
}

func main() {
	flag.Parse()

	if flagA == "" || flagB == "" {
		log.Fatal("both -a and -b are required")
	}

	reportA := mustReadReport(flagA)
	reportB := mustReadReport(flagB)

	out := buildCompare(reportA, reportB)
	mustWriteJSON(flagOutput, out)
}

func buildCompare(a, b llmbench.Report) llmbench.CompareReport {
	// Collect per-run binary vectors for statistical tests.
	// success[i] = 1.0 if run i was DiagnosisCorrect && ActionCorrect, else 0.0
	successA := successVector(a.Records)
	successB := successVector(b.Records)

	actionA := actionVector(a.Records)
	actionB := actionVector(b.Records)

	chrA := chrVector(a.Records)
	chrB := chrVector(b.Records)

	latA := latencyVector(a.Records)
	latB := latencyVector(b.Records)

	// Number of metrics being tested — used for Bonferroni correction.
	// We test: ESR, TSA, CHR, latency_p50 = 4 comparisons.
	const nTests = 4

	aggregate := []llmbench.MetricComparison{
		metricCmp("ESR", "Execution Success Rate", true, a.Metrics.ESR, b.Metrics.ESR, successA, successB, nTests),
		metricCmp("TSA", "Tool Selection Accuracy", true, a.Metrics.TSA, b.Metrics.TSA, actionA, actionB, nTests),
		metricCmp("CHR", "Context Hallucination Rate", false, a.Metrics.CHR, b.Metrics.CHR, chrA, chrB, nTests),
		metricCmp("LatP50", "Latency p50 (s)", false, a.Metrics.LatencyP50, b.Metrics.LatencyP50, latA, latB, nTests),
		scalarCmp("FCSR", "First Call Success Rate", true, a.Metrics.FCSR, b.Metrics.FCSR),
		scalarCmp("DAAR", "Destructive Action Rate", false, a.Metrics.DAAR, b.Metrics.DAAR),
		scalarCmp("LAE", "Latency-Action Efficiency", true, a.Metrics.LAE, b.Metrics.LAE),
		scalarCmp("MTTR", "Mean Time To Recovery (s)", false, a.Metrics.MTTR, b.Metrics.MTTR),
		scalarCmp("LatP95", "Latency p95 (s)", false, a.Metrics.LatencyP95, b.Metrics.LatencyP95),
		scalarCmp("LatP99", "Latency p99 (s)", false, a.Metrics.LatencyP99, b.Metrics.LatencyP99),
	}

	perLevel := buildPerLevel(a, b)
	perTask := buildPerTask(a, b)

	cr := llmbench.CompareReport{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		ModelA:      a.Metadata.Provider + "/" + a.Metadata.Model,
		ModelB:      b.Metadata.Provider + "/" + b.Metadata.Model,
		Aggregate:   aggregate,
		PerLevel:    perLevel,
		PerTask:     perTask,
	}
	cr.Raw.A = a
	cr.Raw.B = b
	return cr
}

// metricCmp builds a MetricComparison for metrics that have per-run samples
// (enables Wilcoxon + effect size).
func metricCmp(name, fullName string, higherIsBetter bool, va, vb float64, samplesA, samplesB []float64, nTests int) llmbench.MetricComparison {
	u, p := llmbench.WilcoxonRankSum(samplesA, samplesB)
	pCorrected := math.Min(p*float64(nTests), 1.0) // Bonferroni
	sig := llmbench.WilcoxonSignificanceLabel(pCorrected)
	r := rankBiserialR(u, len(samplesA), len(samplesB))
	return llmbench.MetricComparison{
		Name:            name,
		FullName:        fullName,
		HigherIsBetter:  higherIsBetter,
		ValueA:          va,
		ValueB:          vb,
		Delta:           va - vb,
		WilcoxonU:       u,
		PValue:          p,
		PValueCorrected: pCorrected,
		Significance:    sig,
		EffectSize:      r,
		EffectLabel:     effectLabel(math.Abs(r)),
	}
}

// scalarCmp builds a MetricComparison for scalar-only metrics (no per-run sample).
func scalarCmp(name, fullName string, higherIsBetter bool, va, vb float64) llmbench.MetricComparison {
	return llmbench.MetricComparison{
		Name:           name,
		FullName:       fullName,
		HigherIsBetter: higherIsBetter,
		ValueA:         va,
		ValueB:         vb,
		Delta:          va - vb,
		Significance:   "n/a",
		EffectLabel:    "n/a",
	}
}

func buildPerLevel(a, b llmbench.Report) []llmbench.LevelComparison {
	indexA := make(map[string]llmbench.LevelMetrics)
	for _, l := range a.PerLevel {
		indexA[l.Name] = l
	}
	indexB := make(map[string]llmbench.LevelMetrics)
	for _, l := range b.PerLevel {
		indexB[l.Name] = l
	}

	order := []string{"L1-diagnostic", "L2-repair", "L3-multi-step"}
	out := make([]llmbench.LevelComparison, 0, len(order))
	for _, level := range order {
		la, oka := indexA[level]
		lb, okb := indexB[level]
		if !oka && !okb {
			continue
		}
		out = append(out, llmbench.LevelComparison{
			Level: level,
			ESRA:  la.ESR, ESRB: lb.ESR,
			TSAA: la.TSA, TSAB: lb.TSA,
			CHRA: la.CHR, CHRB: lb.CHR,
			RunsA: la.Runs, RunsB: lb.Runs,
		})
	}
	return out
}

func buildPerTask(a, b llmbench.Report) []llmbench.TaskComparison {
	indexB := make(map[string]llmbench.Summary)
	for _, s := range b.Summaries {
		indexB[s.TaskID] = s
	}

	out := make([]llmbench.TaskComparison, 0, len(a.Summaries))
	for _, sa := range a.Summaries {
		sb := indexB[sa.TaskID]
		out = append(out, llmbench.TaskComparison{
			TaskID: sa.TaskID,
			Level:  sa.Level,
			ESRA:   sa.ESR,
			ESRB:   sb.ESR,
			Delta:  sa.ESR - sb.ESR,
		})
	}
	// Sort by task ID for stable output.
	sort.Slice(out, func(i, j int) bool { return out[i].TaskID < out[j].TaskID })
	return out
}

// ---------------------------------------------------------------------------
// Sample vector helpers
// ---------------------------------------------------------------------------

// successVector returns 1.0 per run that was fully successful, 0.0 otherwise.
func successVector(records []llmbench.Record) []float64 {
	v := make([]float64, len(records))
	for i, r := range records {
		if r.DiagCorrect && r.ActionCorrect {
			v[i] = 1.0
		}
	}
	return v
}

func actionVector(records []llmbench.Record) []float64 {
	v := make([]float64, len(records))
	for i, r := range records {
		if r.ActionCorrect {
			v[i] = 1.0
		}
	}
	return v
}

// chrVector returns the per-run hallucination fraction.
func chrVector(records []llmbench.Record) []float64 {
	v := make([]float64, len(records))
	for i, r := range records {
		if r.TotalEntities > 0 {
			v[i] = float64(r.Hallucinations) / float64(r.TotalEntities)
		}
	}
	return v
}

func latencyVector(records []llmbench.Record) []float64 {
	v := make([]float64, len(records))
	for i, r := range records {
		v[i] = r.LatencySec
	}
	return v
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

// rankBiserialR converts a Wilcoxon U statistic to the rank-biserial
// correlation r, the standard effect size for the rank-sum test.
//
// r = 1 − (2U / (n_a × n_b))
//
// r > 0 means group A tends to have higher values; r < 0 means group B does.
// Magnitude conventions: |r| < 0.10 = negligible, 0.10–0.30 = small,
// 0.30–0.50 = medium, > 0.50 = large.
func rankBiserialR(u float64, na, nb int) float64 {
	denom := float64(na * nb)
	if denom == 0 {
		return 0
	}
	return 1.0 - (2*u)/denom
}

func effectLabel(absR float64) string {
	switch {
	case absR >= 0.50:
		return "large"
	case absR >= 0.30:
		return "medium"
	case absR >= 0.10:
		return "small"
	default:
		return "negligible"
	}
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

func mustReadReport(path string) llmbench.Report {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("cannot read %s: %v", path, err)
	}

	var r llmbench.Report
	if err := json.Unmarshal(data, &r); err != nil {
		log.Fatalf("cannot parse %s: %v", path, err)
	}

	return r
}

func mustWriteJSON(path string, v any) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Fatalf("cannot marshal output: %v", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		log.Fatalf("cannot write %s: %v", path, err)
	}
}
