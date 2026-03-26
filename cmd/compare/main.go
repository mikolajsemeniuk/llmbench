package main

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

func main() {
	flag.StringVar(&flagA, "a", "", "Path to first benchmark JSON (required)")
	flag.StringVar(&flagB, "b", "", "Path to second benchmark JSON (required)")
	flag.StringVar(&flagOutput, "output", "compare.json", "Output path for comparison JSON")
	flag.Parse()

	if flagA == "" || flagB == "" {
		log.Fatal("both -a and -b are required")
	}

	read := func(path string) llmbench.Report {
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

	reportA := read(flagA)
	reportB := read(flagB)

	out := buildCompare(reportA, reportB)
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		log.Fatalf("cannot marshal output: %v", err)
	}
	if err := os.WriteFile(flagOutput, data, 0644); err != nil {
		log.Fatalf("cannot write %s: %v", flagOutput, err)
	}
}

// rawTestResult holds the Wilcoxon U statistic and uncorrected p-value for a
// single metric before any familywise error rate adjustment is applied.
type rawTestResult struct {
	name           string
	fullName       string
	higherIsBetter bool
	valueA         float64
	valueB         float64
	u              float64
	pRaw           float64
	effectSize     float64
	effectLabel    string
}

func buildCompare(a, b llmbench.Report) llmbench.CompareReport {
	// -------------------------------------------------------------------------
	// Collect per-run sample vectors for every metric that admits a
	// Wilcoxon rank-sum test. Only metrics with a per-run binary or
	// continuous observation can be tested; aggregate scalars (LAE, MTTR,
	// FCSR, DAAR, latency percentiles) are reported without a p-value.
	// -------------------------------------------------------------------------
	successA := successVector(a.Records)
	successB := successVector(b.Records)

	actionA := actionVector(a.Records)
	actionB := actionVector(b.Records)

	chrA := chrVector(a.Records)
	chrB := chrVector(b.Records)

	latA := latencyVector(a.Records)
	latB := latencyVector(b.Records)

	// -------------------------------------------------------------------------
	// Step 1 — compute raw U statistics and uncorrected p-values.
	//
	// We do NOT correct here. The complete family of raw p-values must be
	// assembled first so the Holm-Bonferroni procedure can order them globally.
	// Applying per-metric corrections inline (the previous approach) is
	// statistically equivalent to independent Bonferroni tests and loses the
	// step-down power advantage that Holm (1979) provides.
	// -------------------------------------------------------------------------
	rawTests := []rawTestResult{
		buildRaw("ESR", "Execution Success Rate", true, a.Metrics.ESR, b.Metrics.ESR, successA, successB),
		buildRaw("TSA", "Tool Selection Accuracy", true, a.Metrics.TSA, b.Metrics.TSA, actionA, actionB),
		buildRaw("CHR", "Context Hallucination Rate", false, a.Metrics.CHR, b.Metrics.CHR, chrA, chrB),
		buildRaw("LatP50", "Latency p50 (s)", false, a.Metrics.LatencyP50, b.Metrics.LatencyP50, latA, latB),
	}

	// -------------------------------------------------------------------------
	// Step 2 — apply Holm-Bonferroni correction to the complete p-value family.
	//
	// All m=4 raw p-values are passed together. The procedure:
	//   (a) sorts p-values ascending: p_(1) ≤ p_(2) ≤ … ≤ p_(m)
	//   (b) adjusts each: p̃_(k) = min(1, max_{j≤k} (m−j+1)·p_(j))
	//   (c) returns adjusted values in the original input order.
	//
	// This controls the familywise error rate (FWER) at α=0.05 while being
	// uniformly more powerful than classical Bonferroni. It is the procedure
	// required by ACM TOIS and IEEE Access when m > 1 comparisons are made
	// simultaneously on the same dataset.
	// -------------------------------------------------------------------------
	rawPs := make([]float64, len(rawTests))
	for i, rt := range rawTests {
		rawPs[i] = rt.pRaw
	}
	correctedPs := llmbench.HolmBonferroniCorrection(rawPs)

	// -------------------------------------------------------------------------
	// Step 3 — assemble MetricComparison objects using corrected p-values.
	// -------------------------------------------------------------------------
	const correctionMethod = "holm-bonferroni"

	testedMetrics := make([]llmbench.MetricComparison, len(rawTests))
	for i, rt := range rawTests {
		pCorr := correctedPs[i]
		testedMetrics[i] = llmbench.MetricComparison{
			Name:             rt.name,
			FullName:         rt.fullName,
			HigherIsBetter:   rt.higherIsBetter,
			ValueA:           rt.valueA,
			ValueB:           rt.valueB,
			Delta:            rt.valueA - rt.valueB,
			WilcoxonU:        rt.u,
			PValue:           rt.pRaw,
			PValueCorrected:  pCorr,
			CorrectionMethod: correctionMethod,
			Significance:     llmbench.WilcoxonSignificanceLabel(pCorr),
			EffectSize:       rt.effectSize,
			EffectLabel:      rt.effectLabel,
		}
	}

	// Scalar-only metrics: no per-run sample → no rank-sum test possible.
	// CorrectionMethod = "n/a" signals to reviewers that these values are
	// descriptive only and were excluded from the Holm-Bonferroni family.
	scalarMetrics := []llmbench.MetricComparison{
		scalarCmp("FCSR", "First Call Success Rate", true, a.Metrics.FCSR, b.Metrics.FCSR),
		scalarCmp("DAAR", "Destructive Action Rate", false, a.Metrics.DAAR, b.Metrics.DAAR),
		scalarCmp("LAE", "Latency-Action Efficiency", true, a.Metrics.LAE, b.Metrics.LAE),
		scalarCmp("MTTR", "Mean Time To Recovery (s)", false, a.Metrics.MTTR, b.Metrics.MTTR),
		scalarCmp("LatP95", "Latency p95 (s)", false, a.Metrics.LatencyP95, b.Metrics.LatencyP95),
		scalarCmp("LatP99", "Latency p99 (s)", false, a.Metrics.LatencyP99, b.Metrics.LatencyP99),
	}

	aggregate := append(testedMetrics, scalarMetrics...)

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

// buildRaw runs the Wilcoxon rank-sum test for a single metric and returns the
// uncorrected result. The p-value stored here MUST NOT be compared against α
// directly — it is an input to HolmBonferroniCorrection along with all sibling
// metric p-values before any significance decision is made.
func buildRaw(
	name, fullName string,
	higherIsBetter bool,
	va, vb float64,
	samplesA, samplesB []float64,
) rawTestResult {
	u, p := llmbench.WilcoxonRankSum(samplesA, samplesB)
	r := rankBiserialR(u, len(samplesA), len(samplesB))
	return rawTestResult{
		name:           name,
		fullName:       fullName,
		higherIsBetter: higherIsBetter,
		valueA:         va,
		valueB:         vb,
		u:              u,
		pRaw:           p,
		effectSize:     r,
		effectLabel:    effectLabel(math.Abs(r)),
	}
}

// scalarCmp builds a MetricComparison for aggregate scalars that have no
// per-run observation vector (FCSR, DAAR, LAE, MTTR, percentiles beyond p50).
// These metrics are excluded from the Holm-Bonferroni family because the test
// statistic cannot be computed without a sample distribution.
func scalarCmp(name, fullName string, higherIsBetter bool, va, vb float64) llmbench.MetricComparison {
	return llmbench.MetricComparison{
		Name:             name,
		FullName:         fullName,
		HigherIsBetter:   higherIsBetter,
		ValueA:           va,
		ValueB:           vb,
		Delta:            va - vb,
		CorrectionMethod: "n/a",
		Significance:     "n/a",
		EffectLabel:      "n/a",
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
	sort.Slice(out, func(i, j int) bool { return out[i].TaskID < out[j].TaskID })
	return out
}

// ---------------------------------------------------------------------------
// Sample vector helpers
// ---------------------------------------------------------------------------

// successVector returns 1.0 for each run that was fully successful (both
// diagnosis and action correct), 0.0 otherwise. This is the per-run binary
// observation used by the Wilcoxon rank-sum test for the ESR comparison.
func successVector(records []llmbench.Record) []float64 {
	v := make([]float64, len(records))
	for i, r := range records {
		if r.DiagCorrect && r.ActionCorrect {
			v[i] = 1.0
		}
	}
	return v
}

// actionVector returns 1.0 for each run where the model selected the correct
// remediation action (ActionCorrect), used for the TSA Wilcoxon test.
func actionVector(records []llmbench.Record) []float64 {
	v := make([]float64, len(records))
	for i, r := range records {
		if r.ActionCorrect {
			v[i] = 1.0
		}
	}
	return v
}

// chrVector returns the per-run hallucination fraction for CHR testing.
// A run with TotalEntities == 0 contributes 0.0 (no grounded entities → no
// hallucination opportunity), consistent with the CHR formula denominator guard.
func chrVector(records []llmbench.Record) []float64 {
	v := make([]float64, len(records))
	for i, r := range records {
		if r.TotalEntities > 0 {
			v[i] = float64(r.Hallucinations) / float64(r.TotalEntities)
		}
	}
	return v
}

// latencyVector returns the wall-clock latency in seconds for each run,
// used for the p50 latency Wilcoxon test.
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
// correlation coefficient r, the standard effect size reported alongside the
// rank-sum test in ACM TOIS and IEEE Access submissions.
//
// r = 1 − (2U / (n_a × n_b))
//
// Sign convention: r > 0 → group A stochastically dominates B;
// r < 0 → B dominates A. Magnitude thresholds follow Wendt (1972):
// |r| < 0.10 negligible, 0.10–0.30 small, 0.30–0.50 medium, > 0.50 large.
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
