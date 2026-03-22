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

// ---------------------------------------------------------------------------
// CLI flags
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Input types — mirror evaluate/main.go Report exactly so json.Unmarshal works
// ---------------------------------------------------------------------------

type Report struct {
	Metadata  Metadata          `json:"metadata"`
	Metrics   Metrics           `json:"aggregate"`
	PerLevel  []LevelMetrics    `json:"per_level"`
	RAG       RAGQualityMetrics `json:"rag_quality"`
	Summaries []Summary         `json:"per_task"`
	Records   []Record          `json:"runs"`
}

type Metadata struct {
	Provider    string `json:"provider"`
	Model       string `json:"model"`
	ModelDigest string `json:"model_digest,omitempty"`
	ModelFamily string `json:"model_family,omitempty"`
	ModelQuant  string `json:"model_quantization,omitempty"`
	Timestamp   string `json:"timestamp"`
	TotalTasks  int    `json:"total_tasks"`
	RunsPerTask int    `json:"runs_per_task"`
	TotalRuns   int    `json:"total_runs"`
	Seed        int64  `json:"random_seed"`
}

type Metrics struct {
	ESR        float64    `json:"esr"`
	ESRCI      [2]float64 `json:"esr_ci_95"`
	TSA        float64    `json:"tsa"`
	CHR        float64    `json:"chr"`
	DAAR       float64    `json:"daar"`
	FCSR       float64    `json:"fcsr"`
	LAE        float64    `json:"lae"`
	MTTR       float64    `json:"mttr_sec"`
	LatencyP50 float64    `json:"latency_p50_sec"`
	LatencyP95 float64    `json:"latency_p95_sec"`
	LatencyP99 float64    `json:"latency_p99_sec"`
}

type LevelMetrics struct {
	Name string  `json:"name"`
	ESR  float64 `json:"esr"`
	TSA  float64 `json:"tsa"`
	CHR  float64 `json:"chr"`
	Runs int     `json:"runs"`
}

type RAGQualityMetrics struct {
	MeanPrecisionAtK float64 `json:"mean_precision_at_k"`
	MeanRecallAtK    float64 `json:"mean_recall_at_k"`
	MeanMRR          float64 `json:"mean_mrr"`
	MeanNDCGAtK      float64 `json:"mean_ndcg_at_k"`
	MeanFScoreAtK    float64 `json:"mean_f1_at_k"`
}

type Summary struct {
	TaskID     string  `json:"task_id"`
	Level      string  `json:"level"`
	ESR        float64 `json:"esr"`
	TSA        float64 `json:"tsa"`
	CHR        float64 `json:"chr"`
	MeanLatSec float64 `json:"mean_latency_sec"`
}

type Record struct {
	TaskID           string  `json:"task_id"`
	RunIndex         int     `json:"run_index"`
	LatencySec       float64 `json:"latency_sec"`
	DiagCorrect      bool    `json:"diagnosis_correct"`
	ActionCorrect    bool    `json:"action_correct"`
	Hallucinations   int     `json:"hallucinations"`
	TotalEntities    int     `json:"total_entities"`
	Destructive      bool    `json:"destructive_action"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TokensPerSec     float64 `json:"tokens_per_sec"`
	Error            string  `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// Output types — CompareReport
// ---------------------------------------------------------------------------

// CompareReport is written to compare.json and read by cmd/report.
type CompareReport struct {
	// GeneratedAt is the UTC timestamp of this comparison run.
	GeneratedAt string `json:"generated_at"`

	// ModelA / ModelB are the full provider/model strings, e.g.
	// "ollama/qwen2.5:3b-instruct".
	ModelA string `json:"model_a"`
	ModelB string `json:"model_b"`

	// Aggregate holds the metric values for both models side-by-side
	// with the statistical test result for each metric.
	Aggregate []MetricComparison `json:"aggregate"`

	// PerLevel holds level-by-level breakdowns for both models.
	PerLevel []LevelComparison `json:"per_level"`

	// PerTask allows per-task delta visualisation in the report.
	PerTask []TaskComparison `json:"per_task"`

	// Raw holds the original reports for display/download.
	Raw struct {
		A Report `json:"a"`
		B Report `json:"b"`
	} `json:"raw"`
}

// MetricComparison holds a head-to-head comparison for a single scalar metric.
type MetricComparison struct {
	// Name is the metric abbreviation, e.g. "ESR", "TSA".
	Name string `json:"name"`

	// FullName is the human-readable label for the report UI.
	FullName string `json:"full_name"`

	// HigherIsBetter controls the colour coding in the report template:
	// true → higher value is green; false (CHR, DAAR, latency) → lower is green.
	HigherIsBetter bool `json:"higher_is_better"`

	ValueA float64 `json:"value_a"`
	ValueB float64 `json:"value_b"`

	// Delta = ValueA - ValueB (positive means A is better for HigherIsBetter metrics).
	Delta float64 `json:"delta"`

	// WilcoxonU is the U statistic from the rank-sum test on per-run values.
	// Present only for metrics that have a per-run sample (ESR, TSA, CHR).
	// For scalar-only metrics (LAE, MTTR) this is 0.
	WilcoxonU float64 `json:"wilcoxon_u"`

	// PValue is the two-sided p-value before correction.
	PValue float64 `json:"p_value"`

	// PValueCorrected is Bonferroni-corrected over the number of metrics tested.
	PValueCorrected float64 `json:"p_value_corrected"`

	// Significance is the conventional label: "***", "**", "*", or "n.s."
	Significance string `json:"significance"`

	// EffectSize is the rank-biserial correlation r (range −1 to +1).
	// Magnitude: |r| < 0.1 = negligible, 0.1–0.3 = small, 0.3–0.5 = medium, >0.5 = large.
	EffectSize float64 `json:"effect_size_r"`

	// EffectLabel is "negligible", "small", "medium", or "large".
	EffectLabel string `json:"effect_label"`
}

// LevelComparison holds per-level ESR/TSA/CHR for both models.
type LevelComparison struct {
	Level string  `json:"level"`
	ESRA  float64 `json:"esr_a"`
	ESRB  float64 `json:"esr_b"`
	TSAA  float64 `json:"tsa_a"`
	TSAB  float64 `json:"tsa_b"`
	CHRA  float64 `json:"chr_a"`
	CHRB  float64 `json:"chr_b"`
	RunsA int     `json:"runs_a"`
	RunsB int     `json:"runs_b"`
}

// TaskComparison holds per-task ESR delta (A − B) for the scatter view.
type TaskComparison struct {
	TaskID string  `json:"task_id"`
	Level  string  `json:"level"`
	ESRA   float64 `json:"esr_a"`
	ESRB   float64 `json:"esr_b"`
	Delta  float64 `json:"delta"` // ESRA − ESRB
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Comparison builder
// ---------------------------------------------------------------------------

func buildCompare(a, b Report) CompareReport {
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

	aggregate := []MetricComparison{
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

	cr := CompareReport{
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
func metricCmp(name, fullName string, higherIsBetter bool, va, vb float64, samplesA, samplesB []float64, nTests int) MetricComparison {
	u, p := llmbench.WilcoxonRankSum(samplesA, samplesB)
	pCorrected := math.Min(p*float64(nTests), 1.0) // Bonferroni
	sig := llmbench.WilcoxonSignificanceLabel(pCorrected)
	r := rankBiserialR(u, len(samplesA), len(samplesB))
	return MetricComparison{
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
func scalarCmp(name, fullName string, higherIsBetter bool, va, vb float64) MetricComparison {
	return MetricComparison{
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

func buildPerLevel(a, b Report) []LevelComparison {
	indexA := make(map[string]LevelMetrics)
	for _, l := range a.PerLevel {
		indexA[l.Name] = l
	}
	indexB := make(map[string]LevelMetrics)
	for _, l := range b.PerLevel {
		indexB[l.Name] = l
	}

	order := []string{"L1-diagnostic", "L2-repair", "L3-multi-step"}
	out := make([]LevelComparison, 0, len(order))
	for _, level := range order {
		la, oka := indexA[level]
		lb, okb := indexB[level]
		if !oka && !okb {
			continue
		}
		out = append(out, LevelComparison{
			Level: level,
			ESRA:  la.ESR, ESRB: lb.ESR,
			TSAA: la.TSA, TSAB: lb.TSA,
			CHRA: la.CHR, CHRB: lb.CHR,
			RunsA: la.Runs, RunsB: lb.Runs,
		})
	}
	return out
}

func buildPerTask(a, b Report) []TaskComparison {
	indexB := make(map[string]Summary)
	for _, s := range b.Summaries {
		indexB[s.TaskID] = s
	}

	out := make([]TaskComparison, 0, len(a.Summaries))
	for _, sa := range a.Summaries {
		sb := indexB[sa.TaskID]
		out = append(out, TaskComparison{
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
func successVector(records []Record) []float64 {
	v := make([]float64, len(records))
	for i, r := range records {
		if r.DiagCorrect && r.ActionCorrect {
			v[i] = 1.0
		}
	}
	return v
}

func actionVector(records []Record) []float64 {
	v := make([]float64, len(records))
	for i, r := range records {
		if r.ActionCorrect {
			v[i] = 1.0
		}
	}
	return v
}

// chrVector returns the per-run hallucination fraction.
func chrVector(records []Record) []float64 {
	v := make([]float64, len(records))
	for i, r := range records {
		if r.TotalEntities > 0 {
			v[i] = float64(r.Hallucinations) / float64(r.TotalEntities)
		}
	}
	return v
}

func latencyVector(records []Record) []float64 {
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

func mustReadReport(path string) Report {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("cannot read %s: %v", path, err)
	}
	var r Report
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
