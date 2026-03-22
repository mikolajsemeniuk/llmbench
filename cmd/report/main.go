package main

// LLMBench: Report Server
//
// Reads a result JSON produced by cmd/evaluate OR a compare JSON produced
// by cmd/compare, and serves it as an HTML dashboard on localhost.
//
// The program detects which type of file it received automatically —
// compare.json has a "model_a" field, single reports have "metadata".
//
// No output is printed to stdout. All operational messages go to stderr
// via log. The HTTP server runs until Ctrl-C.
//
// Usage — single report:
//
//	go run ./cmd/report -file qwen.json
//
// Usage — comparison report:
//
//	go run ./cmd/report -file compare.json
//
// Usage — custom address:
//
//	go run ./cmd/report -file compare.json -addr :9090

import (
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"log"
	"math"
	"net/http"
	"os"
)

//go:embed index.html
var indexHTML string

var (
	flagFile string
	flagAddr string
)

// ---------------------------------------------------------------------------
// Single-report types  (mirrors evaluate/main.go Report exactly)
// ---------------------------------------------------------------------------

type SingleReport struct {
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
// Compare-report types  (mirrors compare/main.go CompareReport exactly)
// ---------------------------------------------------------------------------

type CompareReport struct {
	GeneratedAt string             `json:"generated_at"`
	ModelA      string             `json:"model_a"`
	ModelB      string             `json:"model_b"`
	Aggregate   []MetricComparison `json:"aggregate"`
	PerLevel    []LevelComparison  `json:"per_level"`
	PerTask     []TaskComparison   `json:"per_task"`
	Raw         struct {
		A SingleReport `json:"a"`
		B SingleReport `json:"b"`
	} `json:"raw"`
}

type MetricComparison struct {
	Name            string  `json:"name"`
	FullName        string  `json:"full_name"`
	HigherIsBetter  bool    `json:"higher_is_better"`
	ValueA          float64 `json:"value_a"`
	ValueB          float64 `json:"value_b"`
	Delta           float64 `json:"delta"`
	WilcoxonU       float64 `json:"wilcoxon_u"`
	PValue          float64 `json:"p_value"`
	PValueCorrected float64 `json:"p_value_corrected"`
	Significance    string  `json:"significance"`
	EffectSize      float64 `json:"effect_size_r"`
	EffectLabel     string  `json:"effect_label"`
}

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

type TaskComparison struct {
	TaskID string  `json:"task_id"`
	Level  string  `json:"level"`
	ESRA   float64 `json:"esr_a"`
	ESRB   float64 `json:"esr_b"`
	Delta  float64 `json:"delta"`
}

type TemplateData struct {
	IsCompare bool
	Single    SingleReport
	Compare   CompareReport
}

func main() {
	flag.StringVar(&flagFile, "file", "results.json", "Path to benchmark JSON (single or compare)")
	flag.StringVar(&flagAddr, "addr", ":8080", "HTTP listen address")
	flag.Parse()

	raw, err := os.ReadFile(flagFile)
	if err != nil {
		log.Fatalf("cannot read %s: %v", flagFile, err)
	}

	data, isCompare := parseFile(raw)

	funcMap := template.FuncMap{
		"pct":      func(v float64) string { return fmt.Sprintf("%.1f%%", v*100) },
		"f4":       func(v float64) string { return fmt.Sprintf("%.4f", v) },
		"f3":       func(v float64) string { return fmt.Sprintf("%.3f", v) },
		"f2":       func(v float64) string { return fmt.Sprintf("%.2f", v) },
		"f1":       func(v float64) string { return fmt.Sprintf("%.1f", v) },
		"f0":       func(v float64) string { return fmt.Sprintf("%.0f", v) },
		"add":      func(a, b int) int { return a + b },
		"deltaFmt": deltaFmt,
	}

	tmpl, err := template.New("report").Funcs(funcMap).Parse(indexHTML)
	if err != nil {
		log.Fatalf("template parse error: %v", err)
	}

	mode := "single"
	if isCompare {
		mode = "compare"
	}
	log.Printf("serving %s report from %s on http://localhost%s", mode, flagFile, flagAddr)

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := tmpl.Execute(w, data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})

	log.Fatal(http.ListenAndServe(flagAddr, nil))
}

// parseFile detects whether raw is a CompareReport or a SingleReport by
// probing for the "model_a" key, then unmarshals accordingly.
func parseFile(raw []byte) (TemplateData, bool) {
	// Quick probe — unmarshal into a map and check for the compare sentinel.
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(raw, &probe); err != nil {
		log.Fatalf("invalid JSON: %v", err)
	}

	if _, ok := probe["model_a"]; ok {
		var cr CompareReport
		if err := json.Unmarshal(raw, &cr); err != nil {
			log.Fatalf("cannot parse compare report: %v", err)
		}
		return TemplateData{IsCompare: true, Compare: cr}, true
	}

	var sr SingleReport
	if err := json.Unmarshal(raw, &sr); err != nil {
		log.Fatalf("cannot parse single report: %v", err)
	}
	return TemplateData{IsCompare: false, Single: sr}, false
}

// deltaFmt formats a delta value with a sign prefix, e.g. "+0.0842" or "−0.0312".
// Uses Unicode minus (−) for negative values so the template stays clean.
func deltaFmt(v float64) string {
	abs := math.Abs(v)
	if v > 0 {
		return fmt.Sprintf("+%.4f", abs)
	}
	if v < 0 {
		return fmt.Sprintf("−%.4f", abs)
	}
	return "0.0000"
}
