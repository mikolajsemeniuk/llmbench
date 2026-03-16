package main

import (
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
)

type Report struct {
	Metadata  Metadata         `json:"metadata"`
	Aggregate Aggregate        `json:"aggregate"`
	PerLevel  map[string]Level `json:"per_level"`
	RAG       RAGQuality       `json:"rag_quality"`
	PerTask   []TaskSummary    `json:"per_task"`
	Runs      []Run            `json:"runs"`
}

type Metadata struct {
	Model       string `json:"model"`
	Timestamp   string `json:"timestamp"`
	TotalTasks  int    `json:"total_tasks"`
	RunsPerTask int    `json:"runs_per_task"`
	TotalRuns   int    `json:"total_runs"`
	Seed        int64  `json:"random_seed"`
}

type Aggregate struct {
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

type Level struct {
	ESR  float64 `json:"esr"`
	TSA  float64 `json:"tsa"`
	CHR  float64 `json:"chr"`
	Runs int     `json:"runs"`
}

type RAGQuality struct {
	PrecisionAtK float64 `json:"mean_precision_at_k"`
	RecallAtK    float64 `json:"mean_recall_at_k"`
	MRR          float64 `json:"mean_mrr"`
	NDCGAtK      float64 `json:"mean_ndcg_at_k"`
	FScoreAtK    float64 `json:"mean_f1_at_k"`
}

type TaskSummary struct {
	TaskID     string  `json:"task_id"`
	Level      string  `json:"level"`
	ESR        float64 `json:"esr"`
	TSA        float64 `json:"tsa"`
	CHR        float64 `json:"chr"`
	MeanLatSec float64 `json:"mean_latency_sec"`
}

type Run struct {
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

// Ordered levels for template iteration.
type LevelEntry struct {
	Name string
	Level
}

type TemplateData struct {
	Report
	Levels []LevelEntry
}

var (
	//go:embed index.html
	index   string
	path    string
	address string
)

func main() {
	flag.StringVar(&path, "file", "results.json", "Path to benchmark results JSON")
	flag.StringVar(&address, "addr", ":8080", "HTTP listen address")
	flag.Parse()

	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("Cannot read %s: %v", path, err)
	}

	var report Report
	if err := json.Unmarshal(data, &report); err != nil {
		log.Fatalf("Cannot parse JSON: %v", err)
	}

	order := []string{"L1-diagnostic", "L2-repair", "L3-multi-step"}
	var levels []LevelEntry
	for _, name := range order {
		if l, ok := report.PerLevel[name]; ok {
			levels = append(levels, LevelEntry{Name: name, Level: l})
		}
	}

	set := template.FuncMap{
		"pct": func(v float64) string { return fmt.Sprintf("%.1f%%", v*100) },
		"f3":  func(v float64) string { return fmt.Sprintf("%.3f", v) },
		"f2":  func(v float64) string { return fmt.Sprintf("%.2f", v) },
		"f1":  func(v float64) string { return fmt.Sprintf("%.1f", v) },
		"f0":  func(v float64) string { return fmt.Sprintf("%.0f", v) },
		"add": func(a, b int) int { return a + b },
	}

	tmp, err := template.New("report").Funcs(set).Parse(index)
	if err != nil {
		log.Fatalf("Template parse error: %v", err)
	}

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")

		arg := TemplateData{Report: report, Levels: levels}
		if err := tmp.Execute(w, arg); err != nil {
			http.Error(w, err.Error(), 500)
		}
	})

	log.Fatal(http.ListenAndServe(address, nil))
}
