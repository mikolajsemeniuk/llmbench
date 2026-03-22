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

	"github.com/mikolajsemeniuk/llmbench"
)

//go:embed index.html
var indexHTML string

var (
	flagFile string
	flagAddr string
)

type CompareReport struct {
	GeneratedAt string             `json:"generated_at"`
	ModelA      string             `json:"model_a"`
	ModelB      string             `json:"model_b"`
	Aggregate   []MetricComparison `json:"aggregate"`
	PerLevel    []LevelComparison  `json:"per_level"`
	PerTask     []TaskComparison   `json:"per_task"`
	Raw         struct {
		A llmbench.Report `json:"a"`
		B llmbench.Report `json:"b"`
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
	Single    llmbench.Report
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

	var sr llmbench.Report
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
