package main

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

type TemplateData struct {
	IsCompare bool
	Single    llmbench.Report
	Compare   llmbench.CompareReport
}

func main() {
	flag.StringVar(&flagFile, "file", "results.json", "Path to benchmark JSON (single or compare)")
	flag.StringVar(&flagAddr, "addr", ":8080", "HTTP listen address")
	flag.Parse()

	raw, err := os.ReadFile(flagFile)
	if err != nil {
		log.Fatalf("cannot read %s: %v", flagFile, err)
	}

	set := template.FuncMap{
		"pct":      func(v float64) string { return fmt.Sprintf("%.1f%%", v*100) },
		"f4":       func(v float64) string { return fmt.Sprintf("%.4f", v) },
		"f3":       func(v float64) string { return fmt.Sprintf("%.3f", v) },
		"f2":       func(v float64) string { return fmt.Sprintf("%.2f", v) },
		"f1":       func(v float64) string { return fmt.Sprintf("%.1f", v) },
		"f0":       func(v float64) string { return fmt.Sprintf("%.0f", v) },
		"add":      func(a, b int) int { return a + b },
		"mul":      func(a, b int) int { return a * b },
		"mulf":     func(a, b float64) float64 { return a * b },
		"deltaFmt": deltaFmt,
	}

	tpl, err := template.New("report").Funcs(set).Parse(indexHTML)
	if err != nil {
		log.Fatalf("template parse error: %v", err)
	}

	data, isCompare := parseFile(raw)
	mode := "single"
	if isCompare {
		mode = "compare"
	}
	log.Printf("serving %s report from %s on http://localhost%s", mode, flagFile, flagAddr)

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := tpl.Execute(w, data); err != nil {
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
		var cr llmbench.CompareReport
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
