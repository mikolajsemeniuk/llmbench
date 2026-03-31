package main

import (
	_ "embed"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"

	"github.com/mikolajsemeniuk/llmbench"
)

//go:embed index.html
var indexHTML string

func main() {
	var flagFile, flagAddr string
	flag.StringVar(&flagFile, "file", "results.json", "Path to benchmark JSON (single or compare)")
	flag.StringVar(&flagAddr, "addr", ":8080", "HTTP listen address")
	flag.Parse()

	raw, err := os.ReadFile(flagFile)
	if err != nil {
		log.Fatalf("cannot read %s: %v", flagFile, err)
	}

	data, err := llmbench.ParseReportFileJSON(raw)
	if err != nil {
		log.Fatalf("parse: %v", err)
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
		"deltaFmt": llmbench.FormatHTMLDelta,
	}

	tpl, err := template.New("report").Funcs(set).Parse(indexHTML)
	if err != nil {
		log.Fatalf("template parse error: %v", err)
	}

	mode := "single"
	if data.IsCompare {
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
