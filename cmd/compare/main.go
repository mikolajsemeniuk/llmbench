package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"

	"github.com/mikolajsemeniuk/llmbench"
)

func main() {
	var flagA, flagB, flagOutput string
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

	out := llmbench.BuildCompareReport(read(flagA), read(flagB))
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		log.Fatalf("cannot marshal output: %v", err)
	}
	if err := os.WriteFile(flagOutput, data, 0644); err != nil {
		log.Fatalf("cannot write %s: %v", flagOutput, err)
	}
}
