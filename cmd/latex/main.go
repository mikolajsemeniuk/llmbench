package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"

	"github.com/mikolajsemeniuk/llmbench"
)

func main() {
	var flagFile, flagOutput string
	flag.StringVar(&flagFile, "file", "compare.json", "Path to compare JSON")
	flag.StringVar(&flagOutput, "output", "tables.tex", "Output .tex file")
	flag.Parse()

	raw, err := os.ReadFile(flagFile)
	if err != nil {
		log.Fatalf("cannot read %s: %v", flagFile, err)
	}

	var cr llmbench.CompareReport
	if err := json.Unmarshal(raw, &cr); err != nil {
		log.Fatalf("cannot parse: %v", err)
	}

	f, err := os.Create(flagOutput)
	if err != nil {
		log.Fatalf("cannot create %s: %v", flagOutput, err)
	}
	defer f.Close()

	if err := llmbench.RenderCompareLatex(f, cr); err != nil {
		log.Fatalf("latex render: %v", err)
	}
	log.Printf("wrote %s", flagOutput)
}
