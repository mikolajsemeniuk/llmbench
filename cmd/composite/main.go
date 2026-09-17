// cmd/composite assembles a per-dimension "composite" metric out of
// existing component score files, e.g. NLI entailment for consistency
// and LGS for the other three dimensions. This is NOT a new scoring
// algorithm -- it re-emits already-computed per-sample scores under a
// dimensional "composite_<dimension>.json" name, exactly the layout
// cmd/confound, cmd/compare and cmd/friedman already expect for
// dimensional metrics like unieval_consistency.json.
//
// Motivation (improvements.txt section 6): a single similarity-based
// signal (cosine, lexical overlap, position) plateaus at a confound-
// controlled partial rho of ~.23 on every dimension in the existing
// pool. The go/no-go probe this session found that a robust
// (ANLI-trained) entailment model clears that ceiling specifically on
// consistency (partial rho .296, nearly matching UniEval's .300) while
// being close to useless on the other three dimensions (partial rho
// .03-.15) -- exactly the dimension-specific signal improvements.txt
// predicted. This tool answers the follow-up question directly: what
// does the mean correlation look like if each dimension gets its best
// available cheap signal instead of one signal doing all four jobs?
//
// Usage:
//
//	go run ./cmd/composite \
//	  -coherence output/lgs.json \
//	  -consistency output/nlilarge.json \
//	  -fluency output/lgs.json \
//	  -relevance output/lgs.json \
//	  -output-dir output
//
// Each -<dimension> flag points at a flat (non-dimensional) report
// whose Scores are reused verbatim for that one dimension. Passing the
// same file to multiple dimensions is expected and is how a component
// that has not been built yet (e.g. fluency-via-perplexity,
// coherence-via-discourse-features) is honestly marked as "borrowed
// from LGS, not yet a real component" -- see the printed summary.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
)

var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

func main() {
	paths := map[string]*string{}
	for _, d := range dimensions {
		paths[d] = flag.String(d, "", "component report .json to use for this dimension (required)")
	}
	outputDir := flag.String("output-dir", "output", "directory to write composite_<dimension>.json into")
	name := flag.String("name", "composite", "base metric name (files become <name>_<dimension>.json)")
	flag.Parse()

	for _, d := range dimensions {
		if *paths[d] == "" {
			log.Fatalf("-%s is required (path to a component report .json)", d)
		}
	}

	fmt.Printf("Composite metric %q assembled from:\n", *name)
	for _, d := range dimensions {
		borrowed := ""
		// Flag a component as "borrowed" (not a dimension-specific
		// signal) when the same source file backs more than one
		// dimension -- this is the common case tonight, since only
		// consistency (NLI) is a genuinely new, validated component.
		for _, d2 := range dimensions {
			if d2 != d && *paths[d2] == *paths[d] {
				borrowed = "  [reused across dimensions -- not a dimension-specific component]"
				break
			}
		}
		fmt.Printf("  %-12s <- %s%s\n", d, *paths[d], borrowed)
	}

	for _, d := range dimensions {
		raw, err := os.ReadFile(*paths[d])
		if err != nil {
			log.Fatalf("read %s: %v", *paths[d], err)
		}
		var r map[string]any
		if err := json.Unmarshal(raw, &r); err != nil {
			log.Fatalf("decode %s: %v", *paths[d], err)
		}

		metricName := fmt.Sprintf("%s_%s", *name, d)
		r["metric"] = metricName
		r["norm"] = fmt.Sprintf("component_source=%s,original_metric=%v", *paths[d], r["metric"])
		// Summary/system-level correlation blocks are recomputed by
		// whatever downstream tool consumes this file (confound,
		// compare, friedman all recompute Spearman from raw Scores
		// against human ratings themselves) -- drop the borrowed
		// report's own correlation blocks so nobody mistakes them for
		// this dimension's correlation.
		delete(r, "summary_level")
		delete(r, "system_level")
		delete(r, "runs")

		out, err := json.MarshalIndent(r, "", "  ")
		if err != nil {
			log.Fatalf("encode %s: %v", metricName, err)
		}
		outPath := filepath.Join(*outputDir, metricName+".json")
		if err := os.WriteFile(outPath, out, 0o644); err != nil {
			log.Fatalf("write %s: %v", outPath, err)
		}
		fmt.Printf("  wrote %s\n", outPath)
	}
}
