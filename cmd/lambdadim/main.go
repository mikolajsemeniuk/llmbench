// cmd/lambdadim answers reviewer #5 #7: the ablation table
// (paper/ablation.gen.tex) reports one λ* selected by MEAN Spearman ρ
// across all four SummEval dimensions, but each dimension individually
// may prefer a different exponent. This binary re-reads the same
// per-sample-free per-dimension summary-level Spearman values already
// stored in ablation/lgs_lead_{dev,test}_l*.json and
// ablation/lgs_recall_{dev,test}.json (no new runs, no embedder), finds
// the dev-argmax λ separately for each dimension, and reports the
// matching test-split value. This is a decision matrix, not a new
// experiment: it makes explicit what the ablation table's aggregate
// already implies.
//
// Usage:
//
//	go run ./cmd/lambdadim -input ablation -output paper/lambdadim.gen.tex
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
)

var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

var dimensionShort = map[string]string{
	"coherence":   "Coherence",
	"consistency": "Consistency",
	"fluency":     "Fluency",
	"relevance":   "Relevance",
}

type variant struct {
	lambda float64
	split  string // "first50" | "last50"
	dims   map[string]float64
}

func main() {
	var inputDir, output string
	flag.StringVar(&inputDir, "input", "ablation", "directory containing LGS ablation JSON reports")
	flag.StringVar(&output, "output", "paper/lambdadim.gen.tex", "path to write LaTeX table (- for stdout)")
	flag.Parse()

	variants, err := loadVariants(inputDir)
	if err != nil {
		log.Fatal(err)
	}
	dev, test := splitByHalf(variants)
	if len(dev) < 2 || len(test) < 2 {
		log.Fatalf("need at least 2 lambda values on both dev and test (got %d dev, %d test)", len(dev), len(test))
	}

	type row struct {
		dim       string
		lambdaHat float64
		devRho    float64
		testRho   float64
		l0DevRho  float64
		l0TestRho float64
	}
	rows := make([]row, 0, len(dimensions))
	for _, dim := range dimensions {
		bestLambda, bestDevRho := argmax(dev, dim)
		testRho, ok := lookupTest(test, bestLambda, dim)
		if !ok {
			log.Fatalf("no test-split value for lambda=%g dimension=%s", bestLambda, dim)
		}
		l0Dev, _ := lookupTest(dev, 0, dim)
		l0Test, _ := lookupTest(test, 0, dim)
		rows = append(rows, row{dim, bestLambda, bestDevRho, testRho, l0Dev, l0Test})
	}

	var b strings.Builder
	fmt.Fprintln(&b, "Per-dimension lead-bias exponent decision matrix (dev-argmax, verified on test)\n")
	fmt.Fprintf(&b, "%-14s %8s %10s %10s %12s %12s\n", "Dimension", "lambda^", "dev rho", "test rho", "l=0 dev", "l=0 test")
	for _, r := range rows {
		fmt.Fprintf(&b, "%-14s %8g %10.4f %10.4f %12.4f %12.4f\n",
			dimensionShort[r.dim], r.lambdaHat, r.devRho, r.testRho, r.l0DevRho, r.l0TestRho)
	}
	fmt.Print(b.String())

	if output != "" {
		var t strings.Builder
		fmt.Fprintln(&t, `\begin{table}[t]`)
		fmt.Fprintln(&t, `\centering`)
		fmt.Fprintln(&t, `\caption{Per-dimension lead-bias exponent decision matrix. The ablation table selects one $\lambda^{\star}$ by mean Spearman $\rho$ across all four dimensions; this table instead selects the dev-argmax $\lambda$ separately for each dimension and reports it on the held-out test split, alongside the no-prior ($\lambda{=}0$) reference. A practitioner tuning LGS for a single dimension should read this table, not the aggregate one.}`)
		fmt.Fprintln(&t, `\label{tab:lambdadim}`)
		fmt.Fprintln(&t, `\small`)
		fmt.Fprintln(&t, `\linespread{1}\selectfont`)
		fmt.Fprintln(&t, `\begin{tabular}{@{}lrrrrr@{}}`)
		fmt.Fprintln(&t, `\toprule`)
		fmt.Fprintln(&t, `Dimension & $\hat\lambda$ & Dev $\rho$ & Test $\rho$ & $\lambda{=}0$ dev & $\lambda{=}0$ test \\`)
		fmt.Fprintln(&t, `\midrule`)
		for _, r := range rows {
			fmt.Fprintf(&t, "%s & %g & %s & %s & %s & %s \\\\\n",
				dimensionShort[r.dim], r.lambdaHat,
				stripLeadingZero(r.devRho), stripLeadingZero(r.testRho),
				stripLeadingZero(r.l0DevRho), stripLeadingZero(r.l0TestRho))
		}
		fmt.Fprintln(&t, `\bottomrule`)
		fmt.Fprintln(&t, `\end{tabular}`)
		fmt.Fprintln(&t, `\end{table}`)

		if err := writeFile(output, t.String()); err != nil {
			log.Fatal(err)
		}
		if output != "-" {
			fmt.Fprintf(os.Stderr, "\nLaTeX table written to %s\n", output)
		}
	}
}

func loadVariants(dir string) ([]variant, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "lgs_*.json"))
	if err != nil {
		return nil, fmt.Errorf("glob: %w", err)
	}
	sort.Strings(matches)

	out := make([]variant, 0, len(matches))
	for _, p := range matches {
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", p, err)
		}
		var r eval.Report
		if err := json.Unmarshal(raw, &r); err != nil {
			return nil, fmt.Errorf("decode %s: %w", p, err)
		}
		if r.Metric != "lgs" {
			continue
		}
		nm := parseNorm(r.Norm)
		split := nm["split"]
		if split != "first50" && split != "last50" {
			continue
		}
		embed := nm["embed_model"]
		if embed != "" && stripTag(embed) != "nomic-embed-text" {
			continue
		}
		lam, ok := parseFloat(nm["lead_lambda"])
		if !ok {
			continue
		}
		dims := map[string]float64{}
		for _, d := range r.SummaryLevel.Dimensions {
			dims[d.Name] = d.Spearman
		}
		out = append(out, variant{lambda: lam, split: split, dims: dims})
	}
	return out, nil
}

func splitByHalf(vs []variant) (dev, test []variant) {
	seenDev := map[float64]bool{}
	seenTest := map[float64]bool{}
	for _, v := range vs {
		if v.split == "first50" {
			if seenDev[v.lambda] {
				continue
			}
			seenDev[v.lambda] = true
			dev = append(dev, v)
		} else {
			if seenTest[v.lambda] {
				continue
			}
			seenTest[v.lambda] = true
			test = append(test, v)
		}
	}
	return dev, test
}

func argmax(vs []variant, dim string) (lambda, rho float64) {
	best := -1.0
	for _, v := range vs {
		if r, ok := v.dims[dim]; ok && r > best {
			best, lambda = r, v.lambda
		}
	}
	return lambda, best
}

func lookupTest(vs []variant, lambda float64, dim string) (float64, bool) {
	for _, v := range vs {
		if v.lambda == lambda {
			r, ok := v.dims[dim]
			return r, ok
		}
	}
	return 0, false
}

func parseNorm(norm string) map[string]string {
	out := map[string]string{}
	for _, p := range strings.Split(norm, ",") {
		p = strings.TrimSpace(p)
		if k, v, ok := strings.Cut(p, "="); ok {
			out[k] = v
		}
	}
	return out
}

func parseFloat(s string) (float64, bool) {
	var f float64
	if s == "" {
		return 0, false
	}
	if _, err := fmt.Sscanf(s, "%g", &f); err != nil {
		return 0, false
	}
	return f, true
}

func stripTag(s string) string {
	if i := strings.Index(s, ":"); i >= 0 {
		return s[:i]
	}
	return s
}

func stripLeadingZero(x float64) string {
	s := fmt.Sprintf("%.4f", x)
	if x >= 0 && x < 1 {
		return strings.TrimPrefix(s, "0")
	}
	if x > -1 && x < 0 {
		return "-" + strings.TrimPrefix(s[1:], "0")
	}
	return s
}

func writeFile(path, content string) error {
	var w io.Writer = os.Stdout
	if path != "" && path != "-" {
		if dir := filepath.Dir(path); dir != "." && dir != "" {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return fmt.Errorf("mkdir %s: %w", dir, err)
			}
		}
		f, err := os.Create(path)
		if err != nil {
			return fmt.Errorf("create %s: %w", path, err)
		}
		defer f.Close()
		w = f
	}
	_, err := io.WriteString(w, content)
	return err
}
