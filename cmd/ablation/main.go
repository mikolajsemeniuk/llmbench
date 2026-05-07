// cmd/ablation reads LGS ablation reports from a directory (default
// `ablation/`) and renders a LaTeX table of the lead-bias λ sweep on
// the held-out development split, with the dev-selected λ* re-evaluated
// on the test split. The dev-selected canonical row is flagged with a ★.
//
// Output target matches the rest of the paper pipeline (default
// `paper/ablation.tex`).
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
	"strconv"
	"strings"

	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
)

var (
	inputDir   string
	outputTex  string
	lambdaStar float64
)

func main() {
	flag.StringVar(&inputDir, "input", "ablation", "directory containing ablation JSON reports")
	flag.StringVar(&outputTex, "output", "paper/ablation.tex", "path to write LaTeX table")
	flag.Float64Var(&lambdaStar, "lambda-star", -1.0, "canonical λ* selected on dev split (−1 disables ★)")
	flag.Parse()

	matches, err := filepath.Glob(filepath.Join(inputDir, "*.json"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	if len(matches) == 0 {
		log.Fatalf("no JSON files in %s", inputDir)
	}
	sort.Strings(matches)

	rows := make([]Row, 0, len(matches))
	for _, p := range matches {
		row, ok := parseRow(p)
		if !ok {
			log.Printf("skipping %s (unrecognised ablation variant)", p)
			continue
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		log.Fatalf("no ablation variants recognised in %s", inputDir)
	}

	rows = dedupRows(rows)
	sortRows(rows)
	if lambdaStar >= 0 {
		setCanonical(rows, lambdaStar)
	}

	if err := writeFile(outputTex, renderLatex(rows)); err != nil {
		log.Fatalf("write tex: %v", err)
	}
	log.Printf("ablation: %d rows → %s", len(rows), outputTex)
}

// Row is one line of the ablation table.
type Row struct {
	Label          string
	Source         string
	LeadBiasLambda float64
	Split          string // "first50" | "last50" | "all"
	IsCanonical    bool
	Coh            float64
	Con            float64
	Flu            float64
	Rel            float64
}

func parseRow(path string) (Row, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read %s: %v", path, err)
	}
	var r eval.Report
	if err := json.Unmarshal(raw, &r); err != nil {
		log.Fatalf("decode %s: %v", path, err)
	}
	if r.Metric != "lgs" {
		return Row{}, false
	}

	row := Row{Source: filepath.Base(path)}
	row.Coh, row.Con, row.Flu, row.Rel = extractDims(r.SummaryLevel)
	row.LeadBiasLambda = parseField(r.Norm, "lead_lambda")
	row.Split = parseStringField(r.Norm, "split")

	// The ablation table is the canonical lead-bias λ sweep with the
	// nomic-embed-text backbone. Embedder-ablation reports and the
	// cross-embedder bge-m3 verification sweep also live in
	// ablation/ but are rendered into separate tables; filter them
	// out here. Empty embed_model is treated as canonical (legacy
	// coarse-sweep files predate the field).
	embedder := parseStringField(r.Norm, "embed_model")
	if embedder != "" && embedder != "nomic-embed-text" {
		return Row{}, false
	}
	if row.Split == "all" {
		return Row{}, false
	}

	var paramStr string
	if row.LeadBiasLambda == 0 {
		paramStr = "Recall baseline ($\\lambda=0$)"
	} else {
		paramStr = fmt.Sprintf("Lead-bias ($\\lambda=%g$)", row.LeadBiasLambda)
	}
	switch row.Split {
	case "first50":
		row.Label = paramStr + " (dev)"
	case "last50":
		row.Label = paramStr + " (test)"
	case "all":
		row.Label = paramStr + " (full set)"
	default:
		row.Label = paramStr
	}
	return row, true
}

func parseField(norm, key string) float64 {
	for _, p := range strings.Split(norm, ",") {
		p = strings.TrimSpace(p)
		prefix := key + "="
		if !strings.HasPrefix(p, prefix) {
			continue
		}
		v, err := strconv.ParseFloat(p[len(prefix):], 64)
		if err == nil {
			return v
		}
	}
	return 0
}

func parseStringField(norm, key string) string {
	for _, p := range strings.Split(norm, ",") {
		p = strings.TrimSpace(p)
		prefix := key + "="
		if strings.HasPrefix(p, prefix) {
			return p[len(prefix):]
		}
	}
	return ""
}

func approxEqual(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-6
}

func extractDims(c eval.Correlation) (coh, con, flu, rel float64) {
	for _, d := range c.Dimensions {
		switch d.Name {
		case "coherence":
			coh = d.Spearman
		case "consistency":
			con = d.Spearman
		case "fluency":
			flu = d.Spearman
		case "relevance":
			rel = d.Spearman
		}
	}
	return
}

// dedupRows collapses duplicate (split, λ) rows to a single entry.
// The coarse λ ∈ {0.25, 0.5, 1.0, 2.0} sweep and the finer-grained
// b5k sweep around the optimum both contain a λ=0.5 run with the
// canonical embedder; deterministic scoring means the point
// estimates are identical, but rendering both produces a confusing
// duplicate row. Prefer the source with the longer file name (the
// b5k variant), which carries bootstrap CI in its underlying JSON.
func dedupRows(rows []Row) []Row {
	type key struct {
		split  string
		lambda float64
	}
	best := map[key]Row{}
	for _, r := range rows {
		k := key{r.Split, r.LeadBiasLambda}
		prev, ok := best[k]
		if !ok || len(r.Source) > len(prev.Source) {
			best[k] = r
		}
	}
	out := make([]Row, 0, len(best))
	for _, r := range best {
		out = append(out, r)
	}
	return out
}

// setCanonical marks the test-split row matching the dev-selected λ*.
func setCanonical(rows []Row, lambda float64) {
	for i := range rows {
		if rows[i].Split == "last50" && approxEqual(rows[i].LeadBiasLambda, lambda) {
			rows[i].IsCanonical = true
			return
		}
	}
}

// sortRows orders the table: dev rows first (ascending λ), then test
// rows (ascending λ), then full-set canonical (if any).
func sortRows(rows []Row) {
	splitRank := func(s string) int {
		switch s {
		case "first50":
			return 0
		case "last50":
			return 1
		case "all":
			return 2
		}
		return 3
	}
	sort.SliceStable(rows, func(i, j int) bool {
		ri, rj := splitRank(rows[i].Split), splitRank(rows[j].Split)
		if ri != rj {
			return ri < rj
		}
		return rows[i].LeadBiasLambda < rows[j].LeadBiasLambda
	})
}

func renderLatex(rows []Row) string {
	var b strings.Builder
	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintln(&b, `\caption{Ablation of LGS on SummEval (summary-level Spearman $\rho$). The metric is $\mathrm{score} = \mathrm{mean}_j\, \max_i\, w(i)\cdot\cos(c_j, s_i)$ with exponential lead-bias prior $w(i)=\exp(-\lambda \cdot i / n)$ on source-sentence position. The exponent $\lambda$ is selected on a held-out development split (50 articles) and re-evaluated on the test split (50 articles); the dev-selected canonical row is marked $\star$.}`)
	fmt.Fprintln(&b, `\label{tab:ablation_lgs}`)
	fmt.Fprintln(&b, `\begin{tabular}{lrrrr}`)
	fmt.Fprintln(&b, `\toprule`)
	fmt.Fprintln(&b, `Variant & Coh & Con & Flu & Rel \\`)
	fmt.Fprintln(&b, `\midrule`)
	prevSplit := ""
	for _, r := range rows {
		if prevSplit != "" && r.Split != prevSplit {
			fmt.Fprintln(&b, `\midrule`)
		}
		prevSplit = r.Split
		label := r.Label
		if r.IsCanonical {
			label += " $\\star$"
		}
		fmt.Fprintf(&b, "%s & %s & %s & %s & %s \\\\\n",
			label,
			fmtCell(r.Coh), fmtCell(r.Con), fmtCell(r.Flu), fmtCell(r.Rel))
	}
	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
}

func fmtCell(x float64) string {
	s := fmt.Sprintf("%.3f", x)
	if x >= 0 && x < 1 {
		return strings.TrimPrefix(s, "0")
	}
	if x > -1 && x < 0 {
		return "-" + strings.TrimPrefix(s[1:], "0")
	}
	return s
}

func writeFile(path, content string) error {
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
	_, err = io.WriteString(f, content)
	return err
}
