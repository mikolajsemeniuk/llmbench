// cmd/ablation reads BGS ablation reports from a directory (default
// `ablation/`) and renders a LaTeX table of the lead-bias λ sweep on
// the held-out development split, with the dev-selected λ* re-evaluated
// on the test split. The dev-selected canonical row is flagged with a ★.
//
// Output target matches the rest of the paper pipeline (default
// `paper/ablation.tex`). Rendering also writes a JSON sidecar so the
// webviewer's Ablation tab can render the same data without
// re-parsing LaTeX.
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
	outputJSON string
	lambdaStar float64
)

func main() {
	flag.StringVar(&inputDir, "input", "ablation", "directory containing ablation JSON reports")
	flag.StringVar(&outputTex, "output", "paper/ablation.tex", "path to write LaTeX table")
	flag.StringVar(&outputJSON, "json", "paper/ablation.json", "path to write JSON sidecar (for webviewer)")
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

	sortRows(rows)
	if lambdaStar >= 0 {
		setCanonical(rows, lambdaStar)
	}

	if err := writeFile(outputTex, renderLatex(rows)); err != nil {
		log.Fatalf("write tex: %v", err)
	}
	if outputJSON != "" {
		raw, err := json.MarshalIndent(rows, "", "  ")
		if err != nil {
			log.Fatalf("marshal json: %v", err)
		}
		if err := writeFile(outputJSON, string(raw)); err != nil {
			log.Fatalf("write json: %v", err)
		}
	}
	log.Printf("ablation: %d rows → %s", len(rows), outputTex)
}

// Row is one line of the ablation table. Exported because the JSON
// sidecar is rendered by the webviewer.
type Row struct {
	Label          string  `json:"label"`
	Source         string  `json:"source"`
	LeadBiasLambda float64 `json:"lead_bias_lambda"`
	Split          string  `json:"split"` // "first50" | "last50" | "all"
	IsCanonical    bool    `json:"is_canonical"`
	Coh            float64 `json:"coh"`
	Con            float64 `json:"con"`
	Flu            float64 `json:"flu"`
	Rel            float64 `json:"rel"`
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
	if r.Metric != "bgs" {
		return Row{}, false
	}

	row := Row{Source: filepath.Base(path)}
	row.Coh, row.Con, row.Flu, row.Rel = extractDims(r.SummaryLevel)
	row.LeadBiasLambda = parseField(r.Norm, "lead_lambda")
	row.Split = parseStringField(r.Norm, "split")

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
	fmt.Fprintln(&b, `\caption{Ablation of BGS on SummEval (summary-level Spearman $\rho$). The metric is $\mathrm{score} = \mathrm{mean}_j\, \max_i\, w(i)\cdot\cos(c_j, s_i)$ with exponential lead-bias prior $w(i)=\exp(-\lambda \cdot i / n)$ on source-sentence position. The exponent $\lambda$ is selected on a held-out development split (50 articles) and re-evaluated on the test split (50 articles); the dev-selected canonical row is marked $\star$.}`)
	fmt.Fprintln(&b, `\label{tab:ablation_bgs}`)
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
