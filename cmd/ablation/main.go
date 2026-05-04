// cmd/ablation reads BGS ablation reports from a directory (default
// `ablation/`) and renders a LaTeX table summarising the β / salience
// sweep. The canonical configuration (β=2, salience-frac=0.30) is
// flagged with a ★. The recall-only baseline (BGS run with the
// `-recall-only` flag, identified by `recall_only=true` in the
// report's Norm field) appears as the bottom row, separated by a
// midrule, to anchor the rest of the sweep.
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
)

func main() {
	flag.StringVar(&inputDir, "input", "ablation", "directory containing ablation JSON reports")
	flag.StringVar(&outputTex, "output", "paper/ablation.tex", "path to write LaTeX table")
	flag.StringVar(&outputJSON, "json", "paper/ablation.json", "path to write JSON sidecar (for webviewer)")
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
	Label        string  `json:"label"`
	Source       string  `json:"source"`
	Beta         float64 `json:"beta"`
	SalienceFrac float64 `json:"salience_frac"`
	IsCanonical  bool    `json:"is_canonical"`
	IsRecallOnly bool    `json:"is_recall_only"`
	Coh          float64 `json:"coh"`
	Con          float64 `json:"con"`
	Flu          float64 `json:"flu"`
	Rel          float64 `json:"rel"`
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

	row := Row{Source: filepath.Base(path)}
	row.Coh, row.Con, row.Flu, row.Rel = extractDims(r.SummaryLevel)

	if r.Metric != "bgs" {
		return Row{}, false
	}
	// Recall-only sentinel (BGS run with -recall-only flag) is the
	// reproducible "no precision side" ablation row. Identified by
	// the norm field "recall_only=true".
	if strings.Contains(r.Norm, "recall_only=true") {
		row.IsRecallOnly = true
		row.Label = "Recall only (no precision side)"
		return row, true
	}
	row.Beta = parseField(r.Norm, "beta")
	row.SalienceFrac = parseField(r.Norm, "salience")
	row.Label = fmt.Sprintf("$\\beta=%g$, frac=%.2f", row.Beta, row.SalienceFrac)
	row.IsCanonical = approxEqual(row.Beta, 2.0) && approxEqual(row.SalienceFrac, 0.30)
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

func sortRows(rows []Row) {
	sort.SliceStable(rows, func(i, j int) bool {
		// Recall-only baseline goes last.
		if rows[i].IsRecallOnly != rows[j].IsRecallOnly {
			return !rows[i].IsRecallOnly
		}
		// β ascending, then salience-frac ascending.
		if rows[i].Beta != rows[j].Beta {
			return rows[i].Beta < rows[j].Beta
		}
		return rows[i].SalienceFrac < rows[j].SalienceFrac
	})
}

func renderLatex(rows []Row) string {
	var b strings.Builder
	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintln(&b, `\caption{Ablation of BGS on SummEval (summary-level Spearman $\rho$). β controls the F$_\beta$ recall vs.\ precision weighting; salience-frac is the top-$k$\% of source sentences (by degree centrality) used in the precision side. Canonical configuration ($\beta=2$, frac $=0.30$) marked $\star$.}`)
	fmt.Fprintln(&b, `\label{tab:ablation_bgs}`)
	fmt.Fprintln(&b, `\begin{tabular}{lrrrr}`)
	fmt.Fprintln(&b, `\toprule`)
	fmt.Fprintln(&b, `Variant & Coh & Con & Flu & Rel \\`)
	fmt.Fprintln(&b, `\midrule`)
	for _, r := range rows {
		label := r.Label
		if r.IsCanonical {
			label += " $\\star$"
		}
		if r.IsRecallOnly {
			fmt.Fprintln(&b, `\midrule`)
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
