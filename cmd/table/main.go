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

	llmbench "github.com/mikolajsemeniuk/llmbench/pkg"
)

var (
	input  string
	output string
)

// dimensions is the canonical ordering in the paper table (columns).
var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

// dimensionShort maps full dimension names to 3-letter headers.
var dimensionShort = map[string]string{
	"coherence":   "Coh",
	"consistency": "Con",
	"fluency":     "Flu",
	"relevance":   "Rel",
}

// metricOrder defines how metric rows are grouped and ordered in the table.
// Metrics not listed appear alphabetically after the listed ones.
var metricOrder = []string{
	"bleu", "rouge", "chrf", "meteor", "smartstring",
	"bertscore", "moverscore", "smartmodel",
	"bartscore", "gptscore", "unieval", "geval",
}

// metricDisplayName maps the internal metric key to the name shown in the table.
var metricDisplayName = map[string]string{
	"bleu":        "BLEU",
	"rouge":       "ROUGE-L",
	"chrf":        "ChrF",
	"meteor":      "METEOR",
	"smartstring": "SMART-String",
	"bertscore":   "BERTScore",
	"moverscore":  "MoverScore",
	"smartmodel":  "SMART-Model",
	"bartscore":   "BARTScore",
	"gptscore":    "GPTScore",
	"unieval":     "UniEval",
	"geval":       "G-Eval",
}

func main() {
	flag.StringVar(&input, "input", "output", "directory containing metric JSON reports")
	flag.StringVar(&output, "output", "paper/tables/correlations.tex", "path to write LaTeX table (- for stdout)")
	flag.Parse()

	reports, err := loadReports(input)
	if err != nil {
		log.Fatal(err)
	}
	if len(reports) == 0 {
		log.Fatalf("no reports found in %s", input)
	}

	rows := buildRows(reports)
	table := renderLatex(rows)

	if err := write(output, table); err != nil {
		log.Fatal(err)
	}
}

func loadReports(dir string) ([]llmbench.Report, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, fmt.Errorf("glob: %w", err)
	}
	sort.Strings(matches)

	reports := make([]llmbench.Report, 0, len(matches))
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		var r llmbench.Report
		if err := json.Unmarshal(data, &r); err != nil {
			return nil, fmt.Errorf("decode %s: %w", path, err)
		}
		reports = append(reports, r)
	}
	return reports, nil
}

// Row is one line in the LaTeX table: a metric label + 4 (spearman, pearson, kendall) triples.
type Row struct {
	Label string
	// cells[dimension] = (spearman, pearson, kendall)
	Cells map[string][3]float64
}

// buildRows converts raw reports to table rows, handling the G-Eval
// per-dimension collapsing: geval_coherence.json + geval_consistency.json +
// geval_fluency.json + geval_relevance.json -> single row "G-Eval" using the
// matched diagonal correlations.
func buildRows(reports []llmbench.Report) []Row {
	// Index reports by metric name for fast lookup.
	byMetric := make(map[string]llmbench.Report, len(reports))
	for _, r := range reports {
		byMetric[r.Metric] = r
	}

	rows := make([]Row, 0, len(reports))
	seen := make(map[string]bool)

	// G-Eval per-dimension collapse.
	if gevalRow, ok := collapseGEval(byMetric); ok {
		rows = append(rows, gevalRow)
		for _, d := range dimensions {
			seen["geval_"+d] = true
		}
	}

	// Regular metrics (single report -> single row).
	for _, r := range reports {
		if seen[r.Metric] {
			continue
		}
		if strings.HasPrefix(r.Metric, "geval_") {
			continue // collapsed above
		}
		rows = append(rows, Row{
			Label: displayName(r.Metric),
			Cells: cellsFromAllDimensions(r),
		})
	}

	sortRows(rows)
	return rows
}

// collapseGEval builds a single "G-Eval" row by taking the diagonal cell
// from each per-dimension report. Returns (row, true) if all 4 dimensions
// are present; otherwise (zero, false).
func collapseGEval(byMetric map[string]llmbench.Report) (Row, bool) {
	cells := make(map[string][3]float64, len(dimensions))
	for _, dim := range dimensions {
		rep, ok := byMetric["geval_"+dim]
		if !ok {
			return Row{}, false
		}
		// Find the matching dimension in this report's correlations (the diagonal).
		for _, d := range rep.Correlations.Dimensions {
			if d.Name == dim {
				cells[dim] = [3]float64{d.Spearman, d.Pearson, d.KendallTau}
				break
			}
		}
		if _, ok := cells[dim]; !ok {
			return Row{}, false
		}
	}
	return Row{Label: "G-Eval", Cells: cells}, true
}

// cellsFromAllDimensions reads all four dimension correlations from a
// single-run report (used for dimension-agnostic metrics like BLEU).
func cellsFromAllDimensions(r llmbench.Report) map[string][3]float64 {
	out := make(map[string][3]float64, len(dimensions))
	for _, d := range r.Correlations.Dimensions {
		out[d.Name] = [3]float64{d.Spearman, d.Pearson, d.KendallTau}
	}
	return out
}

func displayName(metric string) string {
	if v, ok := metricDisplayName[metric]; ok {
		return v
	}
	return metric
}

// sortRows orders rows according to metricOrder; unknown metrics sort alphabetically at the end.
func sortRows(rows []Row) {
	rank := make(map[string]int, len(metricOrder))
	for i, m := range metricOrder {
		rank[metricDisplayName[m]] = i
	}
	sort.SliceStable(rows, func(i, j int) bool {
		ri, oki := rank[rows[i].Label]
		rj, okj := rank[rows[j].Label]
		switch {
		case oki && okj:
			return ri < rj
		case oki:
			return true
		case okj:
			return false
		default:
			return rows[i].Label < rows[j].Label
		}
	})
}

// renderLatex builds the full booktabs-style LaTeX table.
func renderLatex(rows []Row) string {
	var b strings.Builder

	colSpec := "l" + strings.Repeat("rrr", len(dimensions))
	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintln(&b, `\caption{Summary-level correlations with human judgment on SummEval. $\rho$ = Spearman, $r$ = Pearson, $\tau$ = Kendall.}`)
	fmt.Fprintln(&b, `\label{tab:correlations}`)
	fmt.Fprintf(&b, "\\begin{tabular}{%s}\n", colSpec)
	fmt.Fprintln(&b, `\toprule`)

	// Header row: dimension groups.
	fmt.Fprint(&b, "Metric")
	for _, dim := range dimensions {
		fmt.Fprintf(&b, ` & \multicolumn{3}{c}{%s}`, dimensionShort[dim])
	}
	fmt.Fprintln(&b, ` \\`)

	// cmidrules under each group.
	for i := range dimensions {
		fmt.Fprintf(&b, `\cmidrule(lr){%d-%d} `, 2+3*i, 4+3*i)
	}
	fmt.Fprintln(&b)

	// Sub-header with coefficient symbols.
	fmt.Fprint(&b, " ")
	for range dimensions {
		fmt.Fprint(&b, ` & $\rho$ & $r$ & $\tau$`)
	}
	fmt.Fprintln(&b, ` \\`)
	fmt.Fprintln(&b, `\midrule`)

	for _, r := range rows {
		fmt.Fprint(&b, r.Label)
		for _, dim := range dimensions {
			triple, ok := r.Cells[dim]
			if !ok {
				fmt.Fprint(&b, ` & -- & -- & --`)
				continue
			}
			fmt.Fprintf(&b, ` & %s & %s & %s`, fmtCell(triple[0]), fmtCell(triple[1]), fmtCell(triple[2]))
		}
		fmt.Fprintln(&b, ` \\`)
	}

	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
}

// fmtCell formats a correlation value for display.
// Values in [0, 1) are shown without the leading zero (.478 not 0.478)
// to save column width — common convention in ML papers.
func fmtCell(x float64) string {
	s := fmt.Sprintf("%.3f", x)
	if x >= 0 && x < 1 {
		s = strings.TrimPrefix(s, "0")
	} else if x > -1 && x < 0 {
		s = "-" + strings.TrimPrefix(s[1:], "0")
	}
	return s
}

func write(path, content string) error {
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
