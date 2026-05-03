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

var (
	input  string
	output string
	level  string
	withCI bool
)

var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

var dimensionShort = map[string]string{
	"coherence":   "Coh",
	"consistency": "Con",
	"fluency":     "Flu",
	"relevance":   "Rel",
}

var metricOrder = []string{
	"bleu", "rouge", "chrf", "meteor", "smartstring",
	"embedscorer",
	"bertscore", "moverscore", "smartmodel",
	"bartscore", "gptscore", "unieval", "geval",
}

var metricDisplayName = map[string]string{
	"bleu":        "BLEU",
	"rouge":       "ROUGE-L",
	"chrf":        "ChrF",
	"meteor":      "METEOR",
	"smartstring": "SMART-String",
	"embedscorer": "EmbedScorer",
	"bertscore":   "BERTScore",
	"moverscore":  "MoverScore",
	"smartmodel":  "SMART-Model",
	"bartscore":   "BARTScore",
	"gptscore":    "GPTScore",
	"unieval":     "UniEval",
	"geval":       "G-Eval",
}

// dimensionalMetrics lists metrics that produce one report per dimension
// (rather than one report total). Each is collapsed into a single row using
// the matched diagonal correlations.
var dimensionalMetrics = []struct {
	prefix      string // e.g. "geval_" — files are geval_coherence.json, geval_consistency.json, ...
	displayName string
}{
	{"geval_", "G-Eval"},
	{"unieval_", "UniEval"},
}

func main() {
	flag.StringVar(&input, "input", "output", "directory containing metric JSON reports")
	flag.StringVar(&output, "output", "paper/tables/correlations.tex", "path to write LaTeX table (- for stdout)")
	flag.StringVar(&level, "level", "summary", "correlation level: summary|system")
	flag.BoolVar(&withCI, "ci", false, "include 95%% CI in each cell (requires bootstrap data)")
	flag.Parse()

	if level != "summary" && level != "system" {
		log.Fatalf("unknown level %q (available: summary, system)", level)
	}

	reports, err := loadReports(input)
	if err != nil {
		log.Fatal(err)
	}
	if len(reports) == 0 {
		log.Fatalf("no reports found in %s", input)
	}

	rows := buildRows(reports, level)
	table := renderLatex(rows, level, withCI)

	if err := write(output, table); err != nil {
		log.Fatal(err)
	}
}

func loadReports(dir string) ([]eval.Report, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, fmt.Errorf("glob: %w", err)
	}
	sort.Strings(matches)

	reports := make([]eval.Report, 0, len(matches))
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		var r eval.Report
		if err := json.Unmarshal(data, &r); err != nil {
			return nil, fmt.Errorf("decode %s: %w", path, err)
		}
		reports = append(reports, r)
	}
	return reports, nil
}

type Cell struct {
	Value float64
	CI    *eval.CI
}

type Row struct {
	Label string
	Cells map[string][3]Cell
}

func buildRows(reports []eval.Report, level string) []Row {
	byMetric := make(map[string]eval.Report, len(reports))
	for _, r := range reports {
		byMetric[r.Metric] = r
	}

	rows := make([]Row, 0, len(reports))
	seen := make(map[string]bool)

	// Collapse all dimensional metrics (G-Eval, UniEval).
	for _, dm := range dimensionalMetrics {
		if row, ok := collapseDimensional(byMetric, dm.prefix, dm.displayName, level); ok {
			rows = append(rows, row)
			for _, d := range dimensions {
				seen[dm.prefix+d] = true
			}
		}
	}

	// Regular metrics.
	for _, r := range reports {
		if seen[r.Metric] {
			continue
		}
		// Skip dimensional partials that didn't get fully collapsed (missing files).
		if isDimensionalPartial(r.Metric) {
			continue
		}
		rows = append(rows, Row{
			Label: displayName(r.Metric),
			Cells: cellsFromAllDimensions(r, level),
		})
	}

	sortRows(rows)
	return rows
}

// collapseDimensional builds a single row from per-dimension reports
// (e.g. unieval_coherence.json, unieval_consistency.json, ...) by taking
// the diagonal cell from each. Returns (row, true) if all 4 dimensions
// are present; otherwise (zero, false).
func collapseDimensional(byMetric map[string]eval.Report, prefix, displayName, level string) (Row, bool) {
	cells := make(map[string][3]Cell, len(dimensions))
	for _, dim := range dimensions {
		rep, ok := byMetric[prefix+dim]
		if !ok {
			return Row{}, false
		}
		corr := selectLevel(rep, level)
		found := false
		for _, d := range corr.Dimensions {
			if d.Name == dim {
				cells[dim] = [3]Cell{
					{Value: d.Spearman, CI: d.SpearmanCI},
					{Value: d.Pearson, CI: d.PearsonCI},
					{Value: d.KendallTau, CI: d.KendallTauCI},
				}
				found = true
				break
			}
		}
		if !found {
			return Row{}, false
		}
	}
	return Row{Label: displayName, Cells: cells}, true
}

// isDimensionalPartial returns true if the metric name matches a known
// dimensional prefix (geval_, unieval_) — used to skip incomplete sets.
func isDimensionalPartial(metric string) bool {
	for _, dm := range dimensionalMetrics {
		if strings.HasPrefix(metric, dm.prefix) {
			return true
		}
	}
	return false
}

func cellsFromAllDimensions(r eval.Report, level string) map[string][3]Cell {
	out := make(map[string][3]Cell, len(dimensions))
	corr := selectLevel(r, level)
	for _, d := range corr.Dimensions {
		out[d.Name] = [3]Cell{
			{Value: d.Spearman, CI: d.SpearmanCI},
			{Value: d.Pearson, CI: d.PearsonCI},
			{Value: d.KendallTau, CI: d.KendallTauCI},
		}
	}
	return out
}

func selectLevel(r eval.Report, level string) eval.Correlation {
	if level == "system" {
		return r.SystemLevel
	}
	return r.SummaryLevel
}

func displayName(metric string) string {
	if v, ok := metricDisplayName[metric]; ok {
		return v
	}
	return metric
}

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

func renderLatex(rows []Row, level string, withCI bool) string {
	var b strings.Builder

	colSpec := "l" + strings.Repeat("rrr", len(dimensions))
	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)
	levelLabel := "Summary-level"
	if level == "system" {
		levelLabel = "System-level"
	}
	fmt.Fprintf(&b, "\\caption{%s correlations with human judgment on SummEval. "+
		`$\rho$ = Spearman, $r$ = Pearson, $\tau$ = Kendall.}`+"\n", levelLabel)
	fmt.Fprintf(&b, "\\label{tab:correlations_%s}\n", level)
	fmt.Fprintf(&b, "\\begin{tabular}{%s}\n", colSpec)
	fmt.Fprintln(&b, `\toprule`)

	fmt.Fprint(&b, "Metric")
	for _, dim := range dimensions {
		fmt.Fprintf(&b, ` & \multicolumn{3}{c}{%s}`, dimensionShort[dim])
	}
	fmt.Fprintln(&b, ` \\`)

	for i := range dimensions {
		fmt.Fprintf(&b, `\cmidrule(lr){%d-%d} `, 2+3*i, 4+3*i)
	}
	fmt.Fprintln(&b)

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
			fmt.Fprintf(&b, ` & %s & %s & %s`,
				fmtCell(triple[0], withCI),
				fmtCell(triple[1], withCI),
				fmtCell(triple[2], withCI))
		}
		fmt.Fprintln(&b, ` \\`)
	}

	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
}

func fmtCell(c Cell, withCI bool) string {
	value := stripLeadingZero(c.Value)
	if !withCI || c.CI == nil {
		return value
	}
	return fmt.Sprintf(`%s {\scriptsize [%s, %s]}`,
		value,
		stripLeadingZero(c.CI.Low),
		stripLeadingZero(c.CI.High))
}

func stripLeadingZero(x float64) string {
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
