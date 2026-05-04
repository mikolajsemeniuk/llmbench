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
	input        string
	output       string
	level        string
	withCI       bool
	coefficients string
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
	"bgs",
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
	"bgs":         "BGS",
}

var dimensionalMetrics = []struct {
	prefix      string
	displayName string
}{
	{"geval_", "G-Eval"},
	{"unieval_", "UniEval"},
}

// coefficient describes one correlation column: its display symbol and how
// to extract the value + CI from a CorrelationDimension.
type coefficient struct {
	key    string
	symbol string
	value  func(d eval.Dimension) float64
	ci     func(d eval.Dimension) *eval.CI
}

var allCoefficients = map[string]coefficient{
	"spearman": {
		key: "spearman", symbol: `$\rho$`,
		value: func(d eval.Dimension) float64 { return d.Spearman },
		ci:    func(d eval.Dimension) *eval.CI { return d.SpearmanCI },
	},
	"pearson": {
		key: "pearson", symbol: `$r$`,
		value: func(d eval.Dimension) float64 { return d.Pearson },
		ci:    func(d eval.Dimension) *eval.CI { return d.PearsonCI },
	},
	"kendall": {
		key: "kendall", symbol: `$\tau$`,
		value: func(d eval.Dimension) float64 { return d.KendallTau },
		ci:    func(d eval.Dimension) *eval.CI { return d.KendallTauCI },
	},
}

func main() {
	flag.StringVar(&input, "input", "output", "directory containing metric JSON reports")
	flag.StringVar(&output, "output", "paper/correlations.tex", "path to write LaTeX table (- for stdout)")
	flag.StringVar(&level, "level", "summary", "correlation level: summary|system")
	flag.BoolVar(&withCI, "ci", false, "include 95%% CI in each cell (requires bootstrap data)")
	flag.StringVar(&coefficients, "coeffs", "spearman,kendall",
		"comma-separated coefficients to display: spearman, pearson, kendall")
	flag.Parse()

	if level != "summary" && level != "system" {
		log.Fatalf("unknown level %q (available: summary, system)", level)
	}

	coeffs, err := parseCoefficients(coefficients)
	if err != nil {
		log.Fatal(err)
	}

	reports, err := loadReports(input)
	if err != nil {
		log.Fatal(err)
	}
	if len(reports) == 0 {
		log.Fatalf("no reports found in %s", input)
	}

	rows := buildRows(reports, level, coeffs)
	table := renderLatex(rows, level, coeffs, withCI)

	if err := write(output, table); err != nil {
		log.Fatal(err)
	}
}

func parseCoefficients(spec string) ([]coefficient, error) {
	parts := strings.Split(spec, ",")
	out := make([]coefficient, 0, len(parts))
	seen := make(map[string]bool, len(parts))
	for _, p := range parts {
		key := strings.TrimSpace(strings.ToLower(p))
		if key == "" {
			continue
		}
		c, ok := allCoefficients[key]
		if !ok {
			return nil, fmt.Errorf("unknown coefficient %q (available: spearman, pearson, kendall)", key)
		}
		if seen[key] {
			return nil, fmt.Errorf("coefficient %q specified twice", key)
		}
		seen[key] = true
		out = append(out, c)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one coefficient required")
	}
	return out, nil
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
	// Cells maps dimension -> coefficient values (in same order as the
	// coeffs slice provided at construction time).
	Cells map[string][]Cell
}

func buildRows(reports []eval.Report, level string, coeffs []coefficient) []Row {
	byMetric := make(map[string]eval.Report, len(reports))
	for _, r := range reports {
		byMetric[r.Metric] = r
	}

	rows := make([]Row, 0, len(reports))
	seen := make(map[string]bool)

	for _, dm := range dimensionalMetrics {
		if row, ok := collapseDimensional(byMetric, dm.prefix, dm.displayName, level, coeffs); ok {
			rows = append(rows, row)
			for _, d := range dimensions {
				seen[dm.prefix+d] = true
			}
		}
	}

	for _, r := range reports {
		if seen[r.Metric] {
			continue
		}
		if isDimensionalPartial(r.Metric) {
			continue
		}
		rows = append(rows, Row{
			Label: displayName(r.Metric),
			Cells: cellsFromAllDimensions(r, level, coeffs),
		})
	}

	sortRows(rows)
	return rows
}

func collapseDimensional(byMetric map[string]eval.Report, prefix, displayName, level string, coeffs []coefficient) (Row, bool) {
	cells := make(map[string][]Cell, len(dimensions))
	for _, dim := range dimensions {
		rep, ok := byMetric[prefix+dim]
		if !ok {
			return Row{}, false
		}
		corr := selectLevel(rep, level)
		var matched *eval.Dimension
		for i := range corr.Dimensions {
			if corr.Dimensions[i].Name == dim {
				matched = &corr.Dimensions[i]
				break
			}
		}
		if matched == nil {
			return Row{}, false
		}
		cells[dim] = makeCells(*matched, coeffs)
	}
	return Row{Label: displayName, Cells: cells}, true
}

func isDimensionalPartial(metric string) bool {
	for _, dm := range dimensionalMetrics {
		if strings.HasPrefix(metric, dm.prefix) {
			return true
		}
	}
	return false
}

func cellsFromAllDimensions(r eval.Report, level string, coeffs []coefficient) map[string][]Cell {
	out := make(map[string][]Cell, len(dimensions))
	corr := selectLevel(r, level)
	for _, d := range corr.Dimensions {
		out[d.Name] = makeCells(d, coeffs)
	}
	return out
}

func makeCells(d eval.Dimension, coeffs []coefficient) []Cell {
	cells := make([]Cell, len(coeffs))
	for i, c := range coeffs {
		cells[i] = Cell{Value: c.value(d), CI: c.ci(d)}
	}
	return cells
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

func renderLatex(rows []Row, level string, coeffs []coefficient, withCI bool) string {
	var b strings.Builder

	nCoeff := len(coeffs)
	colSpec := "l" + strings.Repeat(strings.Repeat("r", nCoeff), len(dimensions))

	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)

	levelLabel := "Summary-level"
	if level == "system" {
		levelLabel = "System-level"
	}

	caption := fmt.Sprintf("%s correlations with human judgment on SummEval. %s.",
		levelLabel, coeffLegend(coeffs))
	fmt.Fprintf(&b, "\\caption{%s}\n", caption)
	fmt.Fprintf(&b, "\\label{tab:correlations_%s}\n", level)
	fmt.Fprintf(&b, "\\begin{tabular}{%s}\n", colSpec)
	fmt.Fprintln(&b, `\toprule`)

	fmt.Fprint(&b, "Metric")
	for _, dim := range dimensions {
		fmt.Fprintf(&b, ` & \multicolumn{%d}{c}{%s}`, nCoeff, dimensionShort[dim])
	}
	fmt.Fprintln(&b, ` \\`)

	for i := range dimensions {
		from := 2 + nCoeff*i
		to := from + nCoeff - 1
		fmt.Fprintf(&b, `\cmidrule(lr){%d-%d} `, from, to)
	}
	fmt.Fprintln(&b)

	fmt.Fprint(&b, " ")
	for range dimensions {
		for _, c := range coeffs {
			fmt.Fprintf(&b, ` & %s`, c.symbol)
		}
	}
	fmt.Fprintln(&b, ` \\`)
	fmt.Fprintln(&b, `\midrule`)

	for _, r := range rows {
		fmt.Fprint(&b, r.Label)
		for _, dim := range dimensions {
			cells, ok := r.Cells[dim]
			if !ok {
				for range coeffs {
					fmt.Fprint(&b, ` & --`)
				}
				continue
			}
			for _, cell := range cells {
				fmt.Fprintf(&b, ` & %s`, fmtCell(cell, withCI))
			}
		}
		fmt.Fprintln(&b, ` \\`)
	}

	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
}

func coeffLegend(coeffs []coefficient) string {
	parts := make([]string, 0, len(coeffs))
	names := map[string]string{
		"spearman": "Spearman",
		"pearson":  "Pearson",
		"kendall":  "Kendall",
	}
	for _, c := range coeffs {
		parts = append(parts, fmt.Sprintf("%s = %s", c.symbol, names[c.key]))
	}
	return strings.Join(parts, ", ")
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
