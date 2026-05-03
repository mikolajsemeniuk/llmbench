// cmd/compare runs paired bootstrap to test whether a target metric
// correlates with human ratings significantly differently than each of N
// baseline metrics, across the four SummEval dimensions.
//
// Usage:
//
//	go run ./cmd/compare -metric mymetric -baselines unieval,geval
//
// Outputs both a console summary (with arrows for win/tie/loss) and a
// LaTeX table (Δρ with 95% CI and p-value per cell).
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

// ── Configuration ──────────────────────────────────────────────────────

var (
	inputDir  string
	output    string
	target    string
	baselines string
	bootstrap int
	level     string
	seed      uint64
)

var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

var dimensionShort = map[string]string{
	"coherence":   "Coh",
	"consistency": "Con",
	"fluency":     "Flu",
	"relevance":   "Rel",
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

// ── Domain types ───────────────────────────────────────────────────────

// baselineEntry is one baseline + its scores aligned to the canonical
// sample order. scoresByDim differs per dimension only for UniEval/G-Eval
// (which have one report file per dimension); for other metrics, every
// dimension shares the same score slice.
type baselineEntry struct {
	key         string
	scoresByDim map[string][]float64
}

// comparisonCell is one (baseline, dimension) result.
type comparisonCell struct {
	baseline  string
	dimension string
	comp      eval.PairedComparison
	targetRho float64
	baseRho   float64
}

// ── Main ───────────────────────────────────────────────────────────────

func main() {
	flag.StringVar(&inputDir, "input", "output", "directory containing metric JSON reports")
	flag.StringVar(&output, "output", "paper/tables/comparisons.tex",
		"path to write LaTeX table (- for stdout, empty to skip)")
	flag.StringVar(&target, "metric", "",
		"target metric name (file basename without .json, e.g. 'mymetric')")
	flag.StringVar(&baselines, "baselines", "unieval,geval",
		"comma-separated list of baselines to compare against")
	flag.IntVar(&bootstrap, "bootstrap", 5000,
		"number of paired bootstrap resamples (5000+ recommended)")
	flag.StringVar(&level, "level", "summary", "correlation level: summary|system")
	flag.Uint64Var(&seed, "seed", 42, "random seed for reproducibility")
	flag.Parse()

	if target == "" {
		log.Fatal("--metric is required (e.g. -metric mymetric)")
	}
	if level != "summary" && level != "system" {
		log.Fatalf("unknown level %q (must be summary or system)", level)
	}
	if level == "system" {
		log.Println("WARNING: system-level paired bootstrap with N=16 systems gives very wide CI; results are exploratory.")
	}

	baseList := splitCSV(baselines)
	if len(baseList) == 0 {
		log.Fatal("no baselines specified")
	}

	samples, err := eval.NewDataset(eval.SummevalDataset, eval.DefaultDatasetPath, 0)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	targetScores, err := loadScores(samples, inputDir, target)
	if err != nil {
		log.Fatalf("load target %q: %v", target, err)
	}

	baselineEntries := make([]baselineEntry, 0, len(baseList))
	for _, name := range baseList {
		bl, err := loadBaseline(samples, inputDir, name)
		if err != nil {
			log.Fatalf("load baseline %q: %v", name, err)
		}
		baselineEntries = append(baselineEntries, baselineEntry{key: name, scoresByDim: bl})
	}

	cells := make([]comparisonCell, 0, len(baselineEntries)*len(dimensions))
	for _, b := range baselineEntries {
		for _, dim := range dimensions {
			human := humanScores(samples, dim)
			baseScores := b.scoresByDim[dim]

			fn := eval.Spearman
			var comp eval.PairedComparison
			var targetRho, baseRho float64

			if level == "summary" {
				comp = eval.PairedBootstrap(samples, targetScores, baseScores, human,
					fn, bootstrap, seed)
				targetRho = fn(targetScores, human)
				baseRho = fn(baseScores, human)
			} else {
				comp, targetRho, baseRho = systemLevelPaired(samples, targetScores,
					baseScores, human, fn, bootstrap, seed)
			}

			cells = append(cells, comparisonCell{
				baseline:  b.key,
				dimension: dim,
				comp:      comp,
				targetRho: targetRho,
				baseRho:   baseRho,
			})
		}
	}

	fmt.Println(renderConsole(target, cells))

	if output != "" {
		latex := renderLatex(target, baselineEntries, cells, level)
		if err := writeFile(output, latex); err != nil {
			log.Fatal(err)
		}
		if output != "-" {
			fmt.Fprintf(os.Stderr, "\nLaTeX table written to %s\n", output)
		}
	}
}

// ── Loading ────────────────────────────────────────────────────────────

// loadScores reads <dir>/<name>.json and aligns scores to the given
// samples by SampleID. Returns the value slice in the canonical order.
func loadScores(samples []eval.Sample, dir, name string) ([]float64, error) {
	path := filepath.Join(dir, name+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var r eval.Report
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}
	if len(r.Scores) != len(samples) {
		return nil, fmt.Errorf("expected %d scores in %s, got %d",
			len(samples), path, len(r.Scores))
	}

	byID := make(map[string]float64, len(r.Scores))
	for _, s := range r.Scores {
		byID[s.SampleID] = s.Value
	}

	out := make([]float64, len(samples))
	for i, s := range samples {
		v, ok := byID[s.ID]
		if !ok {
			return nil, fmt.Errorf("missing score for sample %s in %s", s.ID, path)
		}
		out[i] = v
	}
	return out, nil
}

// loadBaseline tries dimensional layout first (e.g. unieval_coherence.json,
// unieval_consistency.json, ...). If all four exist, returns one slice per
// dimension. Otherwise treats <name>.json as a single-file metric and
// returns the same slice for every dimension.
func loadBaseline(samples []eval.Sample, dir, name string) (map[string][]float64, error) {
	out := make(map[string][]float64, len(dimensions))

	dimensional := true
	for _, dim := range dimensions {
		path := filepath.Join(dir, name+"_"+dim+".json")
		if _, err := os.Stat(path); err != nil {
			dimensional = false
			break
		}
	}

	if dimensional {
		for _, dim := range dimensions {
			scores, err := loadScores(samples, dir, name+"_"+dim)
			if err != nil {
				return nil, err
			}
			out[dim] = scores
		}
		return out, nil
	}

	scores, err := loadScores(samples, dir, name)
	if err != nil {
		return nil, err
	}
	for _, dim := range dimensions {
		out[dim] = scores
	}
	return out, nil
}

func humanScores(samples []eval.Sample, dim string) []float64 {
	get := map[string]func(eval.Sample) float64{
		"coherence":   func(s eval.Sample) float64 { return s.Coherence },
		"consistency": func(s eval.Sample) float64 { return s.Consistency },
		"fluency":     func(s eval.Sample) float64 { return s.Fluency },
		"relevance":   func(s eval.Sample) float64 { return s.Relevance },
	}
	out := make([]float64, len(samples))
	for i, s := range samples {
		out[i] = get[dim](s)
	}
	return out
}

// ── System-level paired bootstrap ──────────────────────────────────────

// systemLevelPaired aggregates per system, then resamples systems with
// replacement instead of documents.
//
// Trick: we build a synthetic sample list with one entry per system,
// using SystemID also as DocumentID. PairedBootstrap's document-level
// resampling then becomes system-level resampling — exactly what we want.
func systemLevelPaired(samples []eval.Sample, scoresA, scoresB, human []float64,
	fn eval.CorrelationFunc, n int, seed uint64) (eval.PairedComparison, float64, float64) {

	type sysAcc struct {
		a, b, h float64
		count   int
	}
	bySystem := make(map[int]*sysAcc)
	for i, s := range samples {
		acc, ok := bySystem[s.SystemID]
		if !ok {
			acc = &sysAcc{}
			bySystem[s.SystemID] = acc
		}
		acc.a += scoresA[i]
		acc.b += scoresB[i]
		acc.h += human[i]
		acc.count++
	}

	ids := make([]int, 0, len(bySystem))
	for id := range bySystem {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	a := make([]float64, len(ids))
	b := make([]float64, len(ids))
	h := make([]float64, len(ids))
	for i, id := range ids {
		acc := bySystem[id]
		a[i] = acc.a / float64(acc.count)
		b[i] = acc.b / float64(acc.count)
		h[i] = acc.h / float64(acc.count)
	}

	targetRho := fn(a, h)
	baseRho := fn(b, h)

	syn := make([]eval.Sample, len(ids))
	for i, id := range ids {
		syn[i] = eval.Sample{
			ID:         fmt.Sprintf("sys_%d", id),
			DocumentID: fmt.Sprintf("sys_%d", id),
			SystemID:   id,
		}
	}
	comp := eval.PairedBootstrap(syn, a, b, h, fn, n, seed)
	return comp, targetRho, baseRho
}

// ── Output rendering ───────────────────────────────────────────────────

func renderConsole(target string, cells []comparisonCell) string {
	var b strings.Builder

	fmt.Fprintf(&b, "Paired bootstrap: %s vs baselines (Spearman)\n\n", display(target))
	fmt.Fprintf(&b, "%-15s %-12s %8s %8s %8s %20s %8s\n",
		"Baseline", "Dimension", "ours", "base", "Δ", "95% CI", "p")
	fmt.Fprintln(&b, strings.Repeat("─", 88))

	wins, ties, losses := 0, 0, 0
	for _, c := range cells {
		ci := c.comp.DeltaCI
		ciStr := fmt.Sprintf("[%+.3f, %+.3f]", ci.Low, ci.High)
		sig := " "
		switch {
		case ci.Low > 0:
			sig = "↑"
			wins++
		case ci.High < 0:
			sig = "↓"
			losses++
		default:
			ties++
		}
		fmt.Fprintf(&b, "%-15s %-12s %+.3f   %+.3f   %+.3f   %s   %.4f %s\n",
			display(c.baseline), c.dimension,
			c.targetRho, c.baseRho, c.comp.DeltaMean,
			ciStr, c.comp.PValue, sig)
	}

	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Summary: wins=%d, ties=%d, losses=%d (across %d comparisons)\n",
		wins, ties, losses, len(cells))
	fmt.Fprintln(&b, "↑ = ours significantly higher (95% CI excludes 0)")
	fmt.Fprintln(&b, "↓ = ours significantly lower")
	fmt.Fprintln(&b, "  = no significant difference")
	return b.String()
}

func renderLatex(target string, baselines []baselineEntry, cells []comparisonCell, level string) string {
	var b strings.Builder

	colSpec := "l" + strings.Repeat("c", len(dimensions))

	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)

	levelLabel := "summary-level"
	if level == "system" {
		levelLabel = "system-level"
	}
	fmt.Fprintf(&b,
		"\\caption{Paired bootstrap comparison of %s against baselines on %s "+
			"Spearman correlations. Each cell reports $\\Delta\\rho$ "+
			"(ours minus baseline) with 95\\%% CI in brackets and $p$-value. "+
			"Bold marks significant improvements ($p<0.05$, CI excludes 0).}\n",
		display(target), levelLabel)
	fmt.Fprintf(&b, "\\label{tab:compare_%s_%s}\n", target, level)
	fmt.Fprintf(&b, "\\begin{tabular}{%s}\n", colSpec)
	fmt.Fprintln(&b, `\toprule`)

	fmt.Fprint(&b, "Baseline")
	for _, dim := range dimensions {
		fmt.Fprintf(&b, ` & %s`, dimensionShort[dim])
	}
	fmt.Fprintln(&b, ` \\`)
	fmt.Fprintln(&b, `\midrule`)

	byBase := make(map[string]map[string]comparisonCell, len(baselines))
	for _, c := range cells {
		if byBase[c.baseline] == nil {
			byBase[c.baseline] = make(map[string]comparisonCell, len(dimensions))
		}
		byBase[c.baseline][c.dimension] = c
	}

	for _, base := range baselines {
		fmt.Fprintf(&b, "%s", display(base.key))
		for _, dim := range dimensions {
			c := byBase[base.key][dim]
			fmt.Fprintf(&b, " & %s", fmtLatexCell(c.comp))
		}
		fmt.Fprintln(&b, ` \\`)
	}

	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
}

func fmtLatexCell(c eval.PairedComparison) string {
	delta := stripLeadingZero(c.DeltaMean)
	ci := fmt.Sprintf(`[%s, %s]`,
		stripLeadingZero(c.DeltaCI.Low),
		stripLeadingZero(c.DeltaCI.High))
	pStr := fmt.Sprintf("%.3f", c.PValue)
	if c.PValue < 0.001 {
		pStr = "<.001"
	}

	body := fmt.Sprintf("%s {\\scriptsize %s, $p$=%s}", delta, ci, pStr)
	if c.DeltaCI.Low > 0 {
		body = `\textbf{` + body + `}`
	}
	return body
}

// ── Helpers ────────────────────────────────────────────────────────────

func display(metric string) string {
	if v, ok := metricDisplayName[metric]; ok {
		return v
	}
	return metric
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func stripLeadingZero(x float64) string {
	s := fmt.Sprintf("%+.3f", x)
	// "+0.123" → "+.123"; "-0.123" → "-.123"
	if len(s) >= 3 && s[1] == '0' && s[2] == '.' {
		return s[:1] + s[2:]
	}
	return s
}

func writeFile(path, content string) error {
	if path == "-" {
		_, err := io.WriteString(os.Stdout, content)
		return err
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("mkdir %s: %w", dir, err)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.WriteString(f, content)
	return err
}
