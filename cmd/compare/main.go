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

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
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
	fdrAlpha  float64
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
	"lgs":         "LGS",
	"nli":         "NLI",
	"composite":   "Composite (NLI+LGS+PPL)",
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

	// qValue is the Benjamini-Hochberg adjusted p-value over the whole
	// table. Every (baseline, dimension) pair is one test of the same
	// hypothesis family -- "does the target metric correlate
	// differently from this baseline?" -- so reading the raw p-values
	// at 0.05 would inflate the family-wise error rate across the
	// len(baselines) x 4 cells.
	qValue float64
}

func main() {
	flag.StringVar(&inputDir, "input", "output", "directory containing metric JSON reports")
	flag.StringVar(&output, "output", "paper/comparisons.gen.tex",
		"path to write LaTeX table (- for stdout, empty to skip)")
	flag.StringVar(&target, "metric", "",
		"target metric name (file basename without .json, e.g. 'mymetric')")
	flag.StringVar(&baselines, "baselines", "unieval,geval",
		"comma-separated list of baselines to compare against")
	flag.IntVar(&bootstrap, "bootstrap", 5000,
		"number of paired bootstrap resamples (5000+ recommended)")
	flag.StringVar(&level, "level", "summary", "correlation level: summary|system")
	flag.Uint64Var(&seed, "seed", 42, "random seed for reproducibility")
	flag.Float64Var(&fdrAlpha, "fdr-alpha", 0.05,
		"Benjamini-Hochberg false-discovery rate for the family of (baseline, dimension) tests")
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

	samples, err := eval.NewDataset(dataset.Summeval, dataset.SummevalDefaultPath, 0)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	// loadBaseline also handles the common flat case (same score reused
	// for every dimension), so it works for both a flat target like lgs
	// and a dimensional one like unieval or composite.
	targetByDim, err := loadBaseline(samples, inputDir, target)
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
			targetScores := targetByDim[dim]
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

	applyFDR(cells)

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

// ── Multiple-comparison correction ─────────────────────────────────────

// applyFDR fills in each cell's qValue using the Benjamini-Hochberg
// step-up procedure. Sorting the m raw p-values ascending, the adjusted
// value of the k-th is p_(k) * m / k, made monotone by sweeping from
// the largest downward and capping at 1. Controlling the false
// discovery rate rather than the family-wise error rate is the
// conventional choice when the tests are numerous and positively
// dependent, which they are here: the cells share both the target
// metric's scores and the same resampled articles.
func applyFDR(cells []comparisonCell) {
	m := len(cells)
	if m == 0 {
		return
	}
	order := make([]int, m)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return cells[order[a]].comp.PValue < cells[order[b]].comp.PValue
	})

	prev := 1.0
	for k := m - 1; k >= 0; k-- {
		i := order[k]
		q := cells[i].comp.PValue * float64(m) / float64(k+1)
		if q > prev {
			q = prev
		}
		if q > 1 {
			q = 1
		}
		cells[i].qValue = q
		prev = q
	}
}

// significant reports whether a cell survives as a positive finding:
// the adjusted p-value clears the FDR threshold AND the paired
// bootstrap CI for the delta excludes zero on the positive side.
func significant(c comparisonCell) bool {
	return c.qValue < fdrAlpha && c.comp.DeltaCI.Low > 0
}

// ── Output rendering ───────────────────────────────────────────────────

func renderConsole(target string, cells []comparisonCell) string {
	var b strings.Builder

	fmt.Fprintf(&b, "Paired bootstrap: %s vs baselines (Spearman)\n\n", display(target))
	fmt.Fprintf(&b, "%-15s %-12s %8s %8s %8s %20s %8s %8s\n",
		"Baseline", "Dimension", "ours", "base", "Δ", "95% CI", "p", "q(BH)")
	fmt.Fprintln(&b, strings.Repeat("─", 97))

	wins, ties, losses := 0, 0, 0
	for _, c := range cells {
		ci := c.comp.DeltaCI
		ciStr := fmt.Sprintf("[%+.3f, %+.3f]", ci.Low, ci.High)
		sig := " "
		switch {
		case significant(c):
			sig = "↑"
			wins++
		case ci.High < 0 && c.qValue < fdrAlpha:
			sig = "↓"
			losses++
		default:
			ties++
		}
		fmt.Fprintf(&b, "%-15s %-12s %+.3f   %+.3f   %+.3f   %s   %.4f   %.4f %s\n",
			display(c.baseline), c.dimension,
			c.targetRho, c.baseRho, c.comp.DeltaMean,
			ciStr, c.comp.PValue, c.qValue, sig)
	}

	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Summary: wins=%d, ties=%d, losses=%d (across %d comparisons)\n",
		wins, ties, losses, len(cells))
	fmt.Fprintf(&b, "↑ = ours significantly higher (95%% CI excludes 0 and q < %.2f)\n", fdrAlpha)
	fmt.Fprintf(&b, "↓ = ours significantly lower (same criterion)\n")
	fmt.Fprintln(&b, "  = no significant difference")
	fmt.Fprintf(&b, "q = Benjamini-Hochberg adjusted p over all %d cells of this table\n", len(cells))
	return b.String()
}

func renderLatex(target string, baselines []baselineEntry, cells []comparisonCell, level string) string {
	var b strings.Builder

	colSpec := "@{}l" + strings.Repeat("c", len(dimensions)) + "@{}"

	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)

	levelLabel := "summary-level"
	if level == "system" {
		levelLabel = "system-level"
	}
	fmt.Fprintf(&b,
		"\\caption{Paired bootstrap comparison of %s against baselines on %s "+
			"Spearman correlations. Each cell reports $\\Delta\\rho$ "+
			"(ours minus baseline) with 95\\%% CI in brackets, the raw "+
			"two-sided $p$-value, and $q$, the Benjamini-Hochberg adjusted "+
			"$p$-value over all %d (baseline, dimension) cells of this table. "+
			"Bold marks improvements that survive the correction "+
			"($q<%.2f$ and CI excludes 0).}\n",
		display(target), levelLabel, len(cells), fdrAlpha)
	fmt.Fprintf(&b, "\\label{tab:compare_%s_%s}\n", target, level)
	fmt.Fprintln(&b, `\small`)
	// Tables stay single-spaced even when the manuscript is compiled
	// with the double-spaced elsarticle `review` option.
	fmt.Fprintln(&b, `\linespread{1}\selectfont`)
	fmt.Fprintln(&b, `\setlength{\tabcolsep}{4pt}`)
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

	for i, base := range baselines {
		if i > 0 {
			// Each cell spans three lines; without this the blocks of two
			// adjacent baselines run together.
			fmt.Fprintln(&b, `\addlinespace[3pt]`)
		}
		fmt.Fprintf(&b, "%s", display(base.key))
		for _, dim := range dimensions {
			c := byBase[base.key][dim]
			fmt.Fprintf(&b, " & %s", fmtLatexCell(c))
		}
		fmt.Fprintln(&b, ` \\`)
	}

	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
}

func fmtLatexCell(c comparisonCell) string {
	delta := stripLeadingZero(c.comp.DeltaMean)
	ci := fmt.Sprintf(`[%s, %s]`,
		stripLeadingZero(c.comp.DeltaCI.Low),
		stripLeadingZero(c.comp.DeltaCI.High))

	// Stack delta / CI / p / q vertically inside the cell. On one line
	// the five-column table is roughly 1.75x wider than the elsarticle
	// text block; stacked, it fits without scaling or rotation.
	if significant(c) {
		delta = `\textbf{` + delta + `}`
	}
	return fmt.Sprintf(
		`\begin{tabular}[t]{@{}c@{}}%s\\{\scriptsize %s}\\{\scriptsize %s}\\{\scriptsize %s}\end{tabular}`,
		delta, ci, fmtSig("p", c.comp.PValue), fmtSig("q", c.qValue))
}

// fmtSig renders a p- or q-value, collapsing anything below the
// resolution the bootstrap can actually support into an inequality.
func fmtSig(name string, v float64) string {
	if v < 0.001 {
		return fmt.Sprintf("$%s<.001$", name)
	}
	return fmt.Sprintf("$%s$=%.3f", name, v)
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
