// cmd/confound measures how much of each metric's agreement with human
// judgment is explained by a single degenerate signal: how much of the
// candidate summary was copied verbatim out of the source.
//
// The motivation is a control result. A lead-window baseline that keeps
// the first k source sentences and matches them with ROUGE-L --- no
// embedder, no hyperparameter, no model (cmd/leadbaseline) --- reaches
// a mean summary-level Spearman rho on SummEval that rivals metrics
// costing three orders of magnitude more. Before concluding that those
// metrics are pointless, one has to ask what the cheap baseline is
// actually detecting. It is detecting copying: SummEval's human ratings
// are themselves correlated with extractiveness, so any metric that
// rewards verbatim overlap with the source inherits correlation it did
// not earn by modelling quality.
//
// This binary quantifies that. For every metric it reports
//
//	raw rho      Spearman against human ratings (what papers report)
//	partial rho  Spearman against human ratings with the copy rate
//	             partialled out of both sides
//	rho_copy     Spearman between the metric and the copy rate --- how
//	             much of a copy detector the metric is
//
// and recomputes the cost--quality Pareto frontier on the partial axis.
// A metric whose raw and partial values are close measures something
// extractiveness does not; a metric that collapses was riding the
// confound.
//
// The copy rate is the fraction of candidate token bigrams that occur
// in the source. Bigrams rather than unigrams because unigram overlap
// is nearly saturated by function words; see the -ngram flag to vary it.
//
// Reads output/*.json plus the embedded dataset. CPU only.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
)

var (
	datasetPath string
	inputDir    string
	outputTex   string
	ngram       int
	bootstrap   int
	seed        uint64
	oursLabel   string
	partialDims bool
)

var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

func main() {
	flag.StringVar(&inputDir, "input", "output", "directory containing metric JSON reports")
	flag.StringVar(&datasetPath, "dataset", "", "dataset JSONL path (empty = the embedded SummEval release); use with a converted corpus such as data/newsroom.jsonl")
	flag.StringVar(&outputTex, "output", "paper/confound.gen.tex", "path to write LaTeX table (- for stdout)")
	flag.IntVar(&ngram, "ngram", 2, "n-gram order for the copy rate (1 = unigram, 2 = bigram)")
	flag.IntVar(&bootstrap, "bootstrap", 2000, "cluster-bootstrap resamples over articles (0 = point estimates only)")
	flag.Uint64Var(&seed, "seed", 42, "random seed")
	flag.BoolVar(&partialDims, "allow-partial-dims", false, "include dimensional metrics that only cover some dimensions, averaging over the ones they do (UniEval's relevance prompt needs a reference, which Newsroom does not have)")
	flag.StringVar(&oursLabel, "ours", "lgs", "metric base name to mark as ours")
	flag.Parse()

	if ngram < 1 {
		log.Fatalf("-ngram must be ≥ 1, got %d", ngram)
	}

	samples, err := loadDataset(datasetPath)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	copyRate := copyRates(samples, ngram)
	human := humanMatrix(samples)

	metrics, err := loadMetrics(inputDir, samples)
	if err != nil {
		log.Fatal(err)
	}
	if len(metrics) == 0 {
		log.Fatalf("no usable reports in %s", inputDir)
	}

	rows := make([]row, 0, len(metrics)+1)
	for _, m := range metrics {
		rows = append(rows, analyse(m, human, copyRate, samples))
	}

	// The confound itself, as a row: what a bare extractiveness counter
	// scores without any metric at all.
	bare := metricScores{name: "copy-rate", display: "Copy rate (no metric)", perDim: map[string][]float64{}}
	for _, d := range dimensions {
		bare.perDim[d] = copyRate
	}
	bare.runtimeMs = 0
	bareRow := analyse(bare, human, copyRate, samples)
	bareRow.IsConfound = true
	rows = append(rows, bareRow)

	markPareto(rows)
	sort.Slice(rows, func(i, j int) bool { return rows[i].PartialMean > rows[j].PartialMean })

	fmt.Print(renderConsole(rows))
	if err := writeFile(outputTex, renderLatex(rows)); err != nil {
		log.Fatal(err)
	}
	if outputTex != "-" {
		fmt.Fprintf(os.Stderr, "\nLaTeX table written to %s\n", outputTex)
	}
}

// ── Copy rate ──────────────────────────────────────────────────────────

// copyRates returns, per sample, the fraction of the candidate's token
// n-grams that also occur in its source document. This is the crudest
// possible extractiveness signal: it needs no model and no reference,
// and it is exactly what a lead-window lexical baseline maximises.
func copyRates(samples []eval.Sample, n int) []float64 {
	srcCache := map[string]map[string]bool{}
	out := make([]float64, len(samples))

	for i, s := range samples {
		set, ok := srcCache[s.DocumentID]
		if !ok {
			set = ngramSet(tokenize(s.Document), n)
			srcCache[s.DocumentID] = set
		}
		cand := ngramList(tokenize(s.Candidate), n)
		if len(cand) == 0 {
			out[i] = 0
			continue
		}
		hit := 0
		for _, g := range cand {
			if set[g] {
				hit++
			}
		}
		out[i] = float64(hit) / float64(len(cand))
	}
	return out
}

func tokenize(s string) []string {
	s = strings.ToLower(s)
	for _, p := range []string{".", ",", "!", "?", ";", ":", "(", ")", "\"", "'"} {
		s = strings.ReplaceAll(s, p, " "+p+" ")
	}
	return strings.Fields(s)
}

func ngramList(toks []string, n int) []string {
	if len(toks) < n {
		return nil
	}
	out := make([]string, 0, len(toks)-n+1)
	for i := 0; i+n <= len(toks); i++ {
		out = append(out, strings.Join(toks[i:i+n], " "))
	}
	return out
}

func ngramSet(toks []string, n int) map[string]bool {
	out := map[string]bool{}
	for _, g := range ngramList(toks, n) {
		out[g] = true
	}
	return out
}

// ── Loading ────────────────────────────────────────────────────────────

// metricScores holds one metric's per-sample values aligned to the
// canonical sample order. Dimensional families (G-Eval, UniEval) carry
// a different score vector per dimension; every other metric repeats
// the same vector.
type metricScores struct {
	name      string
	display   string
	perDim    map[string][]float64
	runtimeMs float64
}

func loadMetrics(dir string, samples []eval.Sample) ([]metricScores, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, fmt.Errorf("glob: %w", err)
	}
	sort.Strings(paths)

	type acc struct {
		perDim  map[string][]float64
		msTotal float64
		flat    []float64
		flatMs  float64
		isDim   bool
	}
	groups := map[string]*acc{}

	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", p, err)
		}
		var r eval.Report
		if err := json.Unmarshal(raw, &r); err != nil {
			return nil, fmt.Errorf("decode %s: %w", p, err)
		}
		if r.Samples == 0 || len(r.Scores) != len(samples) {
			log.Printf("skipping %s (%d scores, expected %d)", filepath.Base(p), len(r.Scores), len(samples))
			continue
		}
		vals, err := align(r, samples)
		if err != nil {
			log.Printf("skipping %s: %v", filepath.Base(p), err)
			continue
		}
		ms := 1000.0 * r.RuntimeSec / float64(r.Samples)

		base, dim := splitDimensional(r.Metric)
		a, ok := groups[base]
		if !ok {
			a = &acc{perDim: map[string][]float64{}}
			groups[base] = a
		}
		if dim != "" {
			// A dimensional scorer must be run once per dimension in
			// deployment, so its cost is the sum over dimensions.
			a.isDim = true
			a.perDim[dim] = vals
			a.msTotal += ms
			continue
		}
		a.flat = vals
		a.flatMs = ms
	}

	out := make([]metricScores, 0, len(groups))
	for base, a := range groups {
		m := metricScores{name: base, display: displayName(base), perDim: map[string][]float64{}}
		if a.isDim {
			if len(a.perDim) != len(dimensions) {
				if !partialDims {
					log.Printf("skipping %s (only %d of %d dimension files; -allow-partial-dims includes it)",
						base, len(a.perDim), len(dimensions))
					continue
				}
				log.Printf("including %s over %d of %d dimensions", base, len(a.perDim), len(dimensions))
			}
			m.perDim = a.perDim
			m.runtimeMs = a.msTotal
		} else {
			if a.flat == nil {
				continue
			}
			for _, d := range dimensions {
				m.perDim[d] = a.flat
			}
			m.runtimeMs = a.flatMs
		}
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out, nil
}

func align(r eval.Report, samples []eval.Sample) ([]float64, error) {
	byID := make(map[string]float64, len(r.Scores))
	for _, s := range r.Scores {
		byID[s.SampleID] = s.Value
	}
	out := make([]float64, len(samples))
	for i, s := range samples {
		v, ok := byID[s.ID]
		if !ok {
			return nil, fmt.Errorf("missing score for %s", s.ID)
		}
		out[i] = v
	}
	return out, nil
}

func splitDimensional(metric string) (base, dim string) {
	for _, d := range dimensions {
		if strings.HasSuffix(metric, "_"+d) {
			return strings.TrimSuffix(metric, "_"+d), d
		}
	}
	return metric, ""
}

func humanMatrix(samples []eval.Sample) map[string][]float64 {
	out := map[string][]float64{}
	get := map[string]func(eval.Sample) float64{
		"coherence":   func(s eval.Sample) float64 { return s.Coherence },
		"consistency": func(s eval.Sample) float64 { return s.Consistency },
		"fluency":     func(s eval.Sample) float64 { return s.Fluency },
		"relevance":   func(s eval.Sample) float64 { return s.Relevance },
	}
	for _, d := range dimensions {
		v := make([]float64, len(samples))
		for i, s := range samples {
			v[i] = get[d](s)
		}
		out[d] = v
	}
	return out
}

// ── Analysis ───────────────────────────────────────────────────────────

type row struct {
	Name        string
	Display     string
	RuntimeMs   float64
	RawMean     float64
	PartialMean float64
	PartialCI   eval.CI
	RhoCopy     float64
	RawPerDim   []float64
	PartPerDim  []float64
	Dims        int
	IsOurs      bool
	IsConfound  bool
	ParetoRaw   bool
	ParetoPart  bool
}

// partialSpearman is the rank partial correlation of x and y given z:
// the correlation that survives after the component each shares with z
// is removed. Applied to ranks, this is the standard Spearman partial.
func partialSpearman(x, y, z []float64) float64 {
	rxy, rxz, ryz := eval.Spearman(x, y), eval.Spearman(x, z), eval.Spearman(y, z)
	den := math.Sqrt((1 - rxz*rxz) * (1 - ryz*ryz))
	if den == 0 {
		return 0
	}
	return (rxy - rxz*ryz) / den
}

func analyse(m metricScores, human map[string][]float64, copyRate []float64, samples []eval.Sample) row {
	r := row{
		Name: m.name, Display: m.display, RuntimeMs: m.runtimeMs,
		IsOurs: m.name == oursLabel,
	}
	// A metric may cover only some dimensions (-allow-partial-dims);
	// the mean is then over the dimensions it does cover, and the
	// missing ones are reported as NaN rather than as zero, which would
	// silently penalise the metric.
	present := 0
	for _, d := range dimensions {
		v := m.perDim[d]
		if v == nil {
			r.RawPerDim = append(r.RawPerDim, math.NaN())
			r.PartPerDim = append(r.PartPerDim, math.NaN())
			continue
		}
		raw := eval.Spearman(v, human[d])
		par := partialSpearman(v, human[d], copyRate)
		r.RawPerDim = append(r.RawPerDim, raw)
		r.PartPerDim = append(r.PartPerDim, par)
		r.RawMean += raw
		r.PartialMean += par
		present++
	}
	if present > 0 {
		r.RawMean /= float64(present)
		r.PartialMean /= float64(present)
	}
	r.Dims = present
	if v := m.perDim["consistency"]; v != nil {
		r.RhoCopy = eval.Spearman(v, copyRate)
	} else if v := m.perDim["coherence"]; v != nil {
		r.RhoCopy = eval.Spearman(v, copyRate)
	}

	if bootstrap > 0 {
		r.PartialCI = bootstrapPartial(m, human, copyRate, samples)
	}
	return r
}

// bootstrapPartial resamples articles, not summaries: SummEval's 16
// candidates per article share a source and are not independent.
func bootstrapPartial(m metricScores, human map[string][]float64, copyRate []float64, samples []eval.Sample) eval.CI {
	byDoc := map[string][]int{}
	for i, s := range samples {
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], i)
	}
	docs := make([]string, 0, len(byDoc))
	for d := range byDoc {
		docs = append(docs, d)
	}
	sort.Strings(docs)

	rng := rand.New(rand.NewPCG(seed, seed^0xc0ffee))
	vals := make([]float64, 0, bootstrap)
	pick := make([]int, 0, len(samples))
	bx := make([]float64, 0, len(samples))
	by := make([]float64, 0, len(samples))
	bz := make([]float64, 0, len(samples))

	for b := 0; b < bootstrap; b++ {
		pick = pick[:0]
		for range docs {
			pick = append(pick, byDoc[docs[rng.IntN(len(docs))]]...)
		}
		var total float64
		var present int
		for _, d := range dimensions {
			v, h := m.perDim[d], human[d]
			if v == nil {
				continue
			}
			present++
			bx, by, bz = bx[:0], by[:0], bz[:0]
			for _, p := range pick {
				bx = append(bx, v[p])
				by = append(by, h[p])
				bz = append(bz, copyRate[p])
			}
			total += partialSpearman(bx, by, bz)
		}
		if present > 0 {
			total /= float64(present)
		}
		vals = append(vals, total)
	}
	sort.Float64s(vals)
	at := func(p float64) float64 {
		pos := p * float64(len(vals)-1)
		lo, hi := int(math.Floor(pos)), int(math.Ceil(pos))
		if lo == hi {
			return vals[lo]
		}
		f := pos - float64(lo)
		return vals[lo]*(1-f) + vals[hi]*f
	}
	return eval.CI{Low: at(0.025), High: at(0.975)}
}

// markPareto flags the metrics that nothing cheaper correlates at least
// as well as, on each of the two axes. The confound row is excluded: it
// is a diagnostic, not a candidate metric.
func markPareto(rows []row) {
	idx := make([]int, 0, len(rows))
	for i := range rows {
		if !rows[i].IsConfound {
			idx = append(idx, i)
		}
	}
	sort.Slice(idx, func(a, b int) bool { return rows[idx[a]].RuntimeMs < rows[idx[b]].RuntimeMs })

	bestRaw, bestPart := math.Inf(-1), math.Inf(-1)
	for _, i := range idx {
		if rows[i].RawMean > bestRaw {
			rows[i].ParetoRaw = true
			bestRaw = rows[i].RawMean
		}
		if rows[i].PartialMean > bestPart {
			rows[i].ParetoPart = true
			bestPart = rows[i].PartialMean
		}
	}
}

// ── Rendering ──────────────────────────────────────────────────────────

var displayNames = map[string]string{
	"bleu": "BLEU", "rouge": "ROUGE-L", "chrf": "ChrF", "meteor": "METEOR",
	"smartstring": "SMART-String", "smartmodel": "SMART-Model",
	"embedscorer": "EmbedScorer", "bertscore": "BERTScore",
	"moverscore": "MoverScore", "bartscore": "BARTScore",
	"gptscore": "GPTScore", "unieval": "UniEval", "geval": "G-Eval",
	"lgs": "LGS", "lead3sent": "Lead-3 (mean-of-max)",
	"lead5sent": "Lead-5 (mean-of-max)", "lead3whole": "Lead-3 (whole block)",
}

func displayName(base string) string {
	if v, ok := displayNames[base]; ok {
		return v
	}
	return base
}

func renderConsole(rows []row) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Extractiveness confound on %s (copy rate = fraction of candidate %d-grams found in source)\n\n", corpusName(), ngram)
	fmt.Fprintf(&b, "%-22s %9s %9s %8s %9s %10s %s\n",
		"Metric", "raw rho", "part rho", "drop", "rho_copy", "ms/sample", "Pareto(raw/part)")
	fmt.Fprintln(&b, strings.Repeat("─", 96))
	for _, r := range rows {
		mark := ""
		if r.IsOurs {
			mark = " *"
		}
		if r.IsConfound {
			mark = " †"
		}
		par := "  ."
		switch {
		case r.ParetoRaw && r.ParetoPart:
			par = "raw+part"
		case r.ParetoPart:
			par = "    part"
		case r.ParetoRaw:
			par = "raw"
		}
		fmt.Fprintf(&b, "%-22s %9.3f %9.3f %8.3f %9.3f %10.2f  %s\n",
			r.Display+mark, r.RawMean, r.PartialMean, r.RawMean-r.PartialMean,
			r.RhoCopy, r.RuntimeMs, par)
	}
	fmt.Fprintln(&b, "\n* = ours   † = the confound itself, not a metric")
	fmt.Fprintln(&b, "drop = how much of the raw correlation was extractiveness")

	// Per-dimension partial rho. The mean hides the thing that matters
	// most for choosing a metric in practice: metrics differ by
	// dimension far more than they differ on average, and a metric
	// whose mean is unremarkable can still be the best available
	// coherence detector.
	fmt.Fprintf(&b, "\nPartial rho per dimension (copy rate partialled out)\n\n")
	fmt.Fprintf(&b, "%-22s", "Metric")
	for _, d := range dimensions {
		fmt.Fprintf(&b, "%13s", d)
	}
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, strings.Repeat("─", 22+13*len(dimensions)))
	for _, r := range rows {
		if r.IsConfound {
			continue
		}
		fmt.Fprintf(&b, "%-22s", r.Display)
		for _, v := range r.PartPerDim {
			if math.IsNaN(v) {
				fmt.Fprintf(&b, "%13s", "n/a")
				continue
			}
			fmt.Fprintf(&b, "%13.3f", v)
		}
		fmt.Fprintln(&b)
	}
	return b.String()
}

func renderLatex(rows []row) string {
	var b strings.Builder
	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintf(&b, `\caption{Extractiveness confound on \textsc{%s}. The copy rate of a candidate is the fraction of its token %d-grams that occur in the source; it requires no model, no reference and no hyperparameter. $\bar{\rho}$ is the mean summary-level Spearman correlation with human ratings across the four dimensions, $\bar{\rho}_{\mathrm{part}}$ the same quantity with the copy rate partialled out of both sides, and $\rho_{\mathrm{copy}}$ the correlation between the metric and the copy rate. Brackets give the 95\%% cluster-bootstrap interval over articles. The final row is the copy rate scored as if it were a metric: on the raw axis it outranks most of the pool. Pareto status is given on both axes.}`+"\n", corpusName(), ngram)
	fmt.Fprintln(&b, `\label{tab:confound}`)
	fmt.Fprintln(&b, `\small`)
	fmt.Fprintln(&b, `\linespread{1}\selectfont`)
	fmt.Fprintln(&b, `\setlength{\tabcolsep}{4pt}`)
	fmt.Fprintln(&b, `\begin{tabular}{@{}lrrrrl@{}}`)
	fmt.Fprintln(&b, `\toprule`)
	fmt.Fprintln(&b, `Metric & ms/sample & $\bar{\rho}$ & $\bar{\rho}_{\mathrm{part}}$ & $\rho_{\mathrm{copy}}$ & Pareto \\`)
	fmt.Fprintln(&b, `\midrule`)
	for _, r := range rows {
		if r.IsConfound {
			fmt.Fprintln(&b, `\midrule`)
		}
		label := r.Display
		if r.IsOurs {
			label = `\textbf{` + label + `}`
		}
		status := "---"
		switch {
		case r.ParetoRaw && r.ParetoPart:
			status = "raw, partial"
		case r.ParetoPart:
			status = "partial"
		case r.ParetoRaw:
			status = "raw"
		}
		if r.IsConfound {
			status = "\\emph{diagnostic}"
		}
		ci := ""
		if r.PartialCI.High != 0 || r.PartialCI.Low != 0 {
			ci = fmt.Sprintf(` {\scriptsize [%s, %s]}`, num(r.PartialCI.Low), num(r.PartialCI.High))
		}
		fmt.Fprintf(&b, "%s & %.2f & %s & %s%s & %s & %s \\\\\n",
			label, r.RuntimeMs, num(r.RawMean), num(r.PartialMean), ci, num(r.RhoCopy), status)
	}
	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
}

func num(x float64) string {
	s := fmt.Sprintf("%.3f", x)
	if x >= 0 && x < 1 {
		return strings.TrimPrefix(s, "0")
	}
	if x > -1 && x < 0 {
		return "-" + strings.TrimPrefix(s[1:], "0")
	}
	return s
}

// corpusName labels the table with the corpus actually analysed, so a
// Newsroom run is not reported as SummEval.
func corpusName() string {
	if datasetPath == "" {
		return "SummEval"
	}
	base := filepath.Base(datasetPath)
	return strings.TrimSuffix(strings.TrimSuffix(base, ".jsonl"), ".json")
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

// loadDataset reads the embedded SummEval release by default, or an
// external JSONL corpus in the same shape -- see cmd/newsroom, which
// converts the Newsroom human-evaluation release into it.
func loadDataset(path string) ([]eval.Sample, error) {
	if path == "" {
		return eval.NewDataset(dataset.Summeval, dataset.SummevalDefaultPath, 0)
	}
	return eval.NewDataset(os.DirFS(filepath.Dir(path)), filepath.Base(path), 0)
}
