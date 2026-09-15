// cmd/lambdaci quantifies the sampling uncertainty of the lead-bias
// exponent selection itself. The ablation table reports a point
// estimate of λ* (the grid value maximising dev-split mean Spearman
// ρ), but a point estimate carries no information about whether a
// different λ would have won on a different draw of 50 articles.
// This binary answers that question directly: it resamples articles
// with replacement (the same cluster bootstrap used everywhere else
// in pkg/eval) and, on every resample, re-runs the *entire selection
// procedure* — recompute mean ρ for every λ on the grid, take the
// argmax — then reports how often each grid value wins.
//
// Three statistics are produced.
//
//  1. Per-λ point estimate, 95% CI, and P(argmax): the selection
//     distribution of λ* for one (split, embedder) pair.
//  2. A PAIRED cross-backbone test. Both backbones are swept on the
//     same 50 dev articles, so the same resample can be scored under
//     both grids; this yields P(λ*_A = λ*_B), which is the statistic
//     needed to claim that the optimum "shifts" between backbones
//     rather than merely differing in two noisy point estimates.
//  3. A PAIRED dev-vs-test test for one backbone, which measures the
//     same instability within a single representation space and so
//     calibrates statistic 2 against a no-backbone-change control.
//
// All three read the per-sample scores already present in
// ablation/*.json — no embedder passes are required.
//
// Output: console summary plus paper/lambdaci.gen.tex.
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
	inputDir  string
	outputTex string
	bootstrap int
	seed      uint64
	backboneA string
	backboneB string
	devSplit  string
	testSplit string
)

func main() {
	flag.StringVar(&inputDir, "input", "ablation", "directory containing LGS ablation JSON reports")
	flag.StringVar(&outputTex, "output", "paper/lambdaci.gen.tex", "path to write LaTeX table (- for stdout)")
	flag.IntVar(&bootstrap, "bootstrap", 5000, "cluster-bootstrap resamples over articles")
	flag.Uint64Var(&seed, "seed", 42, "random seed for reproducibility")
	flag.StringVar(&backboneA, "backbone-a", "nomic-embed-text", "first backbone in the paired cross-backbone test")
	flag.StringVar(&backboneB, "backbone-b", "bge-m3", "second backbone in the paired cross-backbone test")
	flag.StringVar(&devSplit, "dev-split", "first50", "development split identifier")
	flag.StringVar(&testSplit, "test-split", "last50", "test split identifier")
	flag.Parse()

	if bootstrap < 2 {
		log.Fatalf("-bootstrap must be ≥ 2, got %d", bootstrap)
	}

	samples, err := eval.NewDataset(dataset.Summeval, dataset.SummevalDefaultPath, 0)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	variants, err := loadVariants(inputDir)
	if err != nil {
		log.Fatal(err)
	}
	if len(variants) == 0 {
		log.Fatalf("no LGS ablation reports found in %s", inputDir)
	}

	groups := groupVariants(variants)

	// ── 1. Selection distribution per (split, backbone) ──────────────
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	results := make([]groupResult, 0, len(keys))
	for _, k := range keys {
		g := groups[k]
		if len(g.grid) < 2 {
			continue // a single λ has no selection to speak of
		}
		res, err := analyse(samples, g, bootstrap, seed)
		if err != nil {
			log.Printf("skipping %s: %v", k, err)
			continue
		}
		results = append(results, res)
	}
	if len(results) == 0 {
		log.Fatal("no (split, backbone) group had a λ grid to analyse")
	}

	// ── 2. Paired cross-backbone test on the dev split ───────────────
	crossBackbone := pairedSelection(samples, groups,
		groupKey(devSplit, backboneA), groupKey(devSplit, backboneB),
		bootstrap, seed+1, false, nil)

	// Both controls below are pinned to the cross-backbone grid so the
	// three agreement probabilities are read off the same scale.
	grid := crossBackbone.Grid

	// ── 3. Controls that change no backbone at all ───────────────────
	// The cross-backbone number is only interpretable against a
	// baseline that isolates pure resampling noise. `selfPaired`
	// compares one group against ITSELF on two independent draws of
	// the same articles: identical data, identical backbone, identical
	// grid, so every disagreement is sampling noise. If the
	// cross-backbone agreement is no lower than this, the apparent
	// "shift" carries no backbone signal. The dev-vs-test pair is
	// reported alongside as the coarser, article-level control.
	selfNoise := pairedSelection(samples, groups,
		groupKey(devSplit, backboneA), groupKey(devSplit, backboneA),
		bootstrap, seed+2, true, grid)
	devTest := pairedSelection(samples, groups,
		groupKey(devSplit, backboneA), groupKey(testSplit, backboneA),
		bootstrap, seed+3, false, grid)

	crossBackbone.Title = "cross-backbone (shared draw)"
	selfNoise.Title = "same backbone, two independent draws (noise floor)"
	devTest.Title = "dev vs test, same backbone"
	controls := []pairedResult{crossBackbone, selfNoise, devTest}

	fmt.Print(renderConsole(results, controls))

	if err := writeFile(outputTex, renderLatex(results, controls)); err != nil {
		log.Fatal(err)
	}
	if outputTex != "-" {
		fmt.Fprintf(os.Stderr, "\nLaTeX table written to %s\n", outputTex)
	}
}

// ── Loading ────────────────────────────────────────────────────────────

// variant is one ablation report: an LGS run at a fixed λ on a fixed
// split with a fixed embedder, plus its per-sample scores.
type variant struct {
	split  string
	embed  string
	lambda float64
	byID   map[string]float64
	source string
}

// group collects every λ of one (split, embedder) pair.
type group struct {
	split string
	embed string
	grid  []variant // ascending λ
}

func groupKey(split, embed string) string { return split + "|" + embed }

func loadVariants(dir string) ([]variant, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
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
		if r.Metric != "lgs" || len(r.Scores) == 0 {
			continue
		}
		nm := parseNorm(r.Norm)
		split, ok := nm["split"]
		if !ok {
			continue
		}
		// Legacy coarse-sweep files predate the embed_model field and
		// were all produced with the canonical backbone.
		embed := nm["embed_model"]
		if embed == "" {
			embed = "nomic-embed-text"
		}
		lambda, ok := parseFloat(nm["lead_lambda"])
		if !ok {
			continue
		}
		byID := make(map[string]float64, len(r.Scores))
		for _, s := range r.Scores {
			byID[s.SampleID] = s.Value
		}
		out = append(out, variant{
			split: split, embed: stripTag(embed), lambda: lambda,
			byID: byID, source: filepath.Base(p),
		})
	}
	return out, nil
}

// groupVariants buckets variants by (split, embedder) and deduplicates
// repeated λ values. The coarse sweep and the finer b5k sweep both
// contain λ=0.5 for the canonical backbone; scoring is deterministic,
// so the two carry identical per-sample values and either may be kept.
// A genuine disagreement indicates stale snapshots and is reported.
func groupVariants(vs []variant) map[string]*group {
	out := map[string]*group{}
	for _, v := range vs {
		k := groupKey(v.split, v.embed)
		g, ok := out[k]
		if !ok {
			g = &group{split: v.split, embed: v.embed}
			out[k] = g
		}
		dup := -1
		for i, e := range g.grid {
			if math.Abs(e.lambda-v.lambda) < 1e-9 {
				dup = i
				break
			}
		}
		if dup >= 0 {
			if !sameScores(g.grid[dup].byID, v.byID) {
				log.Printf("WARNING: %s and %s are both %s λ=%g but disagree on per-sample scores; keeping %s",
					g.grid[dup].source, v.source, k, v.lambda, g.grid[dup].source)
			}
			continue
		}
		g.grid = append(g.grid, v)
	}
	for _, g := range out {
		sort.Slice(g.grid, func(i, j int) bool { return g.grid[i].lambda < g.grid[j].lambda })
	}
	return out
}

func sameScores(a, b map[string]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for k, av := range a {
		bv, ok := b[k]
		if !ok || math.Abs(av-bv) > 1e-12 {
			return false
		}
	}
	return true
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

// stripTag drops Ollama's ":tag" suffix so "bge-m3" and "bge-m3:latest"
// compare equal.
func stripTag(s string) string {
	if i := strings.Index(s, ":"); i >= 0 {
		return s[:i]
	}
	return s
}

// ── Aligned per-split view ─────────────────────────────────────────────

// aligned holds one split's samples in a fixed order, together with the
// human ratings and the per-λ metric scores in that same order, so the
// bootstrap can address everything by integer index.
type aligned struct {
	docs    []string // unique document IDs, ascending
	byDoc   [][]int  // indices of each document's samples
	human   [4][]float64
	scores  [][]float64 // [λ index][sample index]
	lambdas []float64
}

func align(samples []eval.Sample, g *group) (*aligned, error) {
	if len(g.grid) == 0 {
		return nil, fmt.Errorf("empty λ grid")
	}
	first := g.grid[0].byID

	idx := make([]int, 0, len(first))
	for i, s := range samples {
		if _, ok := first[s.ID]; ok {
			idx = append(idx, i)
		}
	}
	if len(idx) != len(first) {
		return nil, fmt.Errorf("split covers %d samples but dataset matched %d", len(first), len(idx))
	}

	a := &aligned{}
	byDoc := map[string][]int{}
	for pos, i := range idx {
		s := samples[i]
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], pos)
		a.human[0] = append(a.human[0], s.Coherence)
		a.human[1] = append(a.human[1], s.Consistency)
		a.human[2] = append(a.human[2], s.Fluency)
		a.human[3] = append(a.human[3], s.Relevance)
	}
	for d := range byDoc {
		a.docs = append(a.docs, d)
	}
	sort.Strings(a.docs)
	for _, d := range a.docs {
		a.byDoc = append(a.byDoc, byDoc[d])
	}

	for _, v := range g.grid {
		col := make([]float64, len(idx))
		for pos, i := range idx {
			val, ok := v.byID[samples[i].ID]
			if !ok {
				return nil, fmt.Errorf("λ=%g missing score for sample %s", v.lambda, samples[i].ID)
			}
			col[pos] = val
		}
		a.scores = append(a.scores, col)
		a.lambdas = append(a.lambdas, v.lambda)
	}
	return a, nil
}

// meanRho is the selection criterion: Spearman ρ against each of the
// four human dimensions, averaged. `pick` lists sample positions.
func (a *aligned) meanRho(lambdaIdx int, pick []int, bufX, bufY []float64) float64 {
	sc := a.scores[lambdaIdx]
	bufX = bufX[:0]
	for _, p := range pick {
		bufX = append(bufX, sc[p])
	}
	var total float64
	for d := 0; d < 4; d++ {
		h := a.human[d]
		bufY = bufY[:0]
		for _, p := range pick {
			bufY = append(bufY, h[p])
		}
		total += eval.Spearman(bufX, bufY)
	}
	return total / 4.0
}

// drawDocs resamples articles with replacement and returns the sample
// positions of the drawn articles.
func (a *aligned) drawDocs(rng *rand.Rand, buf []int) []int {
	buf = buf[:0]
	for range a.docs {
		buf = append(buf, a.byDoc[rng.IntN(len(a.docs))]...)
	}
	return buf
}

// ── Statistic 1: selection distribution ────────────────────────────────

type lambdaRow struct {
	Lambda   float64
	Point    float64
	CI       eval.CI
	ArgmaxP  float64
	Selected bool // carries the point-estimate argmax
}

type groupResult struct {
	Split string
	Embed string
	Rows  []lambdaRow
	N     int
}

func analyse(samples []eval.Sample, g *group, b int, sd uint64) (groupResult, error) {
	a, err := align(samples, g)
	if err != nil {
		return groupResult{}, err
	}

	n := len(a.human[0])
	bufX := make([]float64, 0, n)
	bufY := make([]float64, 0, n)
	pick := make([]int, 0, n)

	all := make([]int, n)
	for i := range all {
		all[i] = i
	}

	rows := make([]lambdaRow, len(a.lambdas))
	dist := make([][]float64, len(a.lambdas))
	for i := range a.lambdas {
		rows[i] = lambdaRow{Lambda: a.lambdas[i], Point: a.meanRho(i, all, bufX, bufY)}
		dist[i] = make([]float64, 0, b)
	}

	rng := rand.New(rand.NewPCG(sd, sd^0xa11ce))
	wins := make([]int, len(a.lambdas))
	vals := make([]float64, len(a.lambdas))
	for r := 0; r < b; r++ {
		pick = a.drawDocs(rng, pick)
		best, bestIdx := math.Inf(-1), 0
		for i := range a.lambdas {
			v := a.meanRho(i, pick, bufX, bufY)
			vals[i] = v
			dist[i] = append(dist[i], v)
			if v > best {
				best, bestIdx = v, i
			}
		}
		wins[bestIdx]++
	}

	bestPoint := 0
	for i := range rows {
		if rows[i].Point > rows[bestPoint].Point {
			bestPoint = i
		}
		rows[i].CI = percentileCI(dist[i])
		rows[i].ArgmaxP = float64(wins[i]) / float64(b)
	}
	rows[bestPoint].Selected = true

	return groupResult{Split: g.split, Embed: g.embed, Rows: rows, N: len(a.docs)}, nil
}

// ── Statistics 2 and 3: paired selection comparison ────────────────────

type pairedResult struct {
	OK       bool
	Title    string
	LabelA   string
	LabelB   string
	Grid     []float64
	PointA   float64
	PointB   float64
	PSame    float64 // P(λ*_A = λ*_B)
	PALower  float64 // P(λ*_A < λ*_B)
	PAHigher float64 // P(λ*_A > λ*_B)
	Reason   string
}

// pairedSelection re-runs the whole selection procedure for two groups
// on the SAME resample, which is what isolates a genuine shift in the
// optimum from two independently noisy point estimates. The comparison
// is restricted to the λ values present in both grids: an unrestricted
// comparison would give the finer grid more chances to win purely by
// having more candidates.
func pairedSelection(samples []eval.Sample, groups map[string]*group, keyA, keyB string, b int, sd uint64, forceIndependent bool, restrictTo []float64) pairedResult {
	res := pairedResult{LabelA: keyA, LabelB: keyB}

	gA, okA := groups[keyA]
	gB, okB := groups[keyB]
	if !okA || !okB {
		res.Reason = fmt.Sprintf("missing sweep for %s or %s", keyA, keyB)
		return res
	}

	common := commonGrid(gA, gB)
	// Every control must be scored on the SAME grid. The probability
	// that two argmaxes coincide falls mechanically as the grid grows,
	// so a noise floor measured on nine candidates cannot be compared
	// against a cross-backbone test measured on five.
	if restrictTo != nil {
		common = intersect(common, restrictTo)
	}
	if len(common) < 2 {
		res.Reason = "fewer than two λ values common to both grids"
		return res
	}
	subA, subB := restrict(gA, common), restrict(gB, common)

	aA, err := align(samples, subA)
	if err != nil {
		res.Reason = fmt.Sprintf("align %s: %v", keyA, err)
		return res
	}
	aB, err := align(samples, subB)
	if err != nil {
		res.Reason = fmt.Sprintf("align %s: %v", keyB, err)
		return res
	}

	// Sharing the draw is what makes the comparison paired: both grids
	// see exactly the same articles, so a disagreement is attributable
	// to the grids rather than to the data. Two groups covering
	// different splits cannot share a draw. `forceIndependent` opts out
	// deliberately, to measure how often the SAME configuration
	// disagrees with itself across two draws — the pure-noise floor
	// against which every other number here must be read.
	shared := sameDocs(aA, aB) && !forceIndependent

	nA, nB := len(aA.human[0]), len(aB.human[0])
	bufX := make([]float64, 0, max(nA, nB))
	bufY := make([]float64, 0, max(nA, nB))
	pickA := make([]int, 0, nA)
	pickB := make([]int, 0, nB)

	allA := seq(nA)
	allB := seq(nB)
	res.Grid = common
	res.PointA = common[argmaxOver(aA, allA, bufX, bufY)]
	res.PointB = common[argmaxOver(aB, allB, bufX, bufY)]

	rng := rand.New(rand.NewPCG(sd, sd^0xb0b))
	var same, lower, higher int
	for r := 0; r < b; r++ {
		pickA = aA.drawDocs(rng, pickA)
		if shared {
			pickB = append(pickB[:0], pickA...)
		} else {
			pickB = aB.drawDocs(rng, pickB)
		}
		ia := argmaxOver(aA, pickA, bufX, bufY)
		ib := argmaxOver(aB, pickB, bufX, bufY)
		switch {
		case ia == ib:
			same++
		case common[ia] < common[ib]:
			lower++
		default:
			higher++
		}
	}
	res.OK = true
	res.PSame = float64(same) / float64(b)
	res.PALower = float64(lower) / float64(b)
	res.PAHigher = float64(higher) / float64(b)
	return res
}

func argmaxOver(a *aligned, pick []int, bufX, bufY []float64) int {
	best, bestIdx := math.Inf(-1), 0
	for i := range a.lambdas {
		if v := a.meanRho(i, pick, bufX, bufY); v > best {
			best, bestIdx = v, i
		}
	}
	return bestIdx
}

// intersect keeps the values of a that also occur in b.
func intersect(a, b []float64) []float64 {
	var out []float64
	for _, x := range a {
		for _, y := range b {
			if math.Abs(x-y) < 1e-9 {
				out = append(out, x)
				break
			}
		}
	}
	return out
}

func commonGrid(a, b *group) []float64 {
	var out []float64
	for _, va := range a.grid {
		for _, vb := range b.grid {
			if math.Abs(va.lambda-vb.lambda) < 1e-9 {
				out = append(out, va.lambda)
				break
			}
		}
	}
	sort.Float64s(out)
	return out
}

func restrict(g *group, grid []float64) *group {
	out := &group{split: g.split, embed: g.embed}
	for _, l := range grid {
		for _, v := range g.grid {
			if math.Abs(v.lambda-l) < 1e-9 {
				out.grid = append(out.grid, v)
				break
			}
		}
	}
	return out
}

func sameDocs(a, b *aligned) bool {
	if len(a.docs) != len(b.docs) {
		return false
	}
	for i := range a.docs {
		if a.docs[i] != b.docs[i] {
			return false
		}
	}
	return true
}

func seq(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

func percentileCI(values []float64) eval.CI {
	if len(values) < 2 {
		return eval.CI{}
	}
	s := make([]float64, len(values))
	copy(s, values)
	sort.Float64s(s)
	at := func(p float64) float64 {
		pos := p * float64(len(s)-1)
		lo, hi := int(math.Floor(pos)), int(math.Ceil(pos))
		if lo == hi {
			return s[lo]
		}
		f := pos - float64(lo)
		return s[lo]*(1-f) + s[hi]*f
	}
	return eval.CI{Low: at(0.025), High: at(0.975)}
}

// ── Rendering ──────────────────────────────────────────────────────────

func renderConsole(results []groupResult, controls []pairedResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Selection uncertainty of the lead-bias exponent (%d cluster-bootstrap resamples over articles)\n", bootstrap)

	for _, r := range results {
		fmt.Fprintf(&b, "\n── split=%s  embedder=%s  (%d articles) ──\n", r.Split, r.Embed, r.N)
		fmt.Fprintf(&b, "%8s %10s %22s %12s\n", "lambda", "mean rho", "95% CI", "P(argmax)")
		fmt.Fprintln(&b, strings.Repeat("─", 56))
		for _, row := range r.Rows {
			mark := " "
			if row.Selected {
				mark = "*"
			}
			fmt.Fprintf(&b, "%7g%s %10.4f %22s %11.1f%%\n",
				row.Lambda, mark, row.Point,
				fmt.Sprintf("[%.3f, %.3f]", row.CI.Low, row.CI.High),
				100*row.ArgmaxP)
		}
	}

	fmt.Fprintln(&b, "\n* = argmax of the point estimate (the value the ablation table reports as λ*)")

	fmt.Fprintln(&b, "\n── Paired selection comparisons (selection procedure re-run on both sides) ──")
	for _, p := range controls {
		fmt.Fprintf(&b, "\n%s\n  %s  vs  %s\n", p.Title, p.LabelA, p.LabelB)
		if !p.OK {
			fmt.Fprintf(&b, "  unavailable: %s\n", p.Reason)
			continue
		}
		fmt.Fprintf(&b, "  common grid      : %v\n", p.Grid)
		fmt.Fprintf(&b, "  point estimates  : λ*_A=%g, λ*_B=%g\n", p.PointA, p.PointB)
		fmt.Fprintf(&b, "  P(λ*_A = λ*_B)   : %.1f%%\n", 100*p.PSame)
		fmt.Fprintf(&b, "  P(λ*_A < λ*_B)   : %.1f%%\n", 100*p.PALower)
		fmt.Fprintf(&b, "  P(λ*_A > λ*_B)   : %.1f%%\n", 100*p.PAHigher)
	}
	return b.String()
}

func renderLatex(results []groupResult, controls []pairedResult) string {
	var b strings.Builder
	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintf(&b, `\caption{Selection uncertainty of the lead-bias exponent. For each (split, backbone) the whole selection procedure --- compute mean Spearman $\rho$ across the four \textsc{SummEval} dimensions at every grid value, take the argmax --- is re-run on %d cluster-bootstrap resamples over articles. $P(\arg\max)$ is the fraction of resamples in which that grid value is selected; a selection procedure that identifies a well-separated optimum concentrates this mass on one row. $\lambda^{\star}$ marks the argmax of the point estimate, i.e.\ the value the ablation table reports.}`+"\n", bootstrap)
	fmt.Fprintln(&b, `\label{tab:lambda_selection_ci}`)
	fmt.Fprintln(&b, `\small`)
	fmt.Fprintln(&b, `\linespread{1}\selectfont`)
	fmt.Fprintln(&b, `\setlength{\tabcolsep}{4pt}`)
	fmt.Fprintln(&b, `\begin{tabular}{@{}llrrr@{}}`)
	fmt.Fprintln(&b, `\toprule`)
	fmt.Fprintln(&b, `Split / backbone & $\lambda$ & Mean $\rho$ & 95\% CI & $P(\arg\max)$ \\`)
	fmt.Fprintln(&b, `\midrule`)

	for gi, r := range results {
		if gi > 0 {
			fmt.Fprintln(&b, `\addlinespace[3pt]`)
		}
		label := fmt.Sprintf(`%s / \texttt{%s}`, splitLabel(r.Split), r.Embed)
		for i, row := range r.Rows {
			cell := ""
			if i == 0 {
				cell = label
			}
			lam := fmt.Sprintf("%g", row.Lambda)
			if row.Selected {
				lam += ` $\star$`
			}
			fmt.Fprintf(&b, "%s & %s & %s & {\\scriptsize [%s, %s]} & %.1f\\%% \\\\\n",
				cell, lam, stripLeadingZero(row.Point),
				stripLeadingZero(row.CI.Low), stripLeadingZero(row.CI.High),
				100*row.ArgmaxP)
		}
	}

	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)

	fmt.Fprintln(&b, `\vspace{2mm}`)
	fmt.Fprintln(&b, `\begin{tabular}{@{}lrrr@{}}`)
	fmt.Fprintln(&b, `\toprule`)
	fmt.Fprintln(&b, `Paired comparison & $P(\lambda^{\star}_A{=}\lambda^{\star}_B)$ & $P(\lambda^{\star}_A{<}\lambda^{\star}_B)$ & $P(\lambda^{\star}_A{>}\lambda^{\star}_B)$ \\`)
	fmt.Fprintln(&b, `\midrule`)
	for _, p := range controls {
		if !p.OK {
			fmt.Fprintf(&b, "%s & \\multicolumn{3}{c}{\\emph{%s}} \\\\\n", p.Title, latexEscape(p.Reason))
			continue
		}
		fmt.Fprintf(&b, "%s & %.1f\\%% & %.1f\\%% & %.1f\\%% \\\\\n",
			p.Title, 100*p.PSame, 100*p.PALower, 100*p.PAHigher)
	}
	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
}

func splitLabel(s string) string {
	switch s {
	case "first50":
		return "dev"
	case "last50":
		return "test"
	case "all":
		return "full"
	}
	return s
}

func latexEscape(s string) string {
	r := strings.NewReplacer("_", `\_`, "|", `/`, "&", `\&`, "%", `\%`, "#", `\#`)
	return `\texttt{` + r.Replace(s) + `}`
}

func stripLeadingZero(x float64) string {
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
