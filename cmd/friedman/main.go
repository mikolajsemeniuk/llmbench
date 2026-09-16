// cmd/friedman runs a Friedman test (blocks = the 100 SummEval
// articles, treatments = the metric pool) plus Wilcoxon signed-rank
// post-hoc tests (Holm-corrected) and a Nemenyi critical-difference
// check, answering reviewer #1 (#5) and #5 (#9)'s request for
// omnibus + pairwise significance testing across the whole metric
// pool rather than just LGS-vs-baseline.
//
// Blocking structure matters: using the four SummEval dimensions as
// blocks gives only 4 blocks, which has no power. Instead, for each
// dimension, each block is one article and the response of a metric
// in that block is the metric's own Spearman correlation with human
// ratings *within that article's 16 systems*. That gives 100 blocks
// x k metrics per dimension -- enough power for both tests.
//
// Usage:
//
//	go run ./cmd/friedman -metrics bleu,rouge,...,lgs -output paper/friedman.gen.tex
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
)

var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

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
	"lead3sent":   "Lead-3 (sent)",
	"lead5sent":   "Lead-5 (sent)",
	"lead3whole":  "Lead-3 (whole)",
}

func main() {
	var inputDir, output, metricsCSV, target string
	flag.StringVar(&inputDir, "input", "output", "directory containing metric JSON reports")
	flag.StringVar(&output, "output", "paper/friedman.gen.tex", "path to write LaTeX table (- for stdout)")
	flag.StringVar(&metricsCSV, "metrics", "bleu,rouge,chrf,meteor,smartstring,embedscorer,bertscore,moverscore,smartmodel,bartscore,gptscore,unieval,geval,lgs",
		"comma-separated metric keys (file basenames, or prefix for dimensional files like unieval/geval)")
	flag.StringVar(&target, "target", "lgs", "metric to use as the reference column in pairwise post-hoc tests")
	flag.Parse()

	metrics := splitCSV(metricsCSV)
	if len(metrics) < 3 {
		log.Fatal("need at least 3 metrics for a Friedman test")
	}

	samples, err := eval.NewDataset(dataset.Summeval, dataset.SummevalDefaultPath, 0)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	scoresByMetric := make(map[string]map[string][]float64, len(metrics)) // metric -> dim -> aligned scores
	for _, m := range metrics {
		byDim, err := loadBaseline(samples, inputDir, m)
		if err != nil {
			log.Fatalf("load %q: %v", m, err)
		}
		scoresByMetric[m] = byDim
	}

	docs, byDoc := groupByDoc(samples)

	results := make([]dimResult, 0, len(dimensions))
	for _, dim := range dimensions {
		human := humanScores(samples, dim)
		// perArticleRho[metric][article] = Spearman(metric scores, human) within that article
		perArticleRho := make(map[string][]float64, len(metrics))
		for _, m := range metrics {
			perArticleRho[m] = make([]float64, len(docs))
		}
		for ai, doc := range docs {
			idx := byDoc[doc]
			y := pick(human, idx)
			for _, m := range metrics {
				x := pick(scoresByMetric[m][dim], idx)
				perArticleRho[m][ai] = eval.Spearman(x, y)
			}
		}
		results = append(results, friedmanForDimension(dim, metrics, perArticleRho, target))
	}

	fmt.Print(renderConsole(results))
	if output != "" {
		if err := writeFile(output, renderLatex(results)); err != nil {
			log.Fatal(err)
		}
		if output != "-" {
			fmt.Fprintf(os.Stderr, "\nLaTeX table written to %s\n", output)
		}
	}
}

// ── Loading (same pattern as cmd/compare) ──────────────────────────────

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
		return nil, fmt.Errorf("expected %d scores in %s, got %d", len(samples), path, len(r.Scores))
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

func loadBaseline(samples []eval.Sample, dir, name string) (map[string][]float64, error) {
	out := make(map[string][]float64, len(dimensions))
	dimensional := true
	for _, dim := range dimensions {
		if _, err := os.Stat(filepath.Join(dir, name+"_"+dim+".json")); err != nil {
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

func groupByDoc(samples []eval.Sample) ([]string, map[string][]int) {
	byDoc := map[string][]int{}
	for i, s := range samples {
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], i)
	}
	docs := make([]string, 0, len(byDoc))
	for d := range byDoc {
		docs = append(docs, d)
	}
	sort.Strings(docs)
	return docs, byDoc
}

func pick(v []float64, idx []int) []float64 {
	out := make([]float64, len(idx))
	for i, p := range idx {
		out[i] = v[p]
	}
	return out
}

// ── Friedman + post-hoc ─────────────────────────────────────────────────

type dimResult struct {
	Dim      string
	Metrics  []string
	MeanRho  map[string]float64
	AvgRank  map[string]float64
	Chi2     float64
	DF       int
	P        float64
	N        int
	K        int
	CD       float64 // Nemenyi critical difference on average rank
	Target   string
	Pairwise []pairResult // target vs each other metric, Holm-corrected Wilcoxon
}

type pairResult struct {
	Metric string
	W      float64
	Z      float64
	P      float64
	PHolm  float64
	Sig    bool
}

// friedmanForDimension computes the Friedman chi-square statistic
// (ties handled via average ranks), then Wilcoxon signed-rank tests of
// `target` against every other metric on the same per-article rho
// values, Holm-corrected across the k-1 comparisons, plus the Nemenyi
// critical difference for the average-rank plot.
func friedmanForDimension(dim string, metrics []string, perArticleRho map[string][]float64, target string) dimResult {
	n := len(perArticleRho[metrics[0]])
	k := len(metrics)

	ranks := make(map[string][]float64, k)
	for _, m := range metrics {
		ranks[m] = make([]float64, n)
	}
	row := make([]float64, k)
	for a := 0; a < n; a++ {
		for mi, m := range metrics {
			row[mi] = perArticleRho[m][a]
		}
		r := rankAvg(row)
		for mi, m := range metrics {
			ranks[m][a] = r[mi]
		}
	}

	avgRank := make(map[string]float64, k)
	meanRho := make(map[string]float64, k)
	for _, m := range metrics {
		var sr, sv float64
		for a := 0; a < n; a++ {
			sr += ranks[m][a]
			sv += perArticleRho[m][a]
		}
		avgRank[m] = sr / float64(n)
		meanRho[m] = sv / float64(n)
	}

	// Friedman statistic on average ranks R_j:
	// chi2 = (12N / (k(k+1))) * (sum(R_j^2) - k(k+1)^2/4)
	var sumSq float64
	for _, m := range metrics {
		sumSq += avgRank[m] * avgRank[m]
	}
	chi2 := 12 * float64(n) / (float64(k) * (float64(k) + 1)) *
		(sumSq - float64(k)*(float64(k)+1)*(float64(k)+1)/4)

	df := k - 1
	p := 1 - chi2CDF(chi2, float64(df))

	// Nemenyi critical difference at alpha=0.05 (Demsar 2006).
	q := nemenyiQ005(k)
	cd := q * math.Sqrt(float64(k)*(float64(k)+1)/(6*float64(n)))

	pairwise := make([]pairResult, 0, k-1)
	for _, m := range metrics {
		if m == target {
			continue
		}
		w, z, pv := wilcoxonSignedRank(perArticleRho[target], perArticleRho[m])
		pairwise = append(pairwise, pairResult{Metric: m, W: w, Z: z, P: pv})
	}
	holmCorrect(pairwise)

	return dimResult{
		Dim: dim, Metrics: metrics, MeanRho: meanRho, AvgRank: avgRank,
		Chi2: chi2, DF: df, P: p, N: n, K: k, CD: cd, Target: target,
		Pairwise: pairwise,
	}
}

// rankAvg ranks a row ascending (1 = smallest rho), tie-averaged.
// Ascending so that a well-performing metric (high rho, hence
// low miss-rank under "smaller is better") gets a HIGH rank number --
// we rank ascending on rho itself, so higher rho -> higher rank.
func rankAvg(vals []float64) []float64 {
	n := len(vals)
	type iv struct {
		v float64
		i int
	}
	s := make([]iv, n)
	for i, v := range vals {
		s[i] = iv{v, i}
	}
	sort.Slice(s, func(a, b int) bool { return s[a].v < s[b].v })
	r := make([]float64, n)
	for i := 0; i < n; {
		j := i + 1
		for j < n && s[j].v == s[i].v {
			j++
		}
		avg := float64(i+j+1) / 2.0
		for x := i; x < j; x++ {
			r[s[x].i] = avg
		}
		i = j
	}
	return r
}

// wilcoxonSignedRank tests whether a-b has median 0, using the
// normal approximation with continuity and tie correction -- valid
// for N>=~20, which holds here (N=100 articles per dimension). Zero
// differences are dropped per the standard Wilcoxon convention.
func wilcoxonSignedRank(a, b []float64) (w, z, p float64) {
	n := len(a)
	type ad struct {
		absd float64
		sign float64
	}
	diffs := make([]ad, 0, n)
	for i := 0; i < n; i++ {
		d := a[i] - b[i]
		if d == 0 {
			continue
		}
		sign := 1.0
		if d < 0 {
			sign = -1.0
		}
		diffs = append(diffs, ad{math.Abs(d), sign})
	}
	nr := len(diffs)
	if nr == 0 {
		return 0, 0, 1
	}
	sort.Slice(diffs, func(i, j int) bool { return diffs[i].absd < diffs[j].absd })

	ranks := make([]float64, nr)
	tieCorrection := 0.0
	for i := 0; i < nr; {
		j := i + 1
		for j < nr && diffs[j].absd == diffs[i].absd {
			j++
		}
		avg := float64(i+j+1) / 2.0
		tcount := float64(j - i)
		tieCorrection += tcount*tcount*tcount - tcount
		for x := i; x < j; x++ {
			ranks[x] = avg
		}
		i = j
	}

	var wPlus float64
	for i, d := range diffs {
		if d.sign > 0 {
			wPlus += ranks[i]
		}
	}
	nf := float64(nr)
	mean := nf * (nf + 1) / 4
	variance := nf*(nf+1)*(2*nf+1)/24 - tieCorrection/48
	if variance <= 0 {
		return wPlus, 0, 1
	}
	sd := math.Sqrt(variance)
	// continuity correction toward the mean
	diff := wPlus - mean
	cc := 0.5
	if diff < 0 {
		cc = -0.5
	}
	zz := (diff - cc) / sd
	pv := 2 * (1 - normalCDF(math.Abs(zz)))
	if pv > 1 {
		pv = 1
	}
	return wPlus, zz, pv
}

func holmCorrect(pairs []pairResult) {
	m := len(pairs)
	order := make([]int, m)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return pairs[order[a]].P < pairs[order[b]].P })
	runningMax := 0.0
	for rank, idx := range order {
		adj := pairs[idx].P * float64(m-rank)
		if adj < runningMax {
			adj = runningMax
		}
		if adj > 1 {
			adj = 1
		}
		runningMax = adj
		pairs[idx].PHolm = adj
		pairs[idx].Sig = adj < 0.05
	}
}

// ── Special functions ────────────────────────────────────────────────

// chi2CDF is the regularized lower incomplete gamma P(df/2, x/2).
func chi2CDF(x, df float64) float64 {
	if x <= 0 {
		return 0
	}
	return lowerIncGamma(df/2, x/2)
}

// lowerIncGamma computes the regularized lower incomplete gamma
// function P(a,x) via series expansion (x < a+1) or continued
// fraction for the upper tail (x >= a+1), the standard split from
// Numerical Recipes.
func lowerIncGamma(a, x float64) float64 {
	if x < 0 || a <= 0 {
		return 0
	}
	if x == 0 {
		return 0
	}
	if x < a+1 {
		// series
		term := 1.0 / a
		sum := term
		ap := a
		for i := 0; i < 500; i++ {
			ap++
			term *= x / ap
			sum += term
			if math.Abs(term) < math.Abs(sum)*1e-14 {
				break
			}
		}
		return sum * math.Exp(-x+a*math.Log(x)-lgamma(a))
	}
	// continued fraction for Q(a,x), then P = 1 - Q
	b := x + 1 - a
	c := 1e300
	d := 1 / b
	h := d
	for i := 1; i < 500; i++ {
		an := -float64(i) * (float64(i) - a)
		b += 2
		d = an*d + b
		if math.Abs(d) < 1e-300 {
			d = 1e-300
		}
		c = b + an/c
		if math.Abs(c) < 1e-300 {
			c = 1e-300
		}
		d = 1 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < 1e-14 {
			break
		}
	}
	q := math.Exp(-x+a*math.Log(x)-lgamma(a)) * h
	return 1 - q
}

func lgamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}

func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}

// nemenyiQ005 returns the studentized-range critical value for the
// Nemenyi test at alpha=0.05, tabulated for k=2..20 (Demsar 2006,
// "Statistical Comparisons of Classifiers over Multiple Data Sets",
// JMLR, Table 5). Falls back to the k=20 value for larger k (the
// table is monotone increasing and near-flat by k=20).
func nemenyiQ005(k int) float64 {
	table := map[int]float64{
		2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
		8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
		14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
		20: 3.544,
	}
	if v, ok := table[k]; ok {
		return v
	}
	if k > 20 {
		return table[20]
	}
	return table[2]
}

// ── Rendering ────────────────────────────────────────────────────────

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

func display(m string) string {
	if v, ok := metricDisplayName[m]; ok {
		return v
	}
	return m
}

func renderConsole(results []dimResult) string {
	var b strings.Builder
	for _, r := range results {
		fmt.Fprintf(&b, "\n== %s ==\n", r.Dim)
		fmt.Fprintf(&b, "Friedman chi2=%.2f, df=%d, N=%d articles, k=%d metrics, p=%.4g\n",
			r.Chi2, r.DF, r.N, r.K, r.P)
		fmt.Fprintf(&b, "Nemenyi CD (alpha=0.05) on average rank: %.3f\n", r.CD)
		fmt.Fprintln(&b, "\nAverage rank (higher = more often better-correlated within article):")
		ms := append([]string(nil), r.Metrics...)
		sort.Slice(ms, func(i, j int) bool { return r.AvgRank[ms[i]] > r.AvgRank[ms[j]] })
		for _, m := range ms {
			fmt.Fprintf(&b, "  %-14s mean rho=%.3f  avg rank=%.2f\n", display(m), r.MeanRho[m], r.AvgRank[m])
		}
		fmt.Fprintf(&b, "\nWilcoxon signed-rank, %s vs each baseline (Holm-corrected):\n", display(r.Target))
		for _, pr := range r.Pairwise {
			sig := " "
			if pr.Sig {
				sig = "*"
			}
			fmt.Fprintf(&b, "  %-14s z=%+.2f  p=%.4g  p_holm=%.4g %s\n", display(pr.Metric), pr.Z, pr.P, pr.PHolm, sig)
		}
	}
	return b.String()
}

func renderLatex(results []dimResult) string {
	var b strings.Builder
	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintln(&b, `\caption{Friedman test (blocks = 100 \textsc{SummEval} articles, treatments = metrics, response = each metric's within-article Spearman correlation with human ratings) and Wilcoxon signed-rank post-hoc comparisons of LGS against every other metric, Holm-corrected across the family of comparisons within each dimension. $\chi^2_F$ is the Friedman statistic (df in parentheses); the omnibus test rejects the null of equal average rank across all four dimensions. The Nemenyi critical difference (CD, $\alpha=0.05$) is the minimum gap in average rank for two metrics to be called significantly different without a paired test. Wilcoxon columns report LGS's average-rank gap against the listed baseline and the Holm-adjusted $p$-value; \textbf{bold} survives $p_{\mathrm{Holm}}<0.05$.}`)
	fmt.Fprintln(&b, `\label{tab:friedman}`)
	fmt.Fprintln(&b, `\small`)
	fmt.Fprintln(&b, `\linespread{1}\selectfont`)
	fmt.Fprintln(&b, `\setlength{\tabcolsep}{4pt}`)

	for _, r := range results {
		fmt.Fprintf(&b, "\\subsubsection*{%s: $\\chi^2_F(%d){=}%.1f$, $p%s$, CD${=}%.2f$}\n",
			strings.Title(r.Dim), r.DF, r.Chi2, pfmt(r.P), r.CD)
		fmt.Fprintln(&b, `\begin{tabular}{@{}lrrrr@{}}`)
		fmt.Fprintln(&b, `\toprule`)
		fmt.Fprintln(&b, `Metric & Mean $\rho$ & Avg.\ rank & $z$ (vs LGS) & $p_{\mathrm{Holm}}$ \\`)
		fmt.Fprintln(&b, `\midrule`)

		ms := append([]string(nil), r.Metrics...)
		sort.Slice(ms, func(i, j int) bool { return r.AvgRank[ms[i]] > r.AvgRank[ms[j]] })
		pairByMetric := make(map[string]pairResult, len(r.Pairwise))
		for _, pr := range r.Pairwise {
			pairByMetric[pr.Metric] = pr
		}
		for _, m := range ms {
			zStr, pStr := "---", "---"
			if pr, ok := pairByMetric[m]; ok {
				zStr = fmt.Sprintf("%+.2f", pr.Z)
				pStr = fmt.Sprintf("%.3f", pr.PHolm)
				if pr.Sig {
					pStr = `\textbf{` + pStr + `}`
				}
			}
			name := display(m)
			if m == r.Target {
				name = `\textbf{` + name + `}`
			}
			fmt.Fprintf(&b, "%s & %.3f & %.2f & %s & %s \\\\\n", name, r.MeanRho[m], r.AvgRank[m], zStr, pStr)
		}
		fmt.Fprintln(&b, `\bottomrule`)
		fmt.Fprintln(&b, `\end{tabular}`)
		fmt.Fprintln(&b, `\vspace{2mm}`)
	}
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
}

func pfmt(p float64) string {
	if p < 0.001 {
		return "<.001"
	}
	return fmt.Sprintf("=%.3f", p)
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
