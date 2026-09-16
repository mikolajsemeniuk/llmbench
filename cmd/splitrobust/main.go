// cmd/splitrobust answers reviewer #3's m1: the dev/test split used
// to select the lead-bias exponent lambda* is "first 50 documents in
// JSONL order" / "last 50", which is unjustified relative to a
// random or stratified split. This binary re-runs the ENTIRE
// selection procedure (sweep the lambda grid on a dev half, take the
// argmax, verify on the complementary test half) under many random
// 50/50 article splits and reports how often the JSONL-order winner
// (lambda=0.5) is also the random-split winner, plus the distribution
// of test-side mean rho the procedure would have reported under each
// draw.
//
// Needs no embedder or model server: it merges the per-sample scores
// already stored in ablation/lgs_recall_{dev,test}.json and
// ablation/lgs_lead_{dev,test}_l*.json (nomic-embed-text, nondeg
// bootstrap=0 files) back into one 100-article dataset per lambda,
// then resplits that dataset directly.
//
// Usage:
//
//	go run ./cmd/splitrobust -input ablation -splits 2000 -output paper/splitrobust.gen.tex
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

func main() {
	var inputDir, output string
	var nSplits int
	var seed uint64
	var embed string
	flag.StringVar(&inputDir, "input", "ablation", "directory with LGS ablation JSON reports")
	flag.StringVar(&output, "output", "paper/splitrobust.gen.tex", "path to write LaTeX table (- for stdout)")
	flag.IntVar(&nSplits, "splits", 2000, "number of random 50/50 article splits to draw")
	flag.Uint64Var(&seed, "seed", 42, "random seed")
	flag.StringVar(&embed, "embed-model", "nomic-embed-text", "embedder tag to select from the ablation files")
	flag.Parse()

	samples, err := eval.NewDataset(dataset.Summeval, dataset.SummevalDefaultPath, 0)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	grid, err := loadGrid(inputDir, embed)
	if err != nil {
		log.Fatal(err)
	}
	if len(grid) < 2 {
		log.Fatalf("need at least 2 lambda values merged across dev+test, found %d", len(grid))
	}

	a, err := alignFull(samples, grid)
	if err != nil {
		log.Fatal(err)
	}

	// Reference point: the JSONL-order split the paper actually uses
	// (first 50 documents = dev, last 50 = test).
	jsonlDev, jsonlTest := a.docs[:50], a.docs[50:]
	jsonlWin, jsonlDevRho := argmaxOnDocs(a, jsonlDev)
	jsonlTestRho := meanRhoOnDocs(a, jsonlWin, jsonlTest)

	rng := rand.New(rand.NewPCG(seed, seed^0xd1e5e1ec7))
	winCounts := make([]int, len(a.lambdas))
	testRhoAtWin := make([]float64, 0, nSplits)
	matchesJSONL := 0

	perm := make([]int, len(a.docs))
	for i := range perm {
		perm[i] = i
	}

	for s := 0; s < nSplits; s++ {
		rng.Shuffle(len(perm), func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
		devIdx := append([]int(nil), perm[:len(perm)/2]...)
		testIdx := append([]int(nil), perm[len(perm)/2:]...)
		devDocs := docsAt(a.docs, devIdx)
		testDocs := docsAt(a.docs, testIdx)

		winIdx, _ := argmaxOnDocs(a, devDocs)
		winCounts[winIdx]++
		testRhoAtWin = append(testRhoAtWin, meanRhoOnDocs(a, winIdx, testDocs))
		if a.lambdas[winIdx] == a.lambdas[jsonlWin] {
			matchesJSONL++
		}
	}

	fmt.Print(renderConsole(a, winCounts, nSplits, jsonlWin, jsonlDevRho, jsonlTestRho, matchesJSONL, testRhoAtWin))
	if output != "" {
		if err := writeFile(output, renderLatex(a, winCounts, nSplits, jsonlWin, jsonlDevRho, jsonlTestRho, matchesJSONL, testRhoAtWin)); err != nil {
			log.Fatal(err)
		}
		if output != "-" {
			fmt.Fprintf(os.Stderr, "\nLaTeX table written to %s\n", output)
		}
	}
}

// ── Loading: merge dev+test ablation snapshots into one full-dataset grid ──

type variant struct {
	lambda float64
	byID   map[string]float64
}

func loadGrid(dir, embed string) ([]variant, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "lgs_*.json"))
	if err != nil {
		return nil, fmt.Errorf("glob: %w", err)
	}
	sort.Strings(matches)

	byLambda := map[float64]map[string]float64{}
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
		split := nm["split"]
		if split != "first50" && split != "last50" {
			continue // skip full-set / other snapshots
		}
		e := nm["embed_model"]
		if e == "" {
			e = "nomic-embed-text"
		}
		if stripTag(e) != embed {
			continue
		}
		lam, ok := parseFloat(nm["lead_lambda"])
		if !ok {
			continue
		}
		byID, ok := byLambda[lam]
		if !ok {
			byID = map[string]float64{}
			byLambda[lam] = byID
		}
		for _, sc := range r.Scores {
			byID[sc.SampleID] = sc.Value
		}
	}

	out := make([]variant, 0, len(byLambda))
	for lam, byID := range byLambda {
		out = append(out, variant{lambda: lam, byID: byID})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].lambda < out[j].lambda })
	return out, nil
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

func stripTag(s string) string {
	if i := strings.Index(s, ":"); i >= 0 {
		return s[:i]
	}
	return s
}

// ── Aligned view over the full (merged) 100-article dataset ──────────

type aligned struct {
	docs    []string
	byDoc   map[string][]int // sample positions per document
	human   [4][]float64     // indexed by sample position
	scores  [][]float64      // [lambda idx][sample position]
	lambdas []float64
}

func alignFull(samples []eval.Sample, grid []variant) (*aligned, error) {
	first := grid[0].byID
	idx := make([]int, 0, len(first))
	for i, s := range samples {
		if _, ok := first[s.ID]; ok {
			idx = append(idx, i)
		}
	}
	if len(idx) != len(first) {
		return nil, fmt.Errorf("merged grid covers %d samples but dataset matched %d", len(first), len(idx))
	}

	a := &aligned{byDoc: map[string][]int{}}
	docSeen := map[string]bool{}
	for pos, i := range idx {
		s := samples[i]
		a.byDoc[s.DocumentID] = append(a.byDoc[s.DocumentID], pos)
		if !docSeen[s.DocumentID] {
			docSeen[s.DocumentID] = true
			a.docs = append(a.docs, s.DocumentID)
		}
		a.human[0] = append(a.human[0], s.Coherence)
		a.human[1] = append(a.human[1], s.Consistency)
		a.human[2] = append(a.human[2], s.Fluency)
		a.human[3] = append(a.human[3], s.Relevance)
	}
	sort.Strings(a.docs)
	if len(a.docs) != 100 {
		return nil, fmt.Errorf("merged grid covers %d documents, expected 100", len(a.docs))
	}

	for _, v := range grid {
		col := make([]float64, len(idx))
		for pos, i := range idx {
			val, ok := v.byID[samples[i].ID]
			if !ok {
				return nil, fmt.Errorf("lambda=%g missing score for sample %s", v.lambda, samples[i].ID)
			}
			col[pos] = val
		}
		a.scores = append(a.scores, col)
		a.lambdas = append(a.lambdas, v.lambda)
	}
	return a, nil
}

func docsAt(docs []string, idx []int) []string {
	out := make([]string, len(idx))
	for i, p := range idx {
		out[i] = docs[p]
	}
	return out
}

func positionsOf(a *aligned, docs []string) []int {
	var out []int
	for _, d := range docs {
		out = append(out, a.byDoc[d]...)
	}
	return out
}

func meanRhoOnDocs(a *aligned, lambdaIdx int, docs []string) float64 {
	pick := positionsOf(a, docs)
	sc := a.scores[lambdaIdx]
	x := make([]float64, len(pick))
	for i, p := range pick {
		x[i] = sc[p]
	}
	var total float64
	for d := 0; d < 4; d++ {
		y := make([]float64, len(pick))
		for i, p := range pick {
			y[i] = a.human[d][p]
		}
		total += eval.Spearman(x, y)
	}
	return total / 4.0
}

func argmaxOnDocs(a *aligned, docs []string) (idx int, rho float64) {
	best, bestIdx := math.Inf(-1), 0
	for i := range a.lambdas {
		v := meanRhoOnDocs(a, i, docs)
		if v > best {
			best, bestIdx = v, i
		}
	}
	return bestIdx, best
}

// ── Rendering ──────────────────────────────────────────────────────

func renderConsole(a *aligned, winCounts []int, nSplits, jsonlWin int, jsonlDevRho, jsonlTestRho float64, matches int, testRho []float64) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Random-split robustness of lambda* selection (%d random 50/50 article splits, embedder %v-column merged grid)\n\n", nSplits, a.lambdas)
	fmt.Fprintf(&b, "JSONL-order split (paper's actual protocol): lambda*=%g, dev mean rho=%.4f, test mean rho=%.4f\n\n",
		a.lambdas[jsonlWin], jsonlDevRho, jsonlTestRho)
	fmt.Fprintf(&b, "%8s %14s\n", "lambda", "P(win | random split)")
	for i, c := range winCounts {
		fmt.Fprintf(&b, "%8g %13.1f%%\n", a.lambdas[i], 100*float64(c)/float64(nSplits))
	}
	fmt.Fprintf(&b, "\nP(random-split winner = JSONL-order winner lambda=%g): %.1f%%\n",
		a.lambdas[jsonlWin], 100*float64(matches)/float64(nSplits))
	mean, sd := meanSD(testRho)
	fmt.Fprintf(&b, "Test-side mean rho at the random split's own winner: mean=%.4f, sd=%.4f (JSONL test rho at JSONL winner: %.4f)\n",
		mean, sd, jsonlTestRho)
	return b.String()
}

func renderLatex(a *aligned, winCounts []int, nSplits, jsonlWin int, jsonlDevRho, jsonlTestRho float64, matches int, testRho []float64) string {
	var b strings.Builder
	mean, sd := meanSD(testRho)
	fmt.Fprintln(&b, `\begin{table}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintf(&b, `\caption{Robustness of the $\lambda^{\star}$ selection protocol to the choice of dev/test split. The manuscript uses the JSONL-emission-order split (first 50 documents = dev, last 50 = test); here the entire selection procedure --- sweep the grid on a dev half, take the argmax, score it on the complementary test half --- is re-run on %d independent random 50/50 article splits. $P(\mathrm{win})$ is the fraction of random splits whose dev-argmax lands on that grid value. The bottom row summarises the test-side mean $\rho$ the protocol would have reported under each random split's own winner, against the JSONL split's actual test result.}`+"\n", nSplits)
	fmt.Fprintln(&b, `\label{tab:splitrobust}`)
	fmt.Fprintln(&b, `\small`)
	fmt.Fprintln(&b, `\linespread{1}\selectfont`)
	fmt.Fprintln(&b, `\begin{tabular}{@{}lr@{}}`)
	fmt.Fprintln(&b, `\toprule`)
	fmt.Fprintln(&b, `$\lambda$ & $P(\mathrm{win}\mid\text{random split})$ \\`)
	fmt.Fprintln(&b, `\midrule`)
	for i, c := range winCounts {
		mark := ""
		if i == jsonlWin {
			mark = ` $\star$`
		}
		fmt.Fprintf(&b, "%g%s & %.1f\\%% \\\\\n", a.lambdas[i], mark, 100*float64(c)/float64(nSplits))
	}
	fmt.Fprintln(&b, `\midrule`)
	fmt.Fprintf(&b, `\multicolumn{2}{@{}l}{$P(\text{random winner}=\lambda^{\star}_{\mathrm{JSONL}}{=}%s)$: %.1f\%%} \\`+"\n",
		stripLeadingZero(a.lambdas[jsonlWin]), 100*float64(matches)/float64(nSplits))
	fmt.Fprintf(&b, `\multicolumn{2}{@{}l}{Test $\rho$ at random winner: mean %s, sd %s (JSONL test $\rho$: %s)} \\`+"\n",
		stripLeadingZero(mean), stripLeadingZero(sd), stripLeadingZero(jsonlTestRho))
	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table}`)
	return b.String()
}

func meanSD(v []float64) (mean, sd float64) {
	if len(v) == 0 {
		return 0, 0
	}
	for _, x := range v {
		mean += x
	}
	mean /= float64(len(v))
	for _, x := range v {
		sd += (x - mean) * (x - mean)
	}
	sd = math.Sqrt(sd / float64(len(v)))
	return
}

func stripLeadingZero(x float64) string {
	s := fmt.Sprintf("%.4f", x)
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
