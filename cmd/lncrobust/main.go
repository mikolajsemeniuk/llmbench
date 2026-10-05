// cmd/lncrobust checks how much of LNC's held-out SummEval result is the
// canonical split, the systems it was fitted on, and the ridge protocol
// itself. It refits the frozen LNC recipe (pkg/metrics FitLNC, same α)
// from the feature dump on many random article splits and reports the
// copy-partialled Spearman ρ̄ on the held-out articles for:
//
//	random splits   50/50 article splits (canonical split is one draw)
//	system split    fit on 8 systems × 50 articles; test on the other 50
//	                articles, once on the same 8 systems and once on the
//	                other 8. UniEval (zero-shot) on the same subsets
//	                separates a harder subset from a system-specific fit.
//	controls        the same ridge over existing metrics' scores instead
//	                of the LNC features
//	feature groups  the LNC ridge refitted without one group of features
//
// Reads the feature dump and output/*.json only; no model calls.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
	"github.com/mikolajsemeniuk/llmbench/pkg/metrics"
)

var (
	featuresPath string
	inputDir     string
	output       string
	splits       int
	alpha        float64
	seed         uint64
)

var dims = []string{"coherence", "consistency", "fluency", "relevance"}

// groups are LNC's feature groups (indices into metrics.LNCFeatures),
// each dropped in turn to measure what it contributes.
var groups = []struct {
	name string
	cols []int
}{
	{"likelihood features", []int{0, 1, 2}},
	{"perturbation margins", []int{3, 4, 5}},
	{"NLI grounding", []int{6, 7}},
	{"splice-point likelihood", []int{8, 9}},
}

// controls are composites fitted with the LNC protocol over signals that
// already exist. "copy" and "len" are the copy rate and word count.
var controls = []struct{ name, signals string }{
	{"cheap reference-free signals", "copy,len,lead5sent,lgs,nlig"},
	{"existing metrics incl. reference-based", "copy,len,lgs,nlig,bertscore,gptscore,bartscore"},
	{"UniEval's four scorers", "unieval_coherence,unieval_consistency,unieval_fluency,unieval_relevance"},
}

func main() {
	flag.StringVar(&featuresPath, "features", "ablation/summeval/lnc_features.json", "LNC feature dump written by cmd/lnc")
	flag.StringVar(&inputDir, "input", "output/summeval", "directory with the metric reports used by the controls")
	flag.StringVar(&output, "output", "paper/lnc_robust.gen.tex", "LaTeX table (- for stdout)")
	flag.IntVar(&splits, "splits", 200, "random splits per experiment")
	flag.Float64Var(&alpha, "alpha", 10, "ridge penalty (as in cmd/lnc)")
	flag.Uint64Var(&seed, "seed", 42, "random seed")
	flag.Parse()

	samples, err := eval.LoadDataset(dataset.Default, 0)
	if err != nil {
		log.Fatal(err)
	}
	var docs []string
	for _, s := range samples {
		if len(docs) == 0 || docs[len(docs)-1] != s.DocumentID {
			docs = append(docs, s.DocumentID)
		}
	}
	copyRate := make([]float64, len(samples))
	human := map[string][]float64{}
	for i, s := range samples {
		copyRate[i] = metrics.CopyRate(s.Document, s.Candidate)
		human["coherence"] = append(human["coherence"], s.Coherence)
		human["consistency"] = append(human["consistency"], s.Consistency)
		human["fluency"] = append(human["fluency"], s.Fluency)
		human["relevance"] = append(human["relevance"], s.Relevance)
	}

	lnc, err := readFeatures(featuresPath, samples)
	if err != nil {
		log.Fatal(err)
	}
	ctrl := make([][][]float64, len(controls))
	for c, cs := range controls {
		ctrl[c], err = signals(strings.Split(cs.signals, ","), samples, copyRate)
		if err != nil {
			log.Fatal(err)
		}
	}
	without := make([][][]float64, len(groups))
	for g, gr := range groups {
		without[g] = dropCols(lnc, gr.cols)
	}
	uni := make(map[string][]float64, len(dims))
	for _, d := range dims {
		if uni[d], err = loadScores(samples, "unieval_"+d); err != nil {
			log.Fatal(err)
		}
	}

	// heldOut is the copy-partialled ρ̄ of a ridge fitted on train and
	// scored on test.
	heldOut := func(X [][]float64, train, test []int) float64 {
		Xt := pick(X, train)
		h := map[string][]float64{}
		for _, d := range dims {
			h[d] = pickF(human[d], train)
		}
		cal, err := metrics.FitLNC(Xt, h, alpha, "")
		if err != nil {
			log.Fatal(err)
		}
		var m float64
		for _, d := range dims {
			s := make([]float64, len(test))
			for i, j := range test {
				s[i], _ = cal.Score(d, X[j])
			}
			m += eval.PartialSpearman(s, pickF(human[d], test), pickF(copyRate, test)) / 4
		}
		return m
	}
	zeroShot := func(test []int) float64 {
		var m float64
		for _, d := range dims {
			m += eval.PartialSpearman(pickF(uni[d], test), pickF(human[d], test), pickF(copyRate, test)) / 4
		}
		return m
	}

	rng := rand.New(rand.NewPCG(seed, seed^0x5eed))
	var rnd []float64
	ctrlRnd := make([][]float64, len(controls))
	grpRnd := make([][]float64, len(groups))
	var seen, unseen, uSeen, uUnseen []float64
	for b := 0; b < splits; b++ {
		inTrain := map[string]bool{}
		for _, i := range rng.Perm(len(docs))[:len(docs)/2] {
			inTrain[docs[i]] = true
		}
		sysA := map[int]bool{}
		for _, i := range rng.Perm(16)[:8] {
			sysA[i] = true
		}
		var train, test, trainA, testA, testB []int
		for i, s := range samples {
			switch {
			case inTrain[s.DocumentID] && sysA[s.SystemID]:
				train, trainA = append(train, i), append(trainA, i)
			case inTrain[s.DocumentID]:
				train = append(train, i)
			case sysA[s.SystemID]:
				test, testA = append(test, i), append(testA, i)
			default:
				test, testB = append(test, i), append(testB, i)
			}
		}
		rnd = append(rnd, heldOut(lnc, train, test))
		for c := range controls {
			ctrlRnd[c] = append(ctrlRnd[c], heldOut(ctrl[c], train, test))
		}
		for g := range groups {
			grpRnd[g] = append(grpRnd[g], heldOut(without[g], train, test))
		}
		seen = append(seen, heldOut(lnc, trainA, testA))
		unseen = append(unseen, heldOut(lnc, trainA, testB))
		uSeen, uUnseen = append(uSeen, zeroShot(testA)), append(uUnseen, zeroShot(testB))
	}

	worse := 0
	for i := range seen {
		if unseen[i] < seen[i] {
			worse++
		}
	}

	type line struct {
		name string
		vals []float64
	}
	lines := []line{{"LNC, random article splits", rnd}}
	for c, cs := range controls {
		lines = append(lines, line{"Ridge on " + cs.name, ctrlRnd[c]})
	}
	lines = append(lines,
		line{"LNC, fit on 8 systems: same systems", seen},
		line{"LNC, fit on 8 systems: unseen systems", unseen},
		line{"UniEval (zero-shot): same systems", uSeen},
		line{"UniEval (zero-shot): unseen systems", uUnseen},
	)
	for g, gr := range groups {
		lines = append(lines, line{"LNC without " + gr.name, grpRnd[g]})
	}

	fmt.Printf("Held-out copy-partialled rho (mean over 4 dims), %d splits, features %s\n\n", splits, featuresPath)
	for _, l := range lines {
		m, lo, hi := summarise(l.vals)
		fmt.Printf("%-52s %.3f [%.3f, %.3f]\n", l.name, m, lo, hi)
	}
	fmt.Printf("\nunseen < seen systems in %d of %d splits\n", worse, splits)

	var b strings.Builder
	fmt.Fprintln(&b, `\begin{table}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintf(&b, "\\caption{Held-out copy-partialled $\\bar{\\rho}$ of the LNC ridge recipe on SummEval over %d random splits (mean and 2.5--97.5 percentiles). Controls apply the same ridge to existing signals; the last block refits LNC without one feature group. The system split fits on 8 of the 16 systems and 50 articles and tests on the other 50 articles; unseen systems score lower in %d of %d splits, while zero-shot UniEval does not change.}\n", splits, worse, splits)
	fmt.Fprintln(&b, `\label{tab:lnc_robust}`)
	fmt.Fprintln(&b, `\small`)
	fmt.Fprintln(&b, `\begin{tabular}{@{}lc@{}}`)
	fmt.Fprintln(&b, `\toprule`)
	fmt.Fprintln(&b, `Configuration & $\bar{\rho}_{\mathrm{part}}$ \\`)
	fmt.Fprintln(&b, `\midrule`)
	for i, l := range lines {
		if i == 1+len(controls) || i == 5+len(controls) {
			fmt.Fprintln(&b, `\midrule`)
		}
		m, lo, hi := summarise(l.vals)
		fmt.Fprintf(&b, "%s & %s {\\scriptsize [%s, %s]} \\\\\n", l.name, num(m), num(lo), num(hi))
	}
	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table}`)
	if output == "-" {
		fmt.Print(b.String())
		return
	}
	if err := os.WriteFile(output, []byte(b.String()), 0o644); err != nil {
		log.Fatal(err)
	}
}

func dropCols(X [][]float64, cols []int) [][]float64 {
	skip := map[int]bool{}
	for _, c := range cols {
		skip[c] = true
	}
	out := make([][]float64, len(X))
	for i, r := range X {
		for j, v := range r {
			if !skip[j] {
				out[i] = append(out[i], v)
			}
		}
	}
	return out
}

func summarise(v []float64) (mean, lo, hi float64) {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	for _, x := range s {
		mean += x / float64(len(s))
	}
	return mean, s[int(0.025*float64(len(s)-1))], s[int(0.975*float64(len(s)-1))]
}

func num(x float64) string {
	return strings.Replace(fmt.Sprintf("%.3f", x), "0.", ".", 1)
}

func pick(X [][]float64, idx []int) [][]float64 {
	out := make([][]float64, len(idx))
	for i, j := range idx {
		out[i] = X[j]
	}
	return out
}

func pickF(x []float64, idx []int) []float64 {
	out := make([]float64, len(idx))
	for i, j := range idx {
		out[i] = x[j]
	}
	return out
}

// signals builds a feature matrix from report names, "copy" and "len".
func signals(names []string, samples []eval.Sample, copyRate []float64) ([][]float64, error) {
	X := make([][]float64, len(samples))
	for _, n := range names {
		var col []float64
		switch n {
		case "copy":
			col = copyRate
		case "len":
			for _, s := range samples {
				col = append(col, float64(len(strings.Fields(s.Candidate))))
			}
		default:
			var err error
			if col, err = loadScores(samples, n); err != nil {
				return nil, err
			}
		}
		for i := range X {
			X[i] = append(X[i], col[i])
		}
	}
	return X, nil
}

func loadScores(samples []eval.Sample, name string) ([]float64, error) {
	b, err := os.ReadFile(filepath.Join(inputDir, name+".json"))
	if err != nil {
		return nil, err
	}
	var r eval.Report
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	byID := make(map[string]float64, len(r.Scores))
	for _, s := range r.Scores {
		byID[s.SampleID] = s.Value
	}
	out := make([]float64, len(samples))
	for i, s := range samples {
		v, ok := byID[s.ID]
		if !ok {
			return nil, fmt.Errorf("%s: missing %s", name, s.ID)
		}
		out[i] = v
	}
	return out, nil
}

func readFeatures(path string, samples []eval.Sample) ([][]float64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var d struct {
		Rows []struct {
			SampleID string    `json:"sample_id"`
			Features []float64 `json:"features"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(b, &d); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(d.Rows) != len(samples) {
		return nil, fmt.Errorf("%s: %d rows, expected %d", path, len(d.Rows), len(samples))
	}
	X := make([][]float64, len(samples))
	for i, r := range d.Rows {
		if r.SampleID != samples[i].ID {
			return nil, fmt.Errorf("%s: row %d is %s, expected %s", path, i, r.SampleID, samples[i].ID)
		}
		X[i] = r.Features
	}
	return X, nil
}
