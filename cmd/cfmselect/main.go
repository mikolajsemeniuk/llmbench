// cmd/cfmselect assigns counterfactual-margin signals to SummEval
// dimensions on the development articles and verifies the assignment on
// the held-out half.
//
// cmd/cfm emits a family of margins rather than a single score, because
// which corruption tracks which human dimension is an empirical
// question: an order-permutation margin might measure coherence, or it
// might measure fluency, and reading the answer off the same data that
// reports the correlation is exactly the forking-paths problem
// reviewers raised against the lambda sweep in the previous manuscript.
// So the assignment is a selection procedure with a protocol:
//
//  1. On the development articles (first half in dataset order), score
//     every signal against every dimension on the extractiveness-
//     partialled axis -- the axis the metric is designed for.
//  2. Per dimension, select the single best signal or the rank-average
//     of the best k (k <= -max-k), whichever maximises dev partial rho.
//     Rank averaging has no weights to fit.
//  3. Report the selected assignment on the held-out articles next to
//     its dev value, so shrinkage is visible.
//  4. With -bootstrap-select, re-run the selection inside a cluster
//     bootstrap over dev articles and report how often each signal
//     wins -- the treatment cmd/lambdaci gives the lambda sweep. A
//     selection that wins a third of resamples is an interval over
//     signals, not a fact.
//
// With -emit the selected per-dimension scores are written as four
// reports (<prefix>_{coherence,consistency,fluency,relevance}.json) so
// the metric enters the pool tables through cmd/paper and cmd/confound
// like any other dimensional scorer.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
)

var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

var (
	datasetPath   string
	inputDir      string
	prefix        string
	outputDir     string
	outputTex     string
	maxK          int
	ngram         int
	bootstrapSel  int
	emit          bool
	emitBootstrap int
	seed          uint64
	rawAxis       bool
	fixedSpec     string
	signed        bool
	randomSplits  int
)

// choice is one dimension's selected signal set.
type choice struct {
	Dimension string
	Members   []string
	Dev       float64
	Test      float64
	RhoCopy   float64
	Scores    []float64
}

func main() {
	flag.StringVar(&inputDir, "input", "output", "directory holding the cfm signal reports")
	flag.StringVar(&datasetPath, "dataset", "", "dataset JSONL path (empty = the embedded SummEval release); use with a converted corpus such as data/newsroom.jsonl")
	flag.StringVar(&prefix, "prefix", "cfm", "signal report prefix")
	flag.StringVar(&outputDir, "output-dir", "output", "where -emit writes the four per-dimension reports")
	flag.StringVar(&outputTex, "output", "", "write the selection table as LaTeX to this path (- for stdout)")
	flag.IntVar(&maxK, "max-k", 3, "largest rank-average combination considered per dimension")
	flag.IntVar(&ngram, "ngram", 2, "n-gram order of the copy rate that is partialled out")
	flag.IntVar(&bootstrapSel, "bootstrap-select", 0, "resamples for selection stability (0 = off)")
	flag.BoolVar(&emit, "emit", false, "write the selected per-dimension reports")
	flag.IntVar(&emitBootstrap, "emit-bootstrap", 1000, "bootstrap resamples for the emitted reports' CIs")
	flag.Uint64Var(&seed, "seed", 42, "random seed")
	flag.BoolVar(&rawAxis, "raw-axis", false, "select on raw rho instead of the partialled axis")
	flag.IntVar(&randomSplits, "random-splits", 0, "re-run selection under this many random 50/50 article splits and report the distribution of held-out performance (0 = off)")
	flag.BoolVar(&signed, "signed", true, "also consider the negation of each signal; the sign of a margin's relation to a dimension is part of the assignment and is selected on dev like everything else")
	flag.StringVar(&fixedSpec, "fixed", "", "skip selection and use a fixed assignment, e.g. \"coherence=disc_pmi,consistency=spec_sent_min,fluency=scram_margin,relevance=spec_summary\"")
	flag.Parse()

	samples, err := loadDataset(datasetPath)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}
	dev, test := splitMasks(samples)
	copyRate := copyRates(samples, ngram)
	human := humanMatrix(samples)

	sig, order, err := loadSignals(inputDir, prefix, samples)
	if err != nil {
		log.Fatal(err)
	}
	if signed {
		// A margin can relate to a dimension with either sign and the
		// direction is not always the obvious one: on SummEval the
		// discourse PMI is NEGATIVELY related to rated coherence,
		// because a summary whose next sentence is highly predictable
		// from the previous ones is usually redundant. Negations are
		// added as explicit candidates so that this is a selected,
		// reported choice rather than a silent absolute value.
		for _, name := range append([]string(nil), order...) {
			neg := make([]float64, len(sig[name]))
			for i, v := range sig[name] {
				neg[i] = -v
			}
			sig["-"+name] = neg
			order = append(order, "-"+name)
		}
	}
	if len(order) == 0 {
		log.Fatalf("no reports matching %s_* in %s", prefix, inputDir)
	}
	log.Printf("cfmselect: %d signals, axis=%s", len(order), axisName())

	score := func(v, h []float64, mask []bool) float64 {
		x, y, c := subset(v, mask), subset(h, mask), subset(copyRate, mask)
		if rawAxis {
			return eval.Spearman(x, y)
		}
		return partialSpearman(x, y, c)
	}

	fmt.Printf("\nSignal x dimension, %s rho (dev | test)\n\n", axisName())
	fmt.Printf("%-16s", "signal")
	for _, d := range dimensions {
		fmt.Printf("%18s", d)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("─", 16+18*len(dimensions)))
	for _, name := range order {
		fmt.Printf("%-16s", name)
		for _, d := range dimensions {
			fmt.Printf("%9.3f %8.3f", score(sig[name], human[d], dev), score(sig[name], human[d], test))
		}
		fmt.Println()
	}

	if fixedSpec != "" {
		// The pre-registered reading of the design: each corruption is
		// assigned to the dimension it was built for, with no data
		// consulted. Reported alongside the selected assignment so the
		// gain from selection is visible and cannot be mistaken for the
		// gain from the metric.
		fixed, err := parseFixed(fixedSpec, sig)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("\nFixed (pre-registered) assignment\n\n")
		fmt.Printf("%-13s %-46s %8s %8s %9s\n", "dimension", "signals", "dev", "test", "rho_copy")
		fmt.Println(strings.Repeat("─", 88))
		var fd, ft, fc float64
		for _, d := range dimensions {
			comb := rankAverage(sig, fixed[d])
			dv, tv := score(comb, human[d], dev), score(comb, human[d], test)
			rc := eval.Spearman(subset(comb, test), subset(copyRate, test))
			fmt.Printf("%-13s %-46s %8.3f %8.3f %9.3f\n", d, strings.Join(fixed[d], "+"), dv, tv, rc)
			fd, ft, fc = fd+dv/4, ft+tv/4, fc+rc/4
		}
		fmt.Println(strings.Repeat("─", 88))
		fmt.Printf("%-13s %-46s %8.3f %8.3f %9.3f\n", "mean", "", fd, ft, fc)
	}

	var choices []choice
	for _, d := range dimensions {
		ranked := append([]string(nil), order...)
		sort.SliceStable(ranked, func(a, b int) bool {
			return score(sig[ranked[a]], human[d], dev) > score(sig[ranked[b]], human[d], dev)
		})
		best := choice{Dimension: d, Dev: math.Inf(-1)}
		for k := 1; k <= min(maxK, len(ranked)); k++ {
			members := ranked[:k]
			comb := rankAverage(sig, members)
			devVal := score(comb, human[d], dev)
			if devVal > best.Dev {
				best = choice{
					Dimension: d,
					Members:   append([]string(nil), members...),
					Dev:       devVal,
					Test:      score(comb, human[d], test),
					RhoCopy:   eval.Spearman(subset(comb, test), subset(copyRate, test)),
					Scores:    comb,
				}
			}
		}
		choices = append(choices, best)
	}

	fmt.Printf("\nSelected assignment (dev argmax over k <= %d, verified on test)\n\n", maxK)
	fmt.Printf("%-13s %-46s %8s %8s %9s\n", "dimension", "signals", "dev", "test", "rho_copy")
	fmt.Println(strings.Repeat("─", 88))
	var devMean, testMean, copyMean float64
	for _, c := range choices {
		fmt.Printf("%-13s %-46s %8.3f %8.3f %9.3f\n", c.Dimension,
			strings.Join(c.Members, "+"), c.Dev, c.Test, c.RhoCopy)
		devMean += c.Dev / 4
		testMean += c.Test / 4
		copyMean += c.RhoCopy / 4
	}
	fmt.Println(strings.Repeat("─", 88))
	fmt.Printf("%-13s %-46s %8.3f %8.3f %9.3f\n", "mean", "", devMean, testMean, copyMean)

	if bootstrapSel > 0 {
		fmt.Printf("\nSelection stability: P(signal is the dev argmax), %d article resamples\n\n", bootstrapSel)
		rng := rand.New(rand.NewPCG(seed, seed^0xabcdef))
		byDoc, docs := docIndex(samples, dev)
		for _, d := range dimensions {
			counts := map[string]int{}
			for range bootstrapSel {
				var pick []int
				for range docs {
					pick = append(pick, byDoc[docs[rng.IntN(len(docs))]]...)
				}
				bestName, bestVal := "", math.Inf(-1)
				for _, name := range order {
					x, y, z := gather(sig[name], pick), gather(human[d], pick), gather(copyRate, pick)
					v := partialSpearman(x, y, z)
					if rawAxis {
						v = eval.Spearman(x, y)
					}
					if v > bestVal {
						bestName, bestVal = name, v
					}
				}
				counts[bestName]++
			}
			type kv struct {
				name string
				n    int
			}
			var top []kv
			for k, v := range counts {
				top = append(top, kv{k, v})
			}
			sort.Slice(top, func(a, b int) bool { return top[a].n > top[b].n })
			fmt.Printf("%-13s", d)
			for _, t := range top[:min(4, len(top))] {
				fmt.Printf("  %s %.1f%%", t.name, 100*float64(t.n)/float64(bootstrapSel))
			}
			fmt.Println()
		}
	}

	if randomSplits > 0 {
		// The dataset-order split is arbitrary (reviewer #3 m1). Under
		// random article splits, selection and verification are redone
		// from scratch, which separates "this assignment generalises"
		// from "this assignment fits the first fifty articles".
		rng := rand.New(rand.NewPCG(seed, seed^0x515))
		byDocAll, allDocs := docIndex(samples, allTrue(len(samples)))
		vals := make([]float64, 0, randomSplits)
		chosen := map[string]map[string]int{}
		for _, d := range dimensions {
			chosen[d] = map[string]int{}
		}
		for range randomSplits {
			perm := rng.Perm(len(allDocs))
			devMask := make([]bool, len(samples))
			testMask := make([]bool, len(samples))
			for k, p := range perm {
				m := devMask
				if k >= len(allDocs)/2 {
					m = testMask
				}
				for _, i := range byDocAll[allDocs[p]] {
					m[i] = true
				}
			}
			var total float64
			for _, d := range dimensions {
				ranked := append([]string(nil), order...)
				sort.SliceStable(ranked, func(a, b int) bool {
					return score(sig[ranked[a]], human[d], devMask) > score(sig[ranked[b]], human[d], devMask)
				})
				bestDev, bestTest, bestMembers := math.Inf(-1), 0.0, []string(nil)
				for k := 1; k <= min(maxK, len(ranked)); k++ {
					comb := rankAverage(sig, ranked[:k])
					dv := score(comb, human[d], devMask)
					if dv > bestDev {
						bestDev, bestTest, bestMembers = dv, score(comb, human[d], testMask), ranked[:k]
					}
				}
				chosen[d][strings.Join(bestMembers, "+")]++
				total += bestTest / 4
			}
			vals = append(vals, total)
		}
		sort.Float64s(vals)
		at := func(p float64) float64 { return vals[int(p*float64(len(vals)-1))] }
		fmt.Printf("\nRandom-split robustness of the whole procedure (%d splits)\n\n", randomSplits)
		fmt.Printf("held-out mean partial rho: median %.3f, 90%% interval [%.3f, %.3f]\n",
			at(0.5), at(0.05), at(0.95))
		for _, d := range dimensions {
			type kv struct {
				k string
				n int
			}
			var top []kv
			for k, v := range chosen[d] {
				top = append(top, kv{k, v})
			}
			sort.Slice(top, func(a, b int) bool { return top[a].n > top[b].n })
			fmt.Printf("%-13s most-selected: ", d)
			for _, t := range top[:min(3, len(top))] {
				fmt.Printf("%s %.0f%%  ", t.k, 100*float64(t.n)/float64(randomSplits))
			}
			fmt.Println()
		}
	}

	if outputTex != "" {
		if err := writeFile(outputTex, renderLatex(choices, devMean, testMean, copyMean)); err != nil {
			log.Fatal(err)
		}
	}

	if emit {
		runtime := passRuntime(inputDir, prefix)
		for _, c := range choices {
			name := fmt.Sprintf("%s_%s", prefix, c.Dimension)
			report := eval.Report{
				Metric:     name,
				Norm:       "rank-average",
				Samples:    len(samples),
				RuntimeSec: runtime / 4, // one pass serves all four dimensions
				Timestamp:  time.Now().UTC().Format(time.RFC3339),
				Scores:     scoreEntries(samples, c.Scores),
				SummaryLevel: eval.NewCorrelation(samples, c.Scores, eval.CorrelationOptions{
					Bootstrap: emitBootstrap, Level: "summary",
				}),
				SystemLevel: eval.NewCorrelation(samples, c.Scores, eval.CorrelationOptions{
					Bootstrap: emitBootstrap, Level: "system",
				}),
			}
			out := filepath.Join(outputDir, name+".json")
			if err := eval.NewReport(out, report); err != nil {
				log.Fatal(err)
			}
			log.Printf("emitted %s (%s)", out, strings.Join(c.Members, "+"))
		}
	}
}

// parseFixed reads "dim=sig+sig,dim=sig" into an assignment, checking
// that every named signal exists.
func parseFixed(spec string, sig map[string][]float64) (map[string][]string, error) {
	out := map[string][]string{}
	for _, part := range strings.Split(spec, ",") {
		kv := strings.SplitN(strings.TrimSpace(part), "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("cfmselect: cannot parse %q as dimension=signal", part)
		}
		if !slicesContains(dimensions, kv[0]) {
			return nil, fmt.Errorf("cfmselect: unknown dimension %q", kv[0])
		}
		for _, m := range strings.Split(kv[1], "+") {
			if _, ok := sig[m]; !ok {
				return nil, fmt.Errorf("cfmselect: unknown signal %q", m)
			}
			out[kv[0]] = append(out[kv[0]], m)
		}
	}
	for _, d := range dimensions {
		if len(out[d]) == 0 {
			return nil, fmt.Errorf("cfmselect: -fixed does not cover %s", d)
		}
	}
	return out, nil
}

func allTrue(n int) []bool {
	out := make([]bool, n)
	for i := range out {
		out[i] = true
	}
	return out
}

func axisName() string {
	if rawAxis {
		return "raw"
	}
	return "partial"
}

// rankAverage converts each member to ranks and sums them, which is
// order-equivalent to averaging and needs no weights or scaling.
func rankAverage(sig map[string][]float64, members []string) []float64 {
	var acc []float64
	for _, m := range members {
		r := ranks(sig[m])
		if acc == nil {
			acc = r
			continue
		}
		for i := range acc {
			acc[i] += r[i]
		}
	}
	return acc
}

func ranks(v []float64) []float64 {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return v[idx[a]] < v[idx[b]] })
	out := make([]float64, len(v))
	for i := 0; i < len(idx); {
		j := i
		for j+1 < len(idx) && v[idx[j+1]] == v[idx[i]] {
			j++
		}
		avg := float64(i+j)/2 + 1
		for k := i; k <= j; k++ {
			out[idx[k]] = avg
		}
		i = j + 1
	}
	return out
}

func partialSpearman(x, y, z []float64) float64 {
	rxy, rxz, ryz := eval.Spearman(x, y), eval.Spearman(x, z), eval.Spearman(y, z)
	den := math.Sqrt((1 - rxz*rxz) * (1 - ryz*ryz))
	if den == 0 {
		return 0
	}
	return (rxy - rxz*ryz) / den
}

func loadSignals(dir, prefix string, samples []eval.Sample) (map[string][]float64, []string, error) {
	paths, err := filepath.Glob(filepath.Join(dir, prefix+"_*.json"))
	if err != nil {
		return nil, nil, err
	}
	sort.Strings(paths)
	out := map[string][]float64{}
	var order []string
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil, nil, err
		}
		var r eval.Report
		if err := json.Unmarshal(raw, &r); err != nil {
			continue // the cost sidecar is not a report
		}
		if len(r.Scores) != len(samples) {
			continue
		}
		name := strings.TrimPrefix(r.Metric, prefix+"_")
		if slicesContains(dimensions, name) {
			continue // do not select over a previous -emit
		}
		vals, err := align(r, samples)
		if err != nil {
			continue
		}
		out[name] = vals
		order = append(order, name)
	}
	return out, order, nil
}

func passRuntime(dir, prefix string) float64 {
	raw, err := os.ReadFile(filepath.Join(dir, prefix+"_cost.json"))
	if err != nil {
		return 0
	}
	var c struct {
		RuntimeSec float64 `json:"runtime_sec"`
	}
	if json.Unmarshal(raw, &c) != nil {
		return 0
	}
	return c.RuntimeSec
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

func humanMatrix(samples []eval.Sample) map[string][]float64 {
	get := map[string]func(eval.Sample) float64{
		"coherence":   func(s eval.Sample) float64 { return s.Coherence },
		"consistency": func(s eval.Sample) float64 { return s.Consistency },
		"fluency":     func(s eval.Sample) float64 { return s.Fluency },
		"relevance":   func(s eval.Sample) float64 { return s.Relevance },
	}
	out := map[string][]float64{}
	for _, d := range dimensions {
		v := make([]float64, len(samples))
		for i, s := range samples {
			v[i] = get[d](s)
		}
		out[d] = v
	}
	return out
}

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

// splitMasks reproduces the dataset-order 50/50 article split used
// throughout the repository. cmd/splitrobust exists because that split
// is arbitrary; the same selection can be re-run under other splits.
func splitMasks(samples []eval.Sample) (dev, test []bool) {
	var order []string
	seen := map[string]bool{}
	for _, s := range samples {
		if !seen[s.DocumentID] {
			seen[s.DocumentID] = true
			order = append(order, s.DocumentID)
		}
	}
	half := len(order) / 2
	inDev := map[string]bool{}
	for _, id := range order[:half] {
		inDev[id] = true
	}
	dev = make([]bool, len(samples))
	test = make([]bool, len(samples))
	for i, s := range samples {
		dev[i] = inDev[s.DocumentID]
		test[i] = !inDev[s.DocumentID]
	}
	return dev, test
}

func docIndex(samples []eval.Sample, mask []bool) (map[string][]int, []string) {
	byDoc := map[string][]int{}
	var docs []string
	for i, s := range samples {
		if !mask[i] {
			continue
		}
		if _, ok := byDoc[s.DocumentID]; !ok {
			docs = append(docs, s.DocumentID)
		}
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], i)
	}
	return byDoc, docs
}

func subset(v []float64, mask []bool) []float64 {
	out := make([]float64, 0, len(v))
	for i, x := range v {
		if mask[i] {
			out = append(out, x)
		}
	}
	return out
}

func gather(v []float64, idx []int) []float64 {
	out := make([]float64, len(idx))
	for i, p := range idx {
		out[i] = v[p]
	}
	return out
}

func scoreEntries(samples []eval.Sample, vals []float64) []eval.Score {
	out := make([]eval.Score, len(samples))
	for i, s := range samples {
		out[i] = eval.Score{SampleID: s.ID, Value: vals[i]}
	}
	return out
}

func slicesContains(hay []string, needle string) bool {
	for _, h := range hay {
		if h == needle {
			return true
		}
	}
	return false
}

func renderLatex(choices []choice, devMean, testMean, copyMean float64) string {
	var b strings.Builder
	b.WriteString("\\begin{table}[t]\n\\centering\n")
	b.WriteString("\\caption{Counterfactual-margin signals assigned to \\textsc{SummEval} dimensions. ")
	b.WriteString("The assignment is the argmax of development partial $\\rho$ over single signals and rank-averages of up to three; ")
	b.WriteString("the test column is the held-out half, never used for selection. ")
	b.WriteString("$\\rho_{\\mathrm{copy}}$ is the correlation of the selected score with the bigram copy rate on the test half.}\n")
	b.WriteString("\\label{tab:cfm_selection}\n\\small\n")
	b.WriteString("\\begin{tabular}{@{}llrrr@{}}\n\\toprule\n")
	b.WriteString("Dimension & Selected signals & Dev $\\rho_{\\mathrm{part}}$ & Test $\\rho_{\\mathrm{part}}$ & $\\rho_{\\mathrm{copy}}$ \\\\\n\\midrule\n")
	for _, c := range choices {
		b.WriteString(fmt.Sprintf("%s & %s & %.3f & %.3f & %.3f \\\\\n",
			c.Dimension, strings.ReplaceAll(strings.Join(c.Members, "+"), "_", "\\_"),
			c.Dev, c.Test, c.RhoCopy))
	}
	b.WriteString("\\midrule\n")
	b.WriteString(fmt.Sprintf("Mean & & %.3f & %.3f & %.3f \\\\\n", devMean, testMean, copyMean))
	b.WriteString("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
	return b.String()
}

func writeFile(path, content string) error {
	if path == "-" {
		fmt.Print(content)
		return nil
	}
	return os.WriteFile(path, []byte(content), 0o644)
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
