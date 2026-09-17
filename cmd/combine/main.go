// cmd/combine evaluates rank-averaged combinations of metrics on the
// extractiveness-partialled axis.
//
// The motivating question is whether a metric adds signal that the pool
// does not already contain. Raw correlation cannot answer it: cmd/confound
// shows that adding any source-similarity signal to a combination raises
// raw rho simply by adding copy detection, while the partial axis does
// not move. A combination is therefore interesting only if it moves
// partial rho, and only if it does so on data that was not used to pick
// the combination -- hence -split, which restricts the evaluation to one
// half of the articles.
//
// Combination rule: per dimension, each member's per-sample scores are
// converted to ranks, the ranks are averaged, and the average is scored
// like any other metric. Rank averaging needs no weights, no training
// and no scale alignment, which keeps the comparison free of any
// fitting that would need its own held-out split.
//
// Reads the same output/*.json reports as cmd/confound. CPU only.
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

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
)

var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

var (
	datasetPath string
	inputDir    string
	comboSpec   string
	split       string
	ngram       int
	bootstrap   int
	seed        uint64
)

func main() {
	flag.StringVar(&inputDir, "input", "output", "directory containing metric JSON reports")
	flag.StringVar(&datasetPath, "dataset", "", "dataset JSONL path (empty = the embedded SummEval release); use with a converted corpus such as data/newsroom.jsonl")
	flag.StringVar(&comboSpec, "combos", "", "comma-separated combinations, members joined by +, e.g. \"lgs,cfm,unieval+cfm\"")
	flag.StringVar(&split, "split", "all", "article-level split to evaluate on: all|first50|last50")
	flag.IntVar(&ngram, "ngram", 2, "n-gram order for the copy rate")
	flag.IntVar(&bootstrap, "bootstrap", 2000, "cluster-bootstrap resamples over articles (0 = point estimates)")
	flag.Uint64Var(&seed, "seed", 42, "random seed")
	flag.Parse()

	if comboSpec == "" {
		log.Fatal("-combos is required, e.g. -combos \"cfm,unieval,unieval+cfm\"")
	}
	switch split {
	case "all", "first50", "last50":
	default:
		log.Fatalf("-split must be all|first50|last50, got %q", split)
	}

	samples, err := loadDataset(datasetPath)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}
	keep := splitMask(samples, split)

	metrics, err := loadMetrics(inputDir, samples)
	if err != nil {
		log.Fatal(err)
	}
	copyRate := subset(copyRates(samples, ngram), keep)
	human := map[string][]float64{}
	for d, v := range humanMatrix(samples) {
		human[d] = subset(v, keep)
	}

	fmt.Printf("Rank-averaged combinations on the partialled axis (split=%s, %d samples)\n\n", split, len(copyRate))
	fmt.Printf("%-34s %8s %8s %9s %9s\n", "combination", "raw", "partial", "rho_copy", "ms")
	fmt.Println(strings.Repeat("─", 72))

	for _, spec := range strings.Split(comboSpec, ",") {
		spec = strings.TrimSpace(spec)
		if spec == "" {
			continue
		}
		members := strings.Split(spec, "+")
		combined := map[string][]float64{}
		var ms float64
		ok := true
		for _, name := range members {
			m, found := metrics[strings.TrimSpace(name)]
			if !found {
				log.Printf("skipping %q: no report for %q in %s", spec, name, inputDir)
				ok = false
				break
			}
			ms += m.runtimeMs
			for _, d := range dimensions {
				combined[d] = addRanks(combined[d], subset(m.perDim[d], keep))
			}
		}
		if !ok {
			continue
		}

		var rawMean, partMean float64
		for _, d := range dimensions {
			rawMean += eval.Spearman(combined[d], human[d]) / 4
			partMean += partialSpearman(combined[d], human[d], copyRate) / 4
		}
		rhoCopy := eval.Spearman(combined["consistency"], copyRate)

		line := fmt.Sprintf("%-34s %8.3f %8.3f %9.3f %9.2f", spec, rawMean, partMean, rhoCopy, ms)
		if bootstrap > 0 {
			ci := bootstrapPartial(combined, human, copyRate, samples, keep)
			line += fmt.Sprintf("  [%.3f, %.3f]", ci.Low, ci.High)
		}
		fmt.Println(line)
	}
}

// addRanks accumulates the ranks of v into acc (element-wise sum of
// ranks, which is order-equivalent to the mean).
func addRanks(acc, v []float64) []float64 {
	r := ranks(v)
	if acc == nil {
		return r
	}
	for i := range acc {
		acc[i] += r[i]
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

func bootstrapPartial(combined, human map[string][]float64, copyRate []float64, samples []eval.Sample, keep []bool) eval.CI {
	var docIDs []string
	byDoc := map[string][]int{}
	k := 0
	for i, s := range samples {
		if !keep[i] {
			continue
		}
		if _, ok := byDoc[s.DocumentID]; !ok {
			docIDs = append(docIDs, s.DocumentID)
		}
		byDoc[s.DocumentID] = append(byDoc[s.DocumentID], k)
		k++
	}

	rng := rand.New(rand.NewPCG(seed, seed^0xc0ffee))
	vals := make([]float64, 0, bootstrap)
	for range bootstrap {
		var pick []int
		for range docIDs {
			pick = append(pick, byDoc[docIDs[rng.IntN(len(docIDs))]]...)
		}
		var total float64
		for _, d := range dimensions {
			bx := gather(combined[d], pick)
			by := gather(human[d], pick)
			bz := gather(copyRate, pick)
			total += partialSpearman(bx, by, bz)
		}
		vals = append(vals, total/4)
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

func gather(v []float64, idx []int) []float64 {
	out := make([]float64, len(idx))
	for i, p := range idx {
		out[i] = v[p]
	}
	return out
}

func subset(v []float64, keep []bool) []float64 {
	out := make([]float64, 0, len(v))
	for i, x := range v {
		if keep[i] {
			out = append(out, x)
		}
	}
	return out
}

func splitMask(samples []eval.Sample, split string) []bool {
	keep := make([]bool, len(samples))
	if split == "all" {
		for i := range keep {
			keep[i] = true
		}
		return keep
	}
	var order []string
	seen := map[string]bool{}
	for _, s := range samples {
		if !seen[s.DocumentID] {
			seen[s.DocumentID] = true
			order = append(order, s.DocumentID)
		}
	}
	half := len(order) / 2
	sel := order[:half]
	if split == "last50" {
		sel = order[half:]
	}
	in := map[string]bool{}
	for _, id := range sel {
		in[id] = true
	}
	for i, s := range samples {
		keep[i] = in[s.DocumentID]
	}
	return keep
}

// ── shared loading helpers (mirrors cmd/confound) ──────────────────────

type metricScores struct {
	perDim    map[string][]float64
	runtimeMs float64
}

func loadMetrics(dir string, samples []eval.Sample) (map[string]metricScores, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
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
			return nil, err
		}
		var r eval.Report
		if err := json.Unmarshal(raw, &r); err != nil {
			return nil, err
		}
		if len(r.Scores) != len(samples) {
			continue
		}
		vals, err := align(r, samples)
		if err != nil {
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
			a.isDim = true
			a.perDim[dim] = vals
			a.msTotal += ms
			continue
		}
		a.flat = vals
		a.flatMs = ms
	}

	out := map[string]metricScores{}
	for base, a := range groups {
		m := metricScores{perDim: map[string][]float64{}}
		if a.isDim {
			if len(a.perDim) != len(dimensions) {
				continue
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
		out[base] = m
	}
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

// loadDataset reads the embedded SummEval release by default, or an
// external JSONL corpus in the same shape -- see cmd/newsroom, which
// converts the Newsroom human-evaluation release into it.
func loadDataset(path string) ([]eval.Sample, error) {
	if path == "" {
		return eval.NewDataset(dataset.Summeval, dataset.SummevalDefaultPath, 0)
	}
	return eval.NewDataset(os.DirFS(filepath.Dir(path)), filepath.Base(path), 0)
}
