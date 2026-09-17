// cmd/cfm — counterfactual-margin signals for reference-free summary
// evaluation.
//
// WHY THIS EXISTS. cmd/confound shows that mean summary-level
// correlation on SummEval is largely a measure of extractiveness: a
// bare bigram-copy counter reaches mean Spearman rho .275, outranking
// most of the published pool, and every metric built out of
// candidate-source similarity -- lexical, positional or embedding --
// plateaus at partial rho ~= .23 once the copy rate is partialled out.
// Similarity to the source is a confounded signal class: a summary that
// copies scores like a summary that is good.
//
// CFM changes the question asked of the model. Instead of "how similar
// is the candidate to the source?" it asks "does a frozen language
// model prefer this candidate over a minimally corrupted version of
// itself, or over the same candidate read against a different source?":
//
//	margin = s(C, D) − E_{C' ~ T(C)} [ s(C', D) ]
//
// with s a length-normalised teacher-forced log-probability under a
// small causal LM and T a corruption that preserves the candidate's
// lexical overlap with the source.
//
// DECORRELATION BY CONSTRUCTION. Every candidate-side corruption either
// permutes the candidate's own tokens (order, word order) or replaces a
// mention with another mention taken out of the same source (entities),
// so C' carries essentially the same verbatim overlap with D as C. A
// metric that only counts copied n-grams therefore scores C and C'
// identically and its margin is exactly zero: extractiveness cannot
// enter a margin the way it enters a similarity. The source-side
// counterfactual (score C against a different article) is decorrelated
// for the same reason in reverse -- what survives the subtraction is
// only the part of the candidate that is specific to THIS source.
// Measured, over the full pool: rho_copy is .39 for LGS, .68 for the
// lead-window baseline, and |rho_copy| < .2 for every margin below.
//
// WHAT THIS BINARY WRITES. Not one score but a family of margin
// signals, one report per signal, because which margin tracks which
// human dimension is an empirical question and must not be settled on
// the same data that reports the answer. cmd/cfmselect does the
// assignment on the development articles and verifies it on the
// held-out half.
//
//	perm_margin     s(C) − mean s(sentence-permuted C)          [order]
//	perm_winrate    fraction of permutations scoring below C
//	scram_margin    s(C) − mean s(word-order-corrupted C)       [local syntax]
//	scram_winrate   fraction of corruptions scoring below C
//	swap_margin     s(C|D) − mean s(entity-swapped C|D)         [factual alignment]
//	spec_summary    s(C|D) − mean over distractors s(C|D')      [source specificity]
//	spec_sent_mean  same, computed per candidate sentence, averaged
//	spec_sent_min   same, minimum over candidate sentences      [worst claim]
//	lm_logprob      s(C), no counterfactual                     [plain fluency]
//	lm_logprob_min  the worst candidate sentence's s             [worst sentence]
//	disc_pmi        mean_j [ s(sent_j | sent_<j) − s(sent_j) ]   [discourse]
//	                how much the preceding summary sentences help
//	                predict the next one: a summary whose sentences
//	                do not cohere gains nothing from its own context.
//
// COST. One pass per article: one unconditional batch (candidates plus
// their permutation and scramble corruptions), one batch conditioned on
// the article, one batch per distractor article. Conditioning encodes
// the article once per batch and reuses it across the 16 candidates of
// that article, so an article is read once per batch rather than once
// per (candidate, corruption) pair. Runtime and a hardware-independent
// token/sequence count are both recorded; see cfm_cost.json.
//
// Needs cmd/lmsrv (default http://localhost:9300). No reference, no
// Ollama, no judge LLM, and no tuned hyperparameter: -perturb and
// -distractors trade cost against variance and are ablated, not fitted.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
	"github.com/mikolajsemeniuk/llmbench/pkg/metrics"
	"github.com/schollz/progressbar/v3"
)

// signals is the order in which reports are written; it is also the
// order cmd/cfmselect displays.
var signals = []string{
	"perm_margin", "perm_winrate",
	"scram_margin", "scram_winrate",
	"swap_margin",
	"spec_summary", "spec_sent_mean", "spec_sent_min",
	"lm_logprob", "lm_logprob_min",
	"disc_pmi",
}

var (
	input       string
	outputDir   string
	prefix      string
	server      string
	nPerturb    int
	nDistract   int
	minSentLen  int
	distOffsets string
	n           int
	bootstrap   int
	seed        uint64
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&outputDir, "output-dir", "output", "directory for the per-signal reports")
	flag.StringVar(&prefix, "prefix", "cfm", "metric-name prefix, e.g. -prefix cfm17 for a second backbone")
	flag.StringVar(&server, "server", "http://localhost:9300", "cmd/lmsrv host")
	flag.IntVar(&nPerturb, "perturb", 4, "corruptions sampled per candidate-side family")
	flag.IntVar(&nDistract, "distractors", 2, "distractor articles per candidate for the source-side margin")
	flag.IntVar(&minSentLen, "min-sent-len", 4, "drop candidate sentences shorter than this many runes")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Uint64Var(&seed, "seed", 42, "random seed for the corruption sampler")
	flag.Parse()

	if nPerturb < 1 {
		log.Fatalf("-perturb must be >= 1, got %d", nPerturb)
	}
	if nDistract < 1 {
		log.Fatalf("-distractors must be >= 1, got %d", nDistract)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	fsys := os.DirFS(filepath.Dir(input))
	path := filepath.Base(input)
	if input == "" {
		fsys = dataset.Summeval
		path = dataset.SummevalDefaultPath
	}
	samples, err := eval.NewDataset(fsys, path, n)
	if err != nil {
		log.Fatal(err)
	}

	docs := groupByDocument(samples)
	log.Printf("cfm: %d samples, %d articles, %d corruptions, %d distractors",
		len(samples), len(docs), nPerturb, nDistract)

	lm := metrics.NewLMServer(server)
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b9))

	feat := map[string][]float64{}
	for _, s := range signals {
		feat[s] = make([]float64, len(samples))
	}
	cost := &costAccount{}
	var noSwap int
	start := time.Now()

	bar := progressbar.NewOptions(len(docs),
		progressbar.OptionSetDescription("cfm"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionSetWriter(os.Stderr),
	)

	for di, doc := range docs {
		cands := make([][]string, len(doc.Idx))
		for k, i := range doc.Idx {
			cands[k] = splitCandidate(samples[i].Candidate, minSentLen)
		}

		// ── unconditional batch: C, permutations, scrambles, sentences ──
		// Layout per candidate: the candidate itself, its permutation
		// and scramble corruptions, then its cumulative sentence
		// prefixes and its sentences in isolation. The last two give
		// the discourse signal: subtracting the sum-log-probability of
		// prefix j-1 from prefix j isolates sentence j read in context,
		// and the isolated sentence is the same span read without it.
		type span struct{ base, nPerm, nScram, prefBase, sentBase, nSent int }
		var uncond []string
		lay := make([]span, len(doc.Idx))
		for k := range doc.Idx {
			perms := metrics.Permute(cands[k], nPerturb, rng)
			scrams := metrics.Scramble(cands[k], nPerturb, rng)
			sp := span{base: len(uncond), nPerm: len(perms), nScram: len(scrams), nSent: len(cands[k])}
			uncond = append(uncond, strings.Join(cands[k], " "))
			uncond = append(uncond, perms...)
			uncond = append(uncond, scrams...)
			sp.prefBase = len(uncond)
			for j := 1; j <= len(cands[k]); j++ {
				uncond = append(uncond, strings.Join(cands[k][:j], " "))
			}
			sp.sentBase = len(uncond)
			uncond = append(uncond, cands[k]...)
			lay[k] = sp
		}
		udet, err := lm.ScoreDetail(ctx, "", uncond)
		if err != nil {
			log.Fatalf("article %s: unconditional batch: %v", doc.ID, err)
		}
		cost.add(udet)

		// ── conditional batch: C, entity swaps, and C's sentences, given D ──
		type cspan struct{ base, nSwap, sentBase, nSent int }
		pool := metrics.NewEntityPool(doc.Document)
		var cond []string
		clay := make([]cspan, len(doc.Idx))
		for k := range doc.Idx {
			whole := strings.Join(cands[k], " ")
			swaps := metrics.SwapEntities(whole, pool, nPerturb, rng)
			if len(swaps) == 0 {
				noSwap++
			}
			c := cspan{base: len(cond), nSwap: len(swaps)}
			cond = append(cond, whole)
			cond = append(cond, swaps...)
			c.sentBase = len(cond)
			c.nSent = len(cands[k])
			cond = append(cond, cands[k]...)
			clay[k] = c
		}
		cdet, err := lm.ScoreDetail(ctx, conditioning(doc.Document), cond)
		if err != nil {
			log.Fatalf("article %s: conditional batch: %v", doc.ID, err)
		}
		cost.add(cdet)

		// ── distractor batches: the same targets, other articles ──
		// Distractors are chosen deterministically at fixed strides so
		// the control is reproducible and no candidate is ever scored
		// against its own source.
		dsum := make([]float64, len(doc.Idx))
		dsent := make([][]float64, len(doc.Idx))
		for k := range dsent {
			dsent[k] = make([]float64, len(cands[k]))
		}
		for r := 1; r <= nDistract; r++ {
			dd := docs[(di+r*(len(docs)/(nDistract+1)+1))%len(docs)]
			if dd.ID == doc.ID {
				dd = docs[(di+r)%len(docs)]
			}
			var tgt []string
			type dspan struct{ base, nSent int }
			dl := make([]dspan, len(doc.Idx))
			for k := range doc.Idx {
				dl[k] = dspan{len(tgt), len(cands[k])}
				tgt = append(tgt, strings.Join(cands[k], " "))
				tgt = append(tgt, cands[k]...)
			}
			ddet, err := lm.ScoreDetail(ctx, conditioning(dd.Document), tgt)
			if err != nil {
				log.Fatalf("article %s: distractor batch: %v", doc.ID, err)
			}
			cost.add(ddet)
			for k := range doc.Idx {
				dsum[k] += ddet.MeanLogprob[dl[k].base] / float64(nDistract)
				for j := range dsent[k] {
					dsent[k][j] += ddet.MeanLogprob[dl[k].base+1+j] / float64(nDistract)
				}
			}
		}

		// ── assemble the signals ──
		for k, i := range doc.Idx {
			l, c := lay[k], clay[k]
			base := udet.MeanLogprob[l.base]
			perms := udet.MeanLogprob[l.base+1 : l.base+1+l.nPerm]
			scrams := udet.MeanLogprob[l.base+1+l.nPerm : l.base+1+l.nPerm+l.nScram]

			feat["lm_logprob"][i] = base

			// Worst-sentence fluency and the discourse PMI, both read
			// off the prefix/sentence spans of the same batch.
			worstLP := 0.0
			var pmiSum float64
			var pmiN int
			for j := 0; j < l.nSent; j++ {
				alone := udet.MeanLogprob[l.sentBase+j]
				if j == 0 || alone < worstLP {
					worstLP = alone
				}
				if j == 0 {
					continue // no preceding context to help
				}
				condSum := udet.SumLogprob[l.prefBase+j] - udet.SumLogprob[l.prefBase+j-1]
				condTok := udet.Tokens[l.prefBase+j] - udet.Tokens[l.prefBase+j-1]
				if condTok <= 0 {
					continue
				}
				pmiSum += condSum/float64(condTok) - alone
				pmiN++
			}
			feat["lm_logprob_min"][i] = worstLP
			if pmiN > 0 {
				feat["disc_pmi"][i] = pmiSum / float64(pmiN)
			}
			feat["perm_margin"][i] = margin(base, perms)
			feat["perm_winrate"][i] = winRate(base, perms)
			feat["scram_margin"][i] = margin(base, scrams)
			feat["scram_winrate"][i] = winRate(base, scrams)

			condBase := cdet.MeanLogprob[c.base]
			swaps := cdet.MeanLogprob[c.base+1 : c.base+1+c.nSwap]
			feat["swap_margin"][i] = margin(condBase, swaps)
			feat["spec_summary"][i] = condBase - dsum[k]

			var sum, worst float64
			worst = 0
			for j := 0; j < c.nSent; j++ {
				d := cdet.MeanLogprob[c.sentBase+j] - dsent[k][j]
				sum += d
				if j == 0 || d < worst {
					worst = d
				}
			}
			if c.nSent > 0 {
				feat["spec_sent_mean"][i] = sum / float64(c.nSent)
				feat["spec_sent_min"][i] = worst
			}
		}
		bar.Add(1)
	}
	elapsed := time.Since(start)
	fmt.Fprintln(os.Stderr)
	log.Printf("cfm: %d/%d candidates had no substitutable entity (swap_margin 0)", noSwap, len(samples))

	// Every signal comes out of the same pass, so they share the cost.
	// Charging each report the full pass would overstate the price of a
	// selected combination; charging a share would understate a single
	// signal. The reports carry the whole-pass runtime and
	// cfm_cost.json carries the breakdown.
	msTotal := 1000 * elapsed.Seconds() / float64(len(samples))
	log.Printf("cfm: one pass = %.2f ms/sample (all %d signals)", msTotal, len(signals))

	if err := writeCost(filepath.Join(outputDir, prefix+"_cost.json"), cost, len(samples), elapsed); err != nil {
		log.Fatal(err)
	}

	for _, sig := range signals {
		name := prefix + "_" + sig
		report := eval.Report{
			Metric:     name,
			Norm:       "none",
			Samples:    len(samples),
			RuntimeSec: elapsed.Seconds(),
			Timestamp:  time.Now().UTC().Format(time.RFC3339),
			Scores:     scoreEntries(samples, feat[sig]),
			SummaryLevel: eval.NewCorrelation(samples, feat[sig], eval.CorrelationOptions{
				Bootstrap: bootstrap, Level: "summary",
			}),
			SystemLevel: eval.NewCorrelation(samples, feat[sig], eval.CorrelationOptions{
				Bootstrap: bootstrap, Level: "system",
			}),
		}
		out := filepath.Join(outputDir, name+".json")
		if err := eval.NewReport(out, report); err != nil {
			log.Fatal(err)
		}
		log.Printf("%-22s %s", name, formatDims(report.SummaryLevel))
	}
}

// conditioning is the prompt prefix that makes the article the context
// for the candidate's log-probability. Deliberately minimal: the metric
// must not turn into an instruction-following judge.
func conditioning(doc string) string {
	return "Article:\n" + doc + "\n\nSummary:\n"
}

// margin is s(C) − mean_i s(C'_i); 0 when no corruption was available,
// which is the value of an uninformative comparison and is reported as
// a coverage statistic rather than hidden.
func margin(base float64, corrupted []float64) float64 {
	if len(corrupted) == 0 {
		return 0
	}
	var sum float64
	for _, v := range corrupted {
		sum += v
	}
	return base - sum/float64(len(corrupted))
}

// winRate is the fraction of corruptions the original outscores. It is
// the rank-based reading of the same comparison: bounded, insensitive
// to how far apart the log-probabilities are, and therefore robust to
// the one or two wild corruptions a sampler occasionally produces.
func winRate(base float64, corrupted []float64) float64 {
	if len(corrupted) == 0 {
		return 0
	}
	win := 0
	for _, v := range corrupted {
		if base > v {
			win++
		}
	}
	return float64(win) / float64(len(corrupted))
}

// costAccount is the hardware-independent side of the cost axis
// (reviewer #3 M1): wall-clock on one machine says as much about the
// machine and the batch size as about the metric, while the number of
// sequences scored and tokens read does not change with the hardware.
type costAccount struct {
	Calls         int `json:"calls"`
	Sequences     int `json:"sequences"`
	ContextTokens int `json:"context_tokens"`
	TargetTokens  int `json:"target_tokens"`
}

func (c *costAccount) add(d metrics.ScoreDetail) {
	c.Calls++
	c.Sequences += d.Sequences
	c.ContextTokens += d.ContextTokens
	c.TargetTokens += d.TargetTokens
}

func writeCost(path string, c *costAccount, samples int, elapsed time.Duration) error {
	out := map[string]any{
		"lm":                 os.Getenv("LM_MODEL"),
		"samples":            samples,
		"runtime_sec":        elapsed.Seconds(),
		"ms_per_sample":      1000 * elapsed.Seconds() / float64(samples),
		"totals":             c,
		"sequences_per_pair": float64(c.Sequences) / float64(samples),
		"tokens_per_pair":    float64(c.ContextTokens+c.TargetTokens) / float64(samples),
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func splitCandidate(text string, minLen int) []string {
	sents := metrics.SplitSentences(text)
	out := make([]string, 0, len(sents))
	for _, s := range sents {
		t := strings.TrimSpace(s)
		if len([]rune(t)) >= minLen {
			out = append(out, t)
		}
	}
	if len(out) == 0 {
		out = []string{strings.TrimSpace(text)}
	}
	return out
}

type docGroup struct {
	ID       string
	Document string
	Idx      []int
}

func groupByDocument(samples []eval.Sample) []docGroup {
	var out []docGroup
	pos := map[string]int{}
	for i, s := range samples {
		p, ok := pos[s.DocumentID]
		if !ok {
			pos[s.DocumentID] = len(out)
			out = append(out, docGroup{ID: s.DocumentID, Document: s.Document, Idx: []int{i}})
			continue
		}
		out[p].Idx = append(out[p].Idx, i)
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

func formatDims(c eval.Correlation) string {
	var b strings.Builder
	for _, d := range c.Dimensions {
		fmt.Fprintf(&b, "%s=%+.3f ", d.Name[:3], d.Spearman)
	}
	return strings.TrimSpace(b.String())
}
