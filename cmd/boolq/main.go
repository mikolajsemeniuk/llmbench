// cmd/boolq — Boolean-question signals and their counterfactual margins.
//
// UniEval's advantage over everything else in the pool survives the
// extractiveness correction almost intact (partial rho .424 of raw
// .470), and what distinguishes it from the similarity metrics is that
// it asks a *question* and reads the probability of the answer rather
// than measuring proximity. UniEval, however, is a model fine-tuned for
// exactly that on synthetic data. This binary asks the same kind of
// question of a general small LM with no fine-tuning at all: the
// context is the article, the summary and a yes/no question, and the
// signal is the length-normalised log-probability of " Yes".
//
// Two families are emitted per dimension:
//
//	bq_<dim>   log P(" Yes" | article, summary, question)
//	bqm_<dim>  the same, minus the mean over corrupted summaries
//
// The margin family is the point. A judge's answer probability carries
// the judge's own biases -- towards long summaries, towards summaries
// that repeat the article's wording -- and those biases are shared by
// the corrupted summary, which has the same length and nearly the same
// overlap with the source. Subtracting them calibrates the judge
// against itself: what remains is how much the judge's answer depends
// on the property the corruption destroyed. The corruption is chosen
// per dimension, as in cmd/cfm.
//
// Cost note: unlike cmd/cfm, the conditioning contains the candidate,
// so the article cannot be encoded once per article and reused across
// its candidates. This is the expensive family, and the ablation of
// interest is whether the margin buys anything over the plain answer
// probability.
//
// Needs cmd/lmsrv.
package main

import (
	"context"
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

var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

// questions are deliberately plain restatements of the SummEval
// annotation guidelines. No chain of thought, no scoring rubric, no
// few-shot examples: the signal has to come from the model's judgement
// of the text, not from prompt engineering that would have to be tuned
// (and then reported as a fitted hyperparameter).
var questions = map[string]string{
	"coherence":   "Is the summary well organised, with its sentences in a sensible order?",
	"consistency": "Is every statement in the summary supported by the article?",
	"fluency":     "Is the summary written in fluent, grammatical English?",
	"relevance":   "Does the summary cover the most important content of the article?",
}

// corruption says which perturbation family the margin for a dimension
// subtracts. Each is the corruption that destroys the property the
// question asks about, and nothing else.
var corruption = map[string]string{
	"coherence":   "permute",  // sentence order
	"consistency": "entities", // factual alignment with the source
	"fluency":     "scramble", // local word order
	"relevance":   "permute",  // content is unchanged; order is not
}

var (
	input      string
	outputDir  string
	prefix     string
	server     string
	nPerturb   int
	minSentLen int
	maxSrcChar int
	n          int
	bootstrap  int
	seed       uint64
	noMargin   bool
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&outputDir, "output-dir", "output", "directory for the per-signal reports")
	flag.StringVar(&prefix, "prefix", "bq", "metric-name prefix")
	flag.StringVar(&server, "server", "http://localhost:9300", "cmd/lmsrv host")
	flag.IntVar(&nPerturb, "perturb", 2, "corruptions per margin (0 with -no-margin)")
	flag.IntVar(&minSentLen, "min-sent-len", 4, "drop candidate sentences shorter than this many runes")
	flag.IntVar(&maxSrcChar, "max-source-chars", 6000, "truncate the article to this many characters")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 0, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Uint64Var(&seed, "seed", 42, "random seed")
	flag.BoolVar(&noMargin, "no-margin", false, "emit only the plain answer probabilities")
	flag.Parse()

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

	lm := metrics.NewLMServer(server)
	rng := rand.New(rand.NewPCG(seed, seed^0x5eed))

	plain := map[string][]float64{}
	marg := map[string][]float64{}
	for _, d := range dimensions {
		plain[d] = make([]float64, len(samples))
		marg[d] = make([]float64, len(samples))
	}

	bar := progressbar.NewOptions(len(samples),
		progressbar.OptionSetDescription("boolq"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionSetWriter(os.Stderr),
	)

	poolCache := map[string]metrics.EntityPool{}
	start := time.Now()

	for i, s := range samples {
		sents := splitCandidate(s.Candidate, minSentLen)
		article := truncate(s.Document, maxSrcChar)

		pool, ok := poolCache[s.DocumentID]
		if !ok {
			pool = metrics.NewEntityPool(s.Document)
			poolCache[s.DocumentID] = pool
		}

		variants := map[string][]string{}
		if !noMargin {
			variants["permute"] = metrics.Permute(sents, nPerturb, rng)
			variants["scramble"] = metrics.Scramble(sents, nPerturb, rng)
			variants["entities"] = metrics.SwapEntities(strings.Join(sents, " "), pool, nPerturb, rng)
		}

		// Every prompt of this sample -- four dimensions, each with its
		// uncorrupted candidate and its corruptions -- goes out as one
		// batched request. The candidate is inside the prompt, so no
		// prefix can be shared; batching over prompts is the only
		// saving available and it is worth ~8x.
		var contexts, targets []string
		counts := map[string]int{}
		for _, d := range dimensions {
			cands := []string{strings.Join(sents, " ")}
			if !noMargin {
				cands = append(cands, variants[corruption[d]]...)
			}
			counts[d] = len(cands)
			for _, c := range cands {
				contexts = append(contexts, prompt(article, c, questions[d]))
				targets = append(targets, " Yes")
			}
		}
		det, err := lm.ScorePairs(ctx, contexts, targets)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}
		at := 0
		for _, d := range dimensions {
			vals := det.MeanLogprob[at : at+counts[d]]
			at += counts[d]
			plain[d][i] = vals[0]
			if len(vals) > 1 {
				var sum float64
				for _, v := range vals[1:] {
					sum += v
				}
				marg[d][i] = vals[0] - sum/float64(len(vals)-1)
			}
		}
		bar.Add(1)
	}
	elapsed := time.Since(start)
	fmt.Fprintln(os.Stderr)

	write := func(name string, vals []float64, share float64) {
		report := eval.Report{
			Metric:     name,
			Norm:       "logprob(\" Yes\")",
			Samples:    len(samples),
			RuntimeSec: elapsed.Seconds() * share,
			Timestamp:  time.Now().UTC().Format(time.RFC3339),
			Scores:     scoreEntries(samples, vals),
			SummaryLevel: eval.NewCorrelation(samples, vals, eval.CorrelationOptions{
				Bootstrap: bootstrap, Level: "summary",
			}),
			SystemLevel: eval.NewCorrelation(samples, vals, eval.CorrelationOptions{
				Bootstrap: bootstrap, Level: "system",
			}),
		}
		if err := eval.NewReport(filepath.Join(outputDir, name+".json"), report); err != nil {
			log.Fatal(err)
		}
		log.Printf("%-24s %s", name, formatDims(report.SummaryLevel))
	}

	// Cost attribution: the plain probability is one prompt out of the
	// 1+nPerturb this binary runs per (sample, dimension), so it is
	// charged that share of the pass; the margin needs all of them.
	shareEach := 1.0 / float64(len(dimensions))
	sharePlain := shareEach / float64(1+nPerturbOr0())
	for _, d := range dimensions {
		write(prefix+"_"+d, plain[d], sharePlain)
		if !noMargin {
			write(prefix+"m_"+d, marg[d], shareEach)
		}
	}
	log.Printf("boolq: whole pass %.1f s = %.2f ms/sample", elapsed.Seconds(),
		1000*elapsed.Seconds()/float64(len(samples)))
}

func nPerturbOr0() int {
	if noMargin {
		return 0
	}
	return nPerturb
}

func prompt(article, summary, question string) string {
	return "Article:\n" + article + "\n\nSummary:\n" + summary +
		"\n\nQuestion: " + question + " Answer Yes or No.\nAnswer:"
}

func truncate(s string, maxChars int) string {
	if maxChars <= 0 || len(s) <= maxChars {
		return s
	}
	return s[:maxChars]
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
