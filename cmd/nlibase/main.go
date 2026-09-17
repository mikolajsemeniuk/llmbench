// cmd/nlibase — zero-shot NLI grounding baseline (SummaC-ZS style).
//
// Two reviewers made the same objection to LGS: cosine similarity is
// not entailment. "Grounding" promises that the source factually
// supports the summary, and a cosine measures only that the two
// sentences are about the same thing. The obvious repair is to swap the
// embedder for an entailment model, which is exactly SummaC-ZS
// (Laban et al., TACL 2022): build the sentence-by-sentence entailment
// matrix between source and summary and aggregate
//
//	score = mean over summary sentences of max over source sentences
//	        of P(entailment)
//
// This binary is that baseline, reimplemented here so it runs under the
// same budget and on the same samples as everything else in the pool.
// It exists for two reasons. First, it is the honest prior-art
// comparison for any reference-free grounding metric (it was missing
// from the submitted manuscript). Second, it is the control for the
// counterfactual-margin metric in cmd/cfm: entailment is invariant to
// paraphrase, so if the extractiveness confound were purely a property
// of *lexical* matching, an NLI metric ought to escape it. Whether it
// does is an empirical question that cmd/confound answers.
//
// -aggregate zs   mean-of-max entailment                 (SummaC-ZS)
// -aggregate conv mean-of-max (entailment − contradiction), a slightly
//
//	stronger variant that penalises contradicted content
//	rather than merely unsupported content.
//
// Needs cmd/lmsrv (default http://localhost:9300) for /nli. No
// reference, no Ollama.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
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

var (
	input      string
	output     string
	server     string
	aggregate  string
	maxSrcSent int
	minSentLen int
	n          int
	bootstrap  int
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "", "write results to file (default: output/nli<aggregate>.json)")
	flag.StringVar(&server, "server", "http://localhost:9300", "cmd/lmsrv host")
	flag.StringVar(&aggregate, "aggregate", "zs", "aggregation: zs (entailment) | conv (entailment − contradiction)")
	flag.IntVar(&maxSrcSent, "max-src-sent", 40, "cap on source sentences considered (cost control)")
	flag.IntVar(&minSentLen, "min-sent-len", 4, "drop sentences shorter than this many runes")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	if aggregate != "zs" && aggregate != "conv" {
		log.Fatalf("-aggregate must be zs|conv, got %q", aggregate)
	}
	if output == "" {
		output = fmt.Sprintf("output/nli%s.json", aggregate)
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

	lm := metrics.NewLMServer(server)
	srcCache := map[string][]string{}

	bar := progressbar.NewOptions(len(samples),
		progressbar.OptionSetDescription("nli-"+aggregate),
		progressbar.OptionSetWidth(20),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
		progressbar.OptionSetPredictTime(true),
	)

	start := time.Now()
	scores := make([]float64, len(samples))

	for i, s := range samples {
		src, ok := srcCache[s.DocumentID]
		if !ok {
			src = trimSentences(metrics.SplitSentences(s.Document), minSentLen, maxSrcSent)
			srcCache[s.DocumentID] = src
		}
		cand := trimSentences(metrics.SplitSentences(s.Candidate), minSentLen, 0)
		if len(src) == 0 || len(cand) == 0 {
			continue
		}

		// One request per sample: the full |cand| x |src| pair grid.
		premises := make([]string, 0, len(cand)*len(src))
		hypotheses := make([]string, 0, len(cand)*len(src))
		for _, c := range cand {
			for _, p := range src {
				premises = append(premises, p)
				hypotheses = append(hypotheses, c)
			}
		}
		ent, con, err := lm.NLI(ctx, premises, hypotheses)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}

		var total float64
		for ci := range cand {
			best := -1.0
			for si := range src {
				k := ci*len(src) + si
				v := ent[k]
				if aggregate == "conv" {
					v = ent[k] - con[k]
				}
				if v > best {
					best = v
				}
			}
			total += best
		}
		scores[i] = total / float64(len(cand))
		bar.Add(1)
	}
	elapsed := time.Since(start)
	fmt.Println()

	name := "nli" + aggregate
	entries := make([]eval.Score, len(samples))
	for i, s := range samples {
		entries[i] = eval.Score{SampleID: s.ID, Value: scores[i]}
	}
	report := eval.Report{
		Metric:     name,
		Norm:       "none",
		Samples:    len(samples),
		RuntimeSec: elapsed.Seconds(),
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Scores:     entries,
		SummaryLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
			Bootstrap: bootstrap, Level: "summary",
		}),
		SystemLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
			Bootstrap: bootstrap, Level: "system",
		}),
	}
	if err := eval.NewReport(output, report); err != nil {
		log.Fatal(err)
	}
	for _, d := range report.SummaryLevel.Dimensions {
		log.Printf("%-8s %s rho=%.3f", name, d.Name, d.Spearman)
	}
	log.Printf("%s: %.2f ms/sample -> %s", name, 1000*elapsed.Seconds()/float64(len(samples)), output)
}

func trimSentences(sents []string, minLen, cap int) []string {
	out := make([]string, 0, len(sents))
	for _, s := range sents {
		t := strings.TrimSpace(s)
		if len([]rune(t)) >= minLen {
			out = append(out, t)
		}
		if cap > 0 && len(out) >= cap {
			break
		}
	}
	return out
}
