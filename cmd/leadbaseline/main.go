// cmd/leadbaseline runs the trivial lead-position controls that LGS
// must beat to justify its machinery.
//
// LGS encodes one idea -- salient content concentrates near the start
// of a news article -- and then spends a sentence embedder, a
// per-candidate argmax and a tuned exponential weight implementing it.
// The controls here implement the same idea with none of that: take
// the first k source sentences, throw the rest of the article away,
// and match against them with ROUGE-L. No embedder, no hyperparameter
// sweep, no model of any kind. Like LGS they are reference-free, so
// the comparison isolates the machinery rather than the input.
//
// Two modes, corresponding to the two things LGS adds over a bag of
// lead sentences:
//
//	whole  ROUGE-L(lead_k as one block, candidate as one block).
//	       The crudest possible reading of "compare the summary to
//	       the top of the article".
//	sent   mean over candidate sentences of the max ROUGE-L against
//	       any of the first k source sentences. This is LGS's own
//	       aggregation structure -- mean-of-max, sentence level --
//	       with lexical overlap substituted for the embedding cosine
//	       and a hard cutoff substituted for the exponential prior.
//	       It is the sharper control: a small gap here means the
//	       embedder is carrying the metric, a small gap in `whole`
//	       means the whole design is.
//
// Output is a standard eval.Report, so these land in the correlation
// tables alongside every other metric and are directly comparable.
// CPU only: no Ollama, no model server.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
	"github.com/mikolajsemeniuk/llmbench/pkg/metrics"
)

var (
	input      string
	output     string
	leadK      int
	mode       string
	minSentLen int
	docSplit   string
	n          int
	bootstrap  int
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "", "write results to file (default: output/lead<k><mode>.json)")
	flag.IntVar(&leadK, "lead-k", 3, "number of leading source sentences to keep")
	flag.StringVar(&mode, "mode", "sent", "matching mode: sent (mean-of-max over sentences) | whole (single block)")
	flag.IntVar(&minSentLen, "min-sent-len", 4, "drop sentences shorter than this many runes (matches cmd/lgs)")
	flag.StringVar(&docSplit, "doc-split", "all", "article-level split: all|first50|last50")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.Parse()

	if leadK < 1 {
		log.Fatalf("-lead-k must be ≥ 1, got %d", leadK)
	}
	if mode != "sent" && mode != "whole" {
		log.Fatalf("-mode must be sent|whole, got %q", mode)
	}
	switch docSplit {
	case "all", "first50", "last50":
	default:
		log.Fatalf("-doc-split must be all|first50|last50, got %q", docSplit)
	}
	if output == "" {
		output = fmt.Sprintf("output/lead%d%s.json", leadK, mode)
	}

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
	samples = applyDocSplit(samples, docSplit)

	log.Printf("lead baseline: k=%d mode=%s split=%s, %d samples", leadK, mode, docSplit, len(samples))

	// The lead sentences depend only on the article, so they are cut
	// once per DocumentID and reused across its 16 candidates -- the
	// same per-document caching cmd/lgs applies to source embeddings.
	leadCache := map[string][]string{}

	start := time.Now()
	scores := make([]float64, len(samples))
	entries := make([]eval.Score, len(samples))
	var sum float64

	for i, s := range samples {
		lead, ok := leadCache[s.DocumentID]
		if !ok {
			lead = leadSentences(s.Document, leadK, minSentLen)
			if len(lead) == 0 {
				log.Fatalf("sample %s: source has no usable sentences", s.ID)
			}
			leadCache[s.DocumentID] = lead
		}

		var v float64
		if mode == "whole" {
			v = metrics.ROUGEL(joinSentences(lead), s.Candidate)
		} else {
			v = meanOfMaxROUGE(lead, s.Candidate, minSentLen)
		}

		scores[i] = v
		entries[i] = eval.Score{SampleID: s.ID, Value: v}
		sum += v
	}

	elapsed := time.Since(start)
	log.Printf("lead baseline: %d samples in %.2fs — mean score=%.3f",
		len(samples), elapsed.Seconds(), sum/float64(len(samples)))

	report := eval.Report{
		Metric:     fmt.Sprintf("lead%d%s", leadK, mode),
		Norm:       fmt.Sprintf("lead_k=%d,mode=%s,split=%s", leadK, mode, docSplit),
		Samples:    len(samples),
		RuntimeSec: elapsed.Seconds(),
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Scores:     entries,
		SummaryLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
			Bootstrap: bootstrap,
			Level:     "summary",
		}),
		SystemLevel: eval.NewCorrelation(samples, scores, eval.CorrelationOptions{
			Bootstrap: bootstrap,
			Level:     "system",
		}),
	}
	if err := eval.NewReport(output, report); err != nil {
		log.Fatal(err)
	}
}

// meanOfMaxROUGE mirrors the LGS aggregator exactly -- for each
// candidate sentence take the best-matching source sentence, then
// average over candidate sentences -- but scores the match with
// ROUGE-L instead of an embedding cosine, and draws the source
// sentences from the lead window instead of the whole article under a
// decaying weight.
func meanOfMaxROUGE(lead []string, candidate string, minLen int) float64 {
	cand := filterShort(metrics.SplitSentences(candidate), minLen)
	if len(cand) == 0 || len(lead) == 0 {
		return 0
	}
	var total float64
	for _, c := range cand {
		best := 0.0
		for _, s := range lead {
			if v := metrics.ROUGEL(s, c); v > best {
				best = v
			}
		}
		total += best
	}
	return total / float64(len(cand))
}

// leadSentences returns the first k usable sentences of the document,
// applying the same degenerate-sentence filter as cmd/lgs so that the
// two metrics see the same source units.
func leadSentences(document string, k, minLen int) []string {
	sents := filterShort(metrics.SplitSentences(document), minLen)
	if len(sents) > k {
		sents = sents[:k]
	}
	return sents
}

func filterShort(sents []string, minLen int) []string {
	out := sents[:0]
	for _, s := range sents {
		if len([]rune(s)) >= minLen {
			out = append(out, s)
		}
	}
	return out
}

func joinSentences(sents []string) string {
	out := ""
	for i, s := range sents {
		if i > 0 {
			out += " "
		}
		out += s
	}
	return out
}

// applyDocSplit mirrors cmd/lgs so the lead controls can be run on the
// same development and test halves as the λ sweep.
func applyDocSplit(samples []eval.Sample, split string) []eval.Sample {
	if split == "all" {
		return samples
	}
	seen := map[string]struct{}{}
	var docs []string
	for _, s := range samples {
		if _, ok := seen[s.DocumentID]; ok {
			continue
		}
		seen[s.DocumentID] = struct{}{}
		docs = append(docs, s.DocumentID)
	}
	if len(docs) < 100 {
		log.Fatalf("doc-split %q expects ≥100 documents, got %d", split, len(docs))
	}
	var keep []string
	switch split {
	case "first50":
		keep = docs[:50]
	case "last50":
		keep = docs[len(docs)-50:]
	}
	keepSet := make(map[string]struct{}, len(keep))
	for _, id := range keep {
		keepSet[id] = struct{}{}
	}
	out := samples[:0]
	for _, s := range samples {
		if _, ok := keepSet[s.DocumentID]; ok {
			out = append(out, s)
		}
	}
	return out
}
