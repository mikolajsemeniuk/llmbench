// cmd/nliscorer is the go/no-go probe from improvements.txt section 6:
// LGS's mean-of-max grounding uses embedding cosine similarity, which
// measures semantic proximity, not entailment — a paraphrase and a
// subtle contradiction can sit at nearly the same cosine distance.
// Consistency (does the candidate avoid unsupported/contradicted
// claims) is exactly the dimension this distinction should matter for,
// and it is LGS's worst dimension (.227 raw rho).
//
// This binary is LGS's own aggregation --  score = mean over candidate
// sentences of max over source sentences -- with the inner similarity
// swapped from cos(emb(c_j), emb(s_i)) to the NLI entailment
// probability P(source sentence s_i entails candidate sentence c_j),
// via the /nli endpoint of cmd/modelsrv. No lead-bias prior: the
// question here is purely whether entailment is a better signal TYPE
// than cosine, holding the aggregation and everything else fixed. No
// Ollama needed, but cmd/modelsrv must be running with an NLI model
// loaded (see cmd/modelsrv/app.py's /nli endpoint).
//
// The decision this feeds: run cmd/confound afterwards (or use the
// -consistency-only workflow in the Makefile) and look at the
// partial-correlation ceiling for consistency. If NLI clears the
// ~.23 partial-rho ceiling every cosine/lexical/positional metric in
// the pool has hit, entailment is a genuinely different signal class
// and worth building a metric around (Paper 2). If it plateaus at the
// same ceiling, the problem was never the signal type.
//
// Usage:
//
//	go run ./cmd/nliscorer -output output/nli.json
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
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
	minSentLen int
	n          int
	bootstrap  int
	concurrent int
	metricName string
	nliModel   string
)

func main() {
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "output/nli.json", "write results to file")
	flag.StringVar(&server, "server", "http://localhost:9200", "model server URL (port 9200), must have /nli loaded")
	flag.IntVar(&minSentLen, "min-sent-len", 4, "drop sentences shorter than this many runes")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95%% CI (0 = disabled)")
	flag.IntVar(&concurrent, "concurrent", 16, "concurrent /nli calls in flight per sample (the source×candidate sentence grid)")
	flag.StringVar(&metricName, "metric", "nli", "metric name written into the report (lets multiple NLI backbones coexist in output/)")
	flag.StringVar(&nliModel, "nli-model", "cross-encoder/nli-deberta-v3-small", "NLI model label recorded in the report norm (must match NLI_MODEL on the running modelsrv)")
	flag.Parse()

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
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

	scorer := metrics.NewNLI(server)

	log.Printf("NLI grounding probe: %d samples, %d-way concurrency per sample", len(samples), concurrent)

	// Source sentences are shared by all 16 candidates of an article;
	// split once per DocumentID and reuse, same caching strategy as
	// cmd/lgs.
	sentCache := make(map[string][]string)

	bar := progressbar.NewOptions(
		len(samples),
		progressbar.OptionSetDescription("nli"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	scores := make([]float64, len(samples))
	entries := make([]eval.Score, len(samples))
	var sumScore float64

	start := time.Now()

	for i, s := range samples {
		srcSents, ok := sentCache[s.DocumentID]
		if !ok {
			srcSents = filteredSplit(s.Document, minSentLen)
			if len(srcSents) == 0 {
				log.Fatalf("sample %s: source has no usable sentences", s.ID)
			}
			sentCache[s.DocumentID] = srcSents
		}
		candSents := filteredSplit(s.Candidate, minSentLen)
		if len(candSents) == 0 {
			log.Fatalf("sample %s: candidate has no usable sentences", s.ID)
		}

		score, err := meanOfMaxEntailment(ctx, scorer, srcSents, candSents, concurrent)
		if err != nil {
			log.Fatalf("sample %s: %v", s.ID, err)
		}

		scores[i] = score
		entries[i] = eval.Score{SampleID: s.ID, Value: score}
		sumScore += score

		bar.Add(1)
	}

	elapsed := time.Since(start)
	N := float64(len(samples))
	log.Printf("NLI grounding probe: %d samples in %.1fs — mean score=%.3f",
		len(samples), elapsed.Seconds(), sumScore/N)

	report := eval.Report{
		Metric:     metricName,
		Norm:       fmt.Sprintf("nli_model=%s,min_sent_len=%d", nliModel, minSentLen),
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

// meanOfMaxEntailment is LGS's own aggregation with the similarity
// swapped: for each candidate sentence, the max entailment probability
// over every source sentence (source entails candidate), averaged over
// candidate sentences. The n*m grid of NLI calls is scored with a
// bounded worker pool since it is the dominant cost.
func meanOfMaxEntailment(ctx context.Context, scorer *metrics.NLI, srcSents, candSents []string, concurrency int) (float64, error) {
	type job struct{ i, j int }
	type result struct {
		i, j  int
		score float64
		err   error
	}

	jobs := make(chan job, len(srcSents)*len(candSents))
	results := make(chan result, len(srcSents)*len(candSents))

	var wg sync.WaitGroup
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for jb := range jobs {
				score, err := scorer.Score(ctx, srcSents[jb.i], candSents[jb.j])
				results <- result{jb.i, jb.j, score, err}
			}
		}()
	}

	for i := range srcSents {
		for j := range candSents {
			jobs <- job{i, j}
		}
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(results)
	}()

	maxByCand := make([]float64, len(candSents))
	for i := range maxByCand {
		maxByCand[i] = -1
	}
	var firstErr error
	for r := range results {
		if r.err != nil {
			if firstErr == nil {
				firstErr = r.err
			}
			continue
		}
		if r.score > maxByCand[r.j] {
			maxByCand[r.j] = r.score
		}
	}
	if firstErr != nil {
		return 0, firstErr
	}

	var sum float64
	for _, v := range maxByCand {
		sum += v
	}
	return sum / float64(len(maxByCand)), nil
}

func filteredSplit(text string, minLen int) []string {
	raw := metrics.SplitSentences(text)
	out := raw[:0]
	for _, s := range raw {
		if len([]rune(s)) >= minLen {
			out = append(out, s)
		}
	}
	return out
}
