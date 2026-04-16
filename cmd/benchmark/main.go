package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"slices"
	"strings"
	"sync"

	"github.com/mikolajsemeniuk/llmbench"
)

var (
	input     string
	host      string
	server    string
	judge     string
	model     string
	embed     string
	dimension string
	n         int

	bleu4       bool
	rougel      bool
	chrf        bool
	meteor      bool
	smartstring bool
	bartscore   bool
	embedscorer bool
	smartmodel  bool
	geval       bool
	gptscore    bool
	bertscore   bool
	moverscore  bool
	unieval     bool
)

func main() {
	flag.StringVar(&input, "input", "model_annotations.aligned.scored.jsonl", "path to SummEval dataset JSON/JSONL file")
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL")
	flag.StringVar(&server, "server", "http://localhost:9200", "model server host")
	flag.StringVar(&judge, "judge", "qwen2.5:7b-instruct-q4_K_M", "judge model for G-Eval")
	flag.StringVar(&model, "model", "qwen2.5:3b-instruct", "generative model for BARTScore")
	flag.StringVar(&embed, "embed", "nomic-embed-text", "embedding model for EmbedScorer / SMART-Model")
	flag.StringVar(&dimension, "dimension", "overall", "UniEval dimension: coherence|consistency|fluency|relevance|overall|all")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.BoolVar(&bleu4, "bleu4", false, "run BLEU-4")
	flag.BoolVar(&rougel, "rougel", false, "run ROUGE-L")
	flag.BoolVar(&chrf, "chrf", false, "run ChrF")
	flag.BoolVar(&meteor, "meteor", false, "run METEOR")
	flag.BoolVar(&smartstring, "smartstring", false, "run SMART-String")
	flag.BoolVar(&bartscore, "bartscore", false, "run BARTScore")
	flag.BoolVar(&embedscorer, "embedscorer", false, "run EmbedScorer")
	flag.BoolVar(&smartmodel, "smartmodel", false, "run SMART-Model")
	flag.BoolVar(&geval, "geval", false, "run G-Eval")
	flag.BoolVar(&gptscore, "gptscore", false, "run GPTScore")
	flag.BoolVar(&bertscore, "bertscore", false, "run BERTScore")
	flag.BoolVar(&moverscore, "moverscore", false, "run MoverScore")
	flag.BoolVar(&unieval, "unieval", false, "run UniEval")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	var opts []llmbench.DatasetOption
	if n > 0 {
		opts = append(opts, llmbench.WithDatasetSize(n))
	}
	dataset, err := llmbench.NewDataset(input, opts...)
	if err != nil {
		log.Fatal(err)
	}

	metrics := []bool{bleu4, rougel, chrf, meteor, smartstring, bartscore, embedscorer, smartmodel, geval, gptscore, bertscore, moverscore, unieval}
	if !slices.Contains(metrics, true) {
		bleu4, rougel, chrf, meteor, smartstring = true, true, true, true, true
		bartscore, embedscorer, smartmodel, geval = true, true, true, true
		gptscore, bertscore, moverscore, unieval = true, true, true, true
	}

	type scorer interface {
		Score([]llmbench.Entry) (llmbench.ScoreOutput, error)
	}

	type result struct {
		name  string
		score llmbench.ScoreOutput
		corr  llmbench.CorrelationOutput
		err   error
	}

	var (
		mu  sync.Mutex
		wg  sync.WaitGroup
		out []result
	)

	run := func(name string, s scorer) {
		score, err := s.Score(dataset)
		mu.Lock()
		defer mu.Unlock()
		if err != nil {
			out = append(out, result{name: name, err: err})
			return
		}

		out = append(out, result{name: name, score: score, corr: llmbench.Correlation(score)})
	}

	if bartscore {
		wg.Go(func() { run("bartscore", llmbench.NewBARTScorer(ctx, host, model)) })
	}
	if embedscorer {
		wg.Go(func() { run("embedscorer", llmbench.NewEmbeddingScorer(ctx, host, embed)) })
	}
	if smartmodel {
		wg.Go(func() { run("smartmodel", llmbench.NewSMARTModelScorer(ctx, host, embed)) })
	}
	if geval {
		wg.Go(func() { run("geval", llmbench.NewGEval(ctx, host, judge)) })
	}
	if gptscore {
		wg.Go(func() { run("gptscore", llmbench.NewGPTScorer(ctx, server)) })
	}
	if bertscore {
		wg.Go(func() { run("bertscore", llmbench.NewBERTScorer(ctx, server)) })
	}
	if moverscore {
		wg.Go(func() { run("moverscore", llmbench.NewMoverScorer(ctx, server)) })
	}
	if unieval {
		wg.Go(func() { run("unieval", llmbench.NewUniEvalScorer(ctx, server, dimension)) })
	}

	wg.Wait()
	fmt.Fprintln(os.Stderr)

	header := fmt.Sprintf("%-16s  %7s  %7s %7s  %7s %7s  %7s %7s  %7s %7s",
		"metric", "samples",
		"coh-ρ", "coh-r",
		"con-ρ", "con-r",
		"flu-ρ", "flu-r",
		"rel-ρ", "rel-r",
	)
	sep := strings.Repeat("-", len(header))
	fmt.Fprintln(os.Stderr, sep)
	fmt.Fprintln(os.Stderr, header)
	fmt.Fprintln(os.Stderr, sep)

	for _, v := range out {
		if v.err != nil {
			fmt.Fprintf(os.Stderr, "%-16s  ERROR: %v\n", v.name, v.err)
			continue
		}

		d := v.corr.Dimensions
		fmt.Fprintf(os.Stderr,
			"%-16s  %7d  %7.4f %7.4f  %7.4f %7.4f  %7.4f %7.4f  %7.4f %7.4f\n",
			v.name,
			len(v.score.Scores),
			d[0].Spearman, d[0].Pearson,
			d[1].Spearman, d[1].Pearson,
			d[2].Spearman, d[2].Pearson,
			d[3].Spearman, d[3].Pearson,
		)
	}

	fmt.Fprintln(os.Stderr, sep)
}
