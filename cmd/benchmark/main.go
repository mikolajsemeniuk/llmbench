package main

import (
	"context"
	"flag"
	"io"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"slices"

	llmbench "github.com/mikolajsemeniuk/llmbench/pkg"
)

var (
	input     string
	output    string
	host      string
	server    string
	judge     string
	model     string
	embed     string
	dimension string
	norm      string
	format    string
	n         int
	workers   int

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
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&output, "output", "", "write results to file instead of stdout")
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL")
	flag.StringVar(&server, "server", "http://localhost:9200", "model server host")
	flag.StringVar(&judge, "judge", "qwen2.5:7b-instruct-q4_K_M", "judge model for G-Eval")
	flag.StringVar(&model, "model", "qwen2.5:3b-instruct", "generative model for BARTScore")
	flag.StringVar(&embed, "embed", "nomic-embed-text", "embedding model for EmbedScorer / SMART-Model")
	flag.StringVar(&dimension, "dimension", "overall", "UniEval dimension: coherence|consistency|fluency|relevance|overall|all")
	flag.StringVar(&norm, "norm", "max", "reference aggregation: max|mean")
	flag.StringVar(&format, "format", "table", "output format: table|json|latex")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&workers, "workers", 8, "concurrent samples per metric")

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

	ctx := context.Background()
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	metrics := []bool{bleu4, rougel, chrf, meteor, smartstring, bartscore, embedscorer, smartmodel, geval, gptscore, bertscore, moverscore, unieval}
	if !slices.Contains(metrics, true) {
		bleu4, rougel, chrf, meteor, smartstring = true, true, true, true, true
		bartscore, embedscorer, smartmodel, geval = true, true, true, true
		gptscore, bertscore, moverscore, unieval = true, true, true, true
	}

	fsys := os.DirFS(filepath.Dir(input))
	path := filepath.Base(input)
	if input == "" {
		fsys = llmbench.SummevalDataset
		path = llmbench.DefaultDatasetPath
	}

	dataset, err := llmbench.NewDataset(fsys, path, n)
	if err != nil {
		log.Fatal(err)
	}

	fn := llmbench.Max
	if norm == "mean" {
		fn = llmbench.Mean
	}

	scorers := map[string]llmbench.Scorer{}
	if bleu4 {
		scorers["BLEU-4"] = llmbench.Sync(llmbench.BLEU, fn)
	}
	if rougel {
		scorers["ROUGE-L"] = llmbench.Sync(llmbench.ROUGEL, fn)
	}
	if chrf {
		scorers["ChrF"] = llmbench.Sync(llmbench.ChrF, fn)
	}
	if meteor {
		scorers["METEOR"] = llmbench.Sync(llmbench.METEOR, fn)
	}
	if smartstring {
		scorers["SMART-String"] = llmbench.Sync(llmbench.SMARTString, fn)
	}
	if bartscore {
		scorers["BARTScore"] = llmbench.Async(llmbench.NewBARTScorer(ctx, host, model).Score, fn)
	}
	if embedscorer {
		scorers["EmbedScorer"] = llmbench.Async(llmbench.NewEmbeddingScorer(ctx, host, embed).Score, fn)
	}
	if smartmodel {
		scorers["SMART-Model"] = llmbench.Async(llmbench.NewSMARTModelScorer(ctx, host, embed).Score, fn)
	}
	if gptscore {
		scorers["GPTScore"] = llmbench.Async(llmbench.NewGPTScorer(server).Score, fn)
	}
	if bertscore {
		scorers["BERTScore"] = llmbench.Async(llmbench.NewModelServer(server).BERTScoreCanonical, fn)
	}
	if moverscore {
		scorers["MoverScore"] = llmbench.Async(llmbench.NewModelServer(server).MoverScore, fn)
	}
	if geval {
		g := llmbench.NewGEval(host, judge)
		scorers["G-Eval"] = func(ctx context.Context, s llmbench.Sample) (float64, error) {
			return llmbench.AggregateAsync(ctx, s.References, s.Candidate, fn,
				func(ctx context.Context, ref, cand string) (float64, error) {
					return g.Score(ctx, s.Document, ref, cand)
				})
		}
	}
	if unieval {
		ms := llmbench.NewModelServer(server)
		scorers["UniEval"] = func(ctx context.Context, s llmbench.Sample) (float64, error) {
			return llmbench.AggregateAsync(ctx, s.References, s.Candidate, fn,
				func(ctx context.Context, ref, cand string) (float64, error) {
					return ms.UniEval(ctx, ref, cand, dimension)
				})
		}
	}

	runner := &llmbench.Runner{
		Dataset:  dataset,
		Scorers:  scorers,
		Workers:  workers,
		Progress: os.Stderr,
	}
	results := runner.Run(ctx)

	out := io.Writer(os.Stdout)
	if output != "" {
		f, err := os.Create(output)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		out = f
	}

	report := &llmbench.Report{Results: results, Norm: norm}
	if err := report.Write(out, format); err != nil {
		log.Fatal(err)
	}
}
