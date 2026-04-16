package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"slices"

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
	norm      string
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
	flag.StringVar(&input, "input", "", "path to dataset JSON/JSONL file")
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL")
	flag.StringVar(&server, "server", "http://localhost:9200", "model server host")
	flag.StringVar(&judge, "judge", "qwen2.5:7b-instruct-q4_K_M", "judge model for G-Eval")
	flag.StringVar(&model, "model", "qwen2.5:3b-instruct", "generative model for BARTScore")
	flag.StringVar(&embed, "embed", "nomic-embed-text", "embedding model for EmbedScorer / SMART-Model")
	flag.StringVar(&dimension, "dimension", "overall", "UniEval dimension: coherence|consistency|fluency|relevance|overall|all")
	flag.StringVar(&norm, "norm", "max", "normalization method for BARTScore: max|mean")
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

	dataset, err := llmbench.NewDatasetV2(fsys, path, n)
	if err != nil {
		log.Fatal(err)
	}

	fn := llmbench.Max
	if norm == "mean" {
		fn = llmbench.Mean
	}

	if bleu4 {
		scores := llmbench.Score(ctx, dataset, llmbench.BLEU, fn)
		corr := llmbench.CorrelationV2(dataset, scores)
		llmbench.PrintResultV2("bleu4", len(scores), corr)
	}

	if rougel {
		scores := llmbench.Score(ctx, dataset, llmbench.ROUGEL, fn)
		corr := llmbench.CorrelationV2(dataset, scores)
		llmbench.PrintResultV2("rougel", len(scores), corr)
	}

	if chrf {
		scores := llmbench.Score(ctx, dataset, llmbench.ChrF, fn)
		corr := llmbench.CorrelationV2(dataset, scores)
		llmbench.PrintResultV2("chrf", len(scores), corr)
	}

	if meteor {
		scores := llmbench.Score(ctx, dataset, llmbench.METEOR, fn)
		corr := llmbench.CorrelationV2(dataset, scores)
		llmbench.PrintResultV2("meteor", len(scores), corr)
	}

	if smartstring {
		scores := llmbench.Score(ctx, dataset, llmbench.SMARTString, fn)
		corr := llmbench.CorrelationV2(dataset, scores)
		llmbench.PrintResultV2("smartstring", len(scores), corr)
	}
}
