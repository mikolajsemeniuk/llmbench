package main

import (
	"flag"
	"log"
	"path/filepath"
	"time"

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
	"github.com/mikolajsemeniuk/llmbench/pkg/metrics"
	"github.com/schollz/progressbar/v3"
)

var (
	datasetName string
	output      string
	norm        string
	n           int
	bootstrap   int
)

func main() {
	flag.StringVar(&datasetName, "dataset", dataset.Default, "embedded corpus in pkg/dataset: summeval|frank_cnndm|frank_xsum|rose_cnndm")
	flag.StringVar(&output, "output", "", "report path (default: output/<dataset>/bleu.json)")
	flag.StringVar(&norm, "norm", eval.DefaultNormName, "reference aggregation: max|mean")
	flag.IntVar(&n, "n", 0, "entries limit (0 = all)")
	flag.IntVar(&bootstrap, "bootstrap", 1000, "bootstrap resamples for 95% CI (0 = disabled)")
	flag.Parse()
	if output == "" {
		output = filepath.Join(eval.OutputDir(datasetName), "bleu.json")
	}

	fn, ok := eval.Norms[norm]
	if !ok {
		fn = eval.DefaultNorm
	}

	samples, err := eval.LoadDataset(datasetName, n)
	if err != nil {
		log.Fatal(err)
	}

	bar := progressbar.NewOptions(
		len(samples),
		progressbar.OptionSetDescription("bleu"),
		progressbar.OptionSetWidth(20),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionShowCount(),
		progressbar.OptionSetElapsedTime(true),
	)

	start := time.Now()
	scores := make([]float64, len(samples))
	entries := make([]eval.Score, len(samples))
	references := make([]float64, 0, 16)

	for i, s := range samples {
		references = references[:0]
		for _, ref := range s.References {
			references = append(references, metrics.BLEU(ref, s.Candidate))
		}
		scores[i] = fn(references)
		entries[i] = eval.Score{SampleID: s.ID, Value: scores[i]}
		bar.Add(1)
	}
	elapsed := time.Since(start)

	report := eval.Report{
		Metric:     "bleu",
		Norm:       norm,
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
