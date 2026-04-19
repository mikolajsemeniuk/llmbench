package llmbench

import (
	"context"
	"fmt"
	"sync"
)

func NewBenchmark(ctx context.Context, dataset []Sample, metrics ...MetricFunc) ([]Result, error) {
	results := make([]Result, len(metrics))
	errs := make([]error, len(metrics))

	var wg sync.WaitGroup
	for i, m := range metrics {
		wg.Go(func() {
			results[i], errs[i] = m(ctx, dataset)
		})
	}

	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

type Result struct {
	Name string
	N    int
	Corr Correlation
}

type MetricFunc func(ctx context.Context, dataset []Sample) (Result, error)
type ScorerFunc func(ctx context.Context, s Sample, ref string) (float64, error)
type AggregateFunc func([]float64) float64

func metric(name string, agg AggregateFunc, fn ScorerFunc) MetricFunc {
	return func(ctx context.Context, dataset []Sample) (Result, error) {
		scores := make([]float64, len(dataset))
		for i, s := range dataset {
			refs := make([]float64, len(s.References))
			for j, ref := range s.References {
				v, err := fn(ctx, s, ref)
				if err != nil {
					return Result{}, fmt.Errorf("%s %s: %w", name, s.ID, err)
				}
				refs[j] = v
			}
			scores[i] = agg(refs)
		}

		out := Result{
			Name: name,
			N:    len(dataset),
			Corr: NewCorrelation(dataset, scores),
		}
		return out, nil
	}
}

func WithBLEU(a AggregateFunc) MetricFunc {
	return metric("BLEU", a, func(_ context.Context, s Sample, ref string) (float64, error) {
		return BLEU(ref, s.Candidate), nil
	})
}

func WithROUGEL(a AggregateFunc) MetricFunc {
	return metric("RougeL", a, func(_ context.Context, s Sample, ref string) (float64, error) {
		return ROUGEL(ref, s.Candidate), nil
	})
}

func WithChrF(a AggregateFunc) MetricFunc {
	return metric("ChrF", a, func(_ context.Context, s Sample, ref string) (float64, error) {
		return ChrF(ref, s.Candidate), nil
	})
}

func WithMETEOR(a AggregateFunc) MetricFunc {
	return metric("METEOR", a, func(_ context.Context, s Sample, ref string) (float64, error) {
		return METEOR(ref, s.Candidate), nil
	})
}

func WithSMARTString(a AggregateFunc) MetricFunc {
	return metric("SMARTString", a, func(_ context.Context, s Sample, ref string) (float64, error) {
		return SMARTString(ref, s.Candidate), nil
	})
}

func WithGPTScore(a AggregateFunc, host string) MetricFunc {
	scorer := NewGPTScorer(host)
	return metric("GPTScore", a, func(ctx context.Context, s Sample, ref string) (float64, error) {
		return scorer.Score(ctx, ref, s.Candidate)
	})
}

func WithGEval(agg AggregateFunc, host, model string) MetricFunc {
	g := NewGEval(host, model)
	return metric("GEval", agg, func(ctx context.Context, s Sample, ref string) (float64, error) {
		return g.Score(ctx, s.Document, ref, s.Candidate)
	})
}
