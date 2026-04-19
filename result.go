package llmbench

import "context"

type Result struct {
	Name string
	N    int
	Corr Correlation
}

type MetricOption func(ctx context.Context, dataset []Sample, norm Norm) (Result, error)

func NewResults(ctx context.Context, dataset []Sample, norm Norm, opts ...MetricOption) ([]Result, error) {
	results := make([]Result, 0, len(opts))
	for _, opt := range opts {
		r, err := opt(ctx, dataset, norm)
		if err != nil {
			return nil, err
		}
		results = append(results, r)
	}

	return results, nil
}

func RunMetric(ctx context.Context, name string, dataset []Sample, m Metric, norm Norm) Result {
	scores := Score(ctx, dataset, m, norm)
	return Result{
		Name: name,
		N:    len(scores),
		Corr: NewCorrelation(dataset, scores),
	}
}

func withMetric(name string, m Metric) MetricOption {
	return func(ctx context.Context, dataset []Sample, norm Norm) (Result, error) {
		return RunMetric(ctx, name, dataset, m, norm), nil
	}
}

func WithBLEU() MetricOption        { return withMetric("BLEU-4", BLEU) }
func WithROUGEL() MetricOption      { return withMetric("ROUGE-L", ROUGEL) }
func WithChrF() MetricOption        { return withMetric("ChrF", ChrF) }
func WithMETEOR() MetricOption      { return withMetric("METEOR", METEOR) }
func WithSMARTString() MetricOption { return withMetric("SMART-String", SMARTString) }

func WithGPTScore(server string) MetricOption {
	return nil
}
