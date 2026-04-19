package llmbench

import "context"

type Result struct {
	Name string
	N    int
	Corr Correlation
}

type MetricOption func(Sample) (float64, error)

func NewResults(dataset []Sample, norm Norm, opts ...MetricOption) ([]Result, error) {
	var results []Result
	for _, opt := range opts {
		scores := make([]float64, len(dataset))
		for i, v := range dataset {
			refs := make([]float64, len(v.References))
			for j, ref := range v.References {
				s := v
				s.Document = ref
				v, err := opt(s)
				if err != nil {
					return nil, err
				}

				refs[j] = v
			}

			scores[i] = norm(refs)
		}

		results = append(results, Result{N: len(scores), Corr: NewCorrelation(dataset, scores)})
	}

	return results, nil
}

func WithBLEU() MetricOption {
	return func(s Sample) (float64, error) {
		return BLEU(s.Document, s.Candidate), nil
	}
}

func WithROUGEL() MetricOption {
	return func(s Sample) (float64, error) {
		return ROUGEL(s.Document, s.Candidate), nil
	}
}

func WithChrF() MetricOption {
	return func(s Sample) (float64, error) {
		return ChrF(s.Document, s.Candidate), nil
	}
}

func WithMETEOR() MetricOption {
	return func(s Sample) (float64, error) {
		return METEOR(s.Document, s.Candidate), nil
	}
}

func WithSMARTString() MetricOption {
	return func(s Sample) (float64, error) {
		return SMARTString(s.Document, s.Candidate), nil
	}
}

func WithGPTScore(server string) MetricOption {
	return func(s Sample) (float64, error) {
		scorer := NewGPTScorer(server)
		return scorer.Score(context.Background(), s.Document, s.Candidate)
	}
}

func WithGEval(host, judge string) MetricOption {
	return func(s Sample) (float64, error) {
		scorer := NewGEval(host, judge)
		return scorer.Score(context.Background(), s.Document, s.Candidate, "")
	}
}
