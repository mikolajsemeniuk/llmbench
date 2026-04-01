package llmbench

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

type Sample struct {
	ID        string `json:"id"`
	Question  string `json:"question"`
	Reference string `json:"reference"`
	Candidate string `json:"candidate"`
}

type Result struct {
	ID    string  `json:"id"`
	Score float64 `json:"score"`
}

type Report struct {
	Metric    string   `json:"metric"`
	Timestamp string   `json:"timestamp"`
	Results   []Result `json:"results"`
	Mean      float64  `json:"mean"`
	Min       float64  `json:"min"`
	Max       float64  `json:"max"`
}

func NewSamples(path string) ([]Sample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open samples: %w", err)
	}
	defer f.Close()

	var samples []Sample
	if err := json.NewDecoder(f).Decode(&samples); err != nil {
		return nil, fmt.Errorf("decode samples: %w", err)
	}

	return samples, nil
}

func NewReport(metric string, results []Result) Report {
	r := Report{
		Metric:    metric,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Results:   results,
	}

	sum := 0.0
	r.Min = results[0].Score
	r.Max = results[0].Score
	for _, s := range results {
		sum += s.Score
		if s.Score < r.Min {
			r.Min = s.Score
		}

		if s.Score > r.Max {
			r.Max = s.Score
		}
	}

	r.Mean = sum / float64(len(results))
	return r
}

func (r Report) WriteJSON(path string) error {
	out, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}

	if path == "" || path == "-" {
		_, err = os.Stdout.Write(append(out, '\n'))
		return err
	}

	return os.WriteFile(path, append(out, '\n'), 0o644)
}
