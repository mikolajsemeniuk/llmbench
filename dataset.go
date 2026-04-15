package llmbench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

// Entry is a single row from the SummEval dataset.
type Entry struct {
	ID               string    `json:"id"`
	Text             string    `json:"text"`
	MachineSummaries []string  `json:"machine_summaries"`
	HumanSummaries   []string  `json:"human_summaries"`
	Relevance        []float64 `json:"relevance"`
	Coherence        []float64 `json:"coherence"`
	Fluency          []float64 `json:"fluency"`
	Consistency      []float64 `json:"consistency"`
}

// DatasetOption configures NewDataset behaviour.
type DatasetOption func(*datasetConfig)

type datasetConfig struct {
	size int // 0 = load all entries
}

// WithDatasetSize limits the number of entries loaded from the dataset.
// Useful for quick smoke-test runs without processing the full corpus.
func WithDatasetSize(n int) DatasetOption {
	return func(c *datasetConfig) { c.size = n }
}

// NewDataset loads SummEval entries from a JSON/JSONL file at path.
// Pass WithDatasetSize(n) to load only the first n entries.
func NewDataset(path string, opts ...DatasetOption) ([]Entry, error) {
	cfg := &datasetConfig{}
	for _, o := range opts {
		o(cfg)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("dataset: %w", err)
	}

	var out []Entry
	dec := json.NewDecoder(bytes.NewReader(data))
	for dec.More() {
		var e Entry
		if err := dec.Decode(&e); err != nil {
			return nil, fmt.Errorf("dataset: %w", err)
		}
		out = append(out, e)
		if cfg.size > 0 && len(out) >= cfg.size {
			break
		}
	}

	return out, nil
}
