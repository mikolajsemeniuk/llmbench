package llmbench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

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

func NewDataset(path string) ([]Entry, error) {
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
	}

	return out, nil
}
