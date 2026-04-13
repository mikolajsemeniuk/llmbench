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

	var ds []Entry

	// Try JSON array first, fall back to JSONL.
	if json.Unmarshal(data, &ds) != nil {
		dec := json.NewDecoder(bytes.NewReader(data))
		for dec.More() {
			var e Entry
			if err := dec.Decode(&e); err != nil {
				return nil, fmt.Errorf("dataset: %w", err)
			}
			ds = append(ds, e)
		}
	}

	return ds, nil
}

// MaxBLEU returns the best BLEU score of candidate against all references.
func MaxBLEU(references []string, candidate string) float64 {
	best := 0.0
	for _, ref := range references {
		if s := BLEU(ref, candidate); s > best {
			best = s
		}
	}
	return best
}

// MaxROUGEL returns the best ROUGE-L score of candidate against all references.
func MaxROUGEL(references []string, candidate string) float64 {
	best := 0.0
	for _, ref := range references {
		if s := ROUGEL(ref, candidate); s > best {
			best = s
		}
	}
	return best
}
