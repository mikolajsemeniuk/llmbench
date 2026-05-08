// SummEval dataset loader. Reads a JSONL file (default: the embedded
// SummEval release, see pkg/dataset/summeval.go) and decodes each
// line into a Sample carrying the source article, the candidate
// summary, the SystemID and DocumentID needed for cluster-bootstrap
// and system-level aggregation, and the four expert human ratings
// (coherence, consistency, fluency, relevance). The Sample struct is
// the contract every cmd/<metric> binary consumes; nothing else in
// pkg/eval depends on the JSON wire format directly.
package eval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
)

type RawSample struct {
	ID               string    `json:"id"`
	Text             string    `json:"text"`
	MachineSummaries []string  `json:"machine_summaries"`
	HumanSummaries   []string  `json:"human_summaries"`
	Relevance        []float64 `json:"relevance"`
	Coherence        []float64 `json:"coherence"`
	Fluency          []float64 `json:"fluency"`
	Consistency      []float64 `json:"consistency"`
}

type Sample struct {
	ID          string
	DocumentID  string
	SystemID    int
	Document    string
	Candidate   string
	References  []string
	Coherence   float64
	Consistency float64
	Fluency     float64
	Relevance   float64
}

func NewDataset(fsys fs.FS, path string, limit int) ([]Sample, error) {
	data, err := fs.ReadFile(fsys, path)
	if err != nil {
		return nil, fmt.Errorf("dataset: %w", err)
	}

	var out []Sample
	expectedSystems := -1 // set from first document

	dec := json.NewDecoder(bytes.NewReader(data))
	for dec.More() {
		var raw RawSample
		if err := dec.Decode(&raw); err != nil {
			return nil, fmt.Errorf("dataset decode: %w", err)
		}

		n := len(raw.MachineSummaries)
		if len(raw.Coherence) != n || len(raw.Consistency) != n ||
			len(raw.Fluency) != n || len(raw.Relevance) != n {
			return nil, fmt.Errorf("dataset: entry %s has %d summaries but mismatched rating lengths", raw.ID, n)
		}

		if expectedSystems < 0 {
			expectedSystems = n
		} else if n != expectedSystems {
			return nil, fmt.Errorf("dataset: entry %s has %d systems but previous entries had %d — system-level correlation assumes consistent system count",
				raw.ID, n, expectedSystems)
		}

		for i, v := range raw.MachineSummaries {
			sample := Sample{
				ID:          fmt.Sprintf("%s#%d", raw.ID, i),
				DocumentID:  raw.ID,
				SystemID:    i,
				Document:    raw.Text,
				Candidate:   v,
				References:  raw.HumanSummaries,
				Coherence:   raw.Coherence[i],
				Consistency: raw.Consistency[i],
				Fluency:     raw.Fluency[i],
				Relevance:   raw.Relevance[i],
			}
			out = append(out, sample)

			if limit > 0 && len(out) >= limit {
				return out, nil
			}
		}
	}

	return out, nil
}
