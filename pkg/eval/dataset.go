// Dataset loader. Reads a corpus in the SummEval JSONL layout (the
// embedded corpora in pkg/dataset) and decodes each line into a Sample
// carrying the source article, the candidate summary, the SystemID and
// DocumentID needed for cluster-bootstrap and system-level aggregation,
// and the four human ratings (coherence, consistency, fluency,
// relevance). The Sample struct is the contract every cmd/<metric>
// binary consumes; nothing else in pkg/eval depends on the JSON wire
// format directly.
package eval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"path/filepath"

	"github.com/mikolajsemeniuk/llmbench/pkg/dataset"
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

// LoadDataset decodes one of the corpora embedded in pkg/dataset by name
// (e.g. "summeval", "frank_cnndm").
func LoadDataset(name string, limit int) ([]Sample, error) {
	return NewDataset(dataset.FS, name+".jsonl", limit)
}

// OutputDir is where every tool writes its reports for a corpus
// (<metric>.json, or <metric>_<dimension>.json for per-dimension scorers).
func OutputDir(dataset string) string { return filepath.Join("output", dataset) }

// AblationDir is where ablation variants, raw dumps, feature dumps and
// calibrations for a corpus go.
func AblationDir(dataset string) string { return filepath.Join("ablation", dataset) }

// SplitDocs keeps the articles of one half of a corpus in dataset order:
// "first50" (the SummEval development half LNC is calibrated on),
// "last50" (the held-out half) or "all".
func SplitDocs(samples []Sample, split string) ([]Sample, error) {
	if split == "all" {
		return samples, nil
	}
	var docs []string
	seen := map[string]bool{}
	for _, s := range samples {
		if !seen[s.DocumentID] {
			seen[s.DocumentID] = true
			docs = append(docs, s.DocumentID)
		}
	}
	if len(docs) < 100 {
		return nil, fmt.Errorf("doc split %q needs at least 100 articles, got %d", split, len(docs))
	}
	keep := map[string]bool{}
	switch split {
	case "first50":
		for _, d := range docs[:50] {
			keep[d] = true
		}
	case "last50":
		for _, d := range docs[len(docs)-50:] {
			keep[d] = true
		}
	default:
		return nil, fmt.Errorf("doc split must be all|first50|last50, got %q", split)
	}
	var out []Sample
	for _, s := range samples {
		if keep[s.DocumentID] {
			out = append(out, s)
		}
	}
	return out, nil
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
