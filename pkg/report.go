package llmbench

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

type Report struct {
	Metric       string      `json:"metric"`
	Norm         string      `json:"norm"`
	Samples      int         `json:"samples"`
	RuntimeSec   float64     `json:"runtime_sec"`
	Timestamp    string      `json:"timestamp"`
	Scores       []Score     `json:"scores"`
	SummaryLevel Correlation `json:"summary_level"`
	SystemLevel  Correlation `json:"system_level"`
}

type Score struct {
	SampleID string  `json:"sample_id"`
	Value    float64 `json:"value"`
}

// NewReport writes the report as indented JSON.
// Empty output or "-" writes to stdout; otherwise creates parent dirs as needed.
func NewReport(output string, r Report) error {
	var w io.Writer = os.Stdout
	if output != "" && output != "-" {
		if dir := filepath.Dir(output); dir != "." && dir != "" {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return fmt.Errorf("report: mkdir %s: %w", dir, err)
			}
		}
		f, err := os.Create(output)
		if err != nil {
			return fmt.Errorf("report: create %s: %w", output, err)
		}
		defer f.Close()
		w = f
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		return fmt.Errorf("report: encode: %w", err)
	}
	return nil
}
