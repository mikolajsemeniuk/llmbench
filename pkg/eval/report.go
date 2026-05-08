// Report is the on-disk JSON contract that every cmd/<metric> binary
// writes to output/<metric>.json after a benchmark run, and that the
// rendering binaries (cmd/paper, cmd/compare, cmd/ablation,
// cmd/embedder, cmd/frontier) read back to produce the LaTeX tables
// and figures consumed by paper/main.tex. The struct carries the raw
// per-sample scores, the summary-level and system-level Correlation
// objects (with optional bootstrap CIs), and an optional RunsAggregate
// used only by stochastic metrics (currently just G-Eval) to record
// across-run mean±std for every correlation coefficient.
package eval

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

type Report struct {
	Metric       string         `json:"metric"`
	Norm         string         `json:"norm"`
	Samples      int            `json:"samples"`
	RuntimeSec   float64        `json:"runtime_sec"`
	Timestamp    string         `json:"timestamp"`
	Scores       []Score        `json:"scores"`
	SummaryLevel Correlation    `json:"summary_level"`
	SystemLevel  Correlation    `json:"system_level"`
	Runs         *RunsAggregate `json:"runs,omitempty"`
}

// RunsAggregate captures the across-run mean and standard deviation
// of correlation coefficients. Used by stochastic metrics (G-Eval)
// where the LLM-judge produces a different score on each independent
// run; we report the mean for canonical comparison and the std as
// a measure of LLM-judge variance.
type RunsAggregate struct {
	NRuns       int                 `json:"n_runs"`
	Temperature float64             `json:"temperature"`
	Summary     []DimensionRunStats `json:"summary_level"`
	System      []DimensionRunStats `json:"system_level"`
}

type DimensionRunStats struct {
	Name     string   `json:"name"`
	Spearman RunStats `json:"spearman"`
	Pearson  RunStats `json:"pearson"`
	Kendall  RunStats `json:"kendall_tau"`
}

type RunStats struct {
	Mean float64 `json:"mean"`
	Std  float64 `json:"std"`
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
