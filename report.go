package llmbench

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Report handles output destination and format. Write is the single call site.
type Report struct {
	output string
	format string
}

func NewReport(output, format string) *Report {
	return &Report{output: output, format: format}
}

func (r *Report) Write(results []Result) (err error) {
	w, err := r.writer()
	if err != nil {
		return err
	}
	defer func() {
		if e := w.Close(); e != nil && err == nil {
			err = e
		}
	}()
	return Print(w, results, r.format)
}

func (r *Report) writer() (io.WriteCloser, error) {
	if r.output == "" {
		return nopCloser{os.Stdout}, nil
	}
	if dir := filepath.Dir(r.output); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	return os.Create(r.output)
}

type nopCloser struct{ io.Writer }

func (nopCloser) Close() error { return nil }

// Print writes results in the requested format to w.
// Supported formats: "table" (default), "json", "latex".
func Print(w io.Writer, results []Result, format string) error {
	switch format {
	case "json":
		return printJSON(w, results)
	case "latex":
		printLatex(w, results)
	default:
		printTable(w, results)
	}
	return nil
}

var dimOrder = []string{"coherence", "consistency", "fluency", "relevance"}

func findDim(corr Correlation, name string) Dimension {
	for _, d := range corr.Dimensions {
		if d.Name == name {
			return d
		}
	}
	return Dimension{Name: name}
}

func printTable(w io.Writer, results []Result) {
	const namW = 16
	const colW = 13

	fmt.Fprintf(w, "| %-*s |", namW, "Metric")
	for _, dn := range dimOrder {
		fmt.Fprintf(w, " %-*s |", colW, dn)
	}
	fmt.Fprintln(w)

	fmt.Fprintf(w, "| %-*s |", namW, "")
	for range dimOrder {
		fmt.Fprintf(w, "  %5s %5s |", "ρ", "τ")
	}
	fmt.Fprintln(w)

	fmt.Fprintf(w, "|%s|", strings.Repeat("-", namW+2))
	for range dimOrder {
		fmt.Fprintf(w, "%s|", strings.Repeat("-", colW+2))
	}
	fmt.Fprintln(w)

	for _, r := range results {
		fmt.Fprintf(w, "| %-*s |", namW, r.Name)
		for _, dn := range dimOrder {
			d := findDim(r.Corr, dn)
			fmt.Fprintf(w, " %5.3f %5.3f |", d.Spearman, d.KendallTau)
		}
		fmt.Fprintln(w)
	}
}

func printJSON(w io.Writer, results []Result) error {
	type dimResult struct {
		Spearman   float64 `json:"spearman"`
		Pearson    float64 `json:"pearson"`
		KendallTau float64 `json:"kendall_tau"`
	}
	type row struct {
		Metric     string               `json:"metric"`
		Samples    int                  `json:"samples"`
		Dimensions map[string]dimResult `json:"dimensions"`
	}

	out := make([]row, len(results))
	for i, r := range results {
		dims := make(map[string]dimResult, len(r.Corr.Dimensions))
		for _, d := range r.Corr.Dimensions {
			dims[d.Name] = dimResult{
				Spearman:   d.Spearman,
				Pearson:    d.Pearson,
				KendallTau: d.KendallTau,
			}
		}
		out[i] = row{Metric: r.Name, Samples: r.N, Dimensions: dims}
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func printLatex(w io.Writer, results []Result) {
	cols := "l" + strings.Repeat("rr", len(dimOrder))

	fmt.Fprintln(w, `\begin{table}[ht]`)
	fmt.Fprintln(w, `\centering`)
	fmt.Fprintf(w, "\\begin{tabular}{%s}\n", cols)
	fmt.Fprintln(w, `\hline`)

	fmt.Fprint(w, "\\textbf{Metric}")
	for _, dn := range dimOrder {
		fmt.Fprintf(w, " & \\multicolumn{2}{c}{\\textbf{%s}}", dn)
	}
	fmt.Fprintln(w, ` \\`)

	for range dimOrder {
		fmt.Fprint(w, ` & $\rho$ & $\tau$`)
	}
	fmt.Fprintln(w, ` \\ \hline`)

	for _, r := range results {
		fmt.Fprintf(w, "%s", r.Name)
		for _, dn := range dimOrder {
			d := findDim(r.Corr, dn)
			fmt.Fprintf(w, " & %.3f & %.3f", d.Spearman, d.KendallTau)
		}
		fmt.Fprintln(w, ` \\`)
	}

	fmt.Fprintln(w, `\hline`)
	fmt.Fprintln(w, `\end{tabular}`)
	fmt.Fprintln(w, `\end{table}`)
}
