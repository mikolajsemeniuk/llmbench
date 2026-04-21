package llmbench

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
	"unicode/utf8"
)

// Report renders benchmark results in a human-readable table, JSON for
// downstream processing, or LaTeX for direct inclusion in a paper.
type Report struct {
	Results []Result
	Norm    string
}

// Write emits the report in the requested format: table | json | latex.
// Unknown formats fall back to the table renderer.
func (r *Report) Write(w io.Writer, format string) error {
	switch strings.ToLower(format) {
	case "json":
		return r.writeJSON(w)
	case "latex":
		return r.writeLaTeX(w)
	default:
		return r.writeTable(w)
	}
}

func (r *Report) writeTable(w io.Writer) error {
	const (
		metricW = 14
		valueW  = 6  // fits "-0.102" (%6.3f)
		dimW    = 1 + valueW + 2 + valueW + 2 + valueW + 1 // " V  V  V " = 24
		timeW   = 16 // fits e.g. "6m5.159s (10e)"
	)
	dims := []string{"Coherence", "Consistency", "Fluency", "Relevance"}

	samples := 0
	if len(r.Results) > 0 {
		samples = len(r.Results[0].Scores)
	}
	fmt.Fprintf(w, "Samples: %d\n\n", samples)

	// Header row 1: dimension names
	fmt.Fprintf(w, "%s |", padRight("Metric", metricW))
	for _, d := range dims {
		fmt.Fprintf(w, "%s|", padCenter(d, dimW))
	}
	fmt.Fprintf(w, " %s\n", padRight("Time", timeW))

	// Header row 2: sub-headers r ρ τ
	fmt.Fprintf(w, "%s |", padRight("", metricW))
	for range dims {
		fmt.Fprintf(w, " %s  %s  %s |",
			padCenter("r", valueW),
			padCenter("ρ", valueW),
			padCenter("τ", valueW))
	}
	fmt.Fprintf(w, " %s\n", padRight("", timeW))

	// Divider
	fmt.Fprintf(w, "%s+", strings.Repeat("-", metricW+1))
	for range dims {
		fmt.Fprintf(w, "%s+", strings.Repeat("-", dimW))
	}
	fmt.Fprintf(w, "%s\n", strings.Repeat("-", timeW+1))

	// Data rows
	for _, res := range r.Results {
		allFailed := res.Errors > 0 && res.Errors == len(res.Scores)
		fmt.Fprintf(w, "%s |", padRight(res.Name, metricW))
		for _, d := range res.Correlation.Dimensions {
			if allFailed {
				fmt.Fprintf(w, " %s  %s  %s |",
					padCenter("—", valueW), padCenter("—", valueW), padCenter("—", valueW))
				continue
			}
			fmt.Fprintf(w, " %6.3f  %6.3f  %6.3f |",
				d.Pearson, d.Spearman, d.KendallTau)
		}
		timeStr := res.Duration.Truncate(time.Millisecond).String()
		if res.Errors > 0 {
			timeStr = fmt.Sprintf("%s (%de)", timeStr, res.Errors)
		}
		fmt.Fprintf(w, " %s\n", padRight(timeStr, timeW))
	}
	return nil
}

func padRight(s string, n int) string {
	w := utf8.RuneCountInString(s)
	if w >= n {
		return s
	}
	return s + strings.Repeat(" ", n-w)
}

func padCenter(s string, n int) string {
	w := utf8.RuneCountInString(s)
	if w >= n {
		return s
	}
	left := (n - w) / 2
	right := n - w - left
	return strings.Repeat(" ", left) + s + strings.Repeat(" ", right)
}

type dimJSON struct {
	Name       string  `json:"name"`
	Pearson    float64 `json:"pearson"`
	Spearman   float64 `json:"spearman"`
	KendallTau float64 `json:"kendall_tau"`
}

type resultJSON struct {
	Name       string    `json:"name"`
	DurationMS int64     `json:"duration_ms"`
	Errors     int       `json:"errors"`
	Dimensions []dimJSON `json:"dimensions"`
}

type reportJSON struct {
	Norm    string       `json:"norm"`
	Results []resultJSON `json:"results"`
}

func (r *Report) writeJSON(w io.Writer) error {
	out := reportJSON{Norm: r.Norm}
	for _, res := range r.Results {
		item := resultJSON{Name: res.Name, DurationMS: res.Duration.Milliseconds(), Errors: res.Errors}
		for _, d := range res.Correlation.Dimensions {
			item.Dimensions = append(item.Dimensions, dimJSON{
				Name:       d.Name,
				Pearson:    d.Pearson,
				Spearman:   d.Spearman,
				KendallTau: d.KendallTau,
			})
		}
		out.Results = append(out.Results, item)
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func (r *Report) writeLaTeX(w io.Writer) error {
	fmt.Fprintln(w, `\begin{tabular}{l l r r r r}`)
	fmt.Fprintln(w, `\toprule`)
	fmt.Fprintln(w, `Metric & Dimension & Pearson & Spearman & Kendall-$\tau$ & Time (s) \\`)
	fmt.Fprintln(w, `\midrule`)
	for _, res := range r.Results {
		for _, d := range res.Correlation.Dimensions {
			fmt.Fprintf(w, "%s & %s & %.4f & %.4f & %.4f & %.2f \\\\\n",
				escapeLaTeX(res.Name), d.Name,
				d.Pearson, d.Spearman, d.KendallTau,
				res.Duration.Seconds())
		}
	}
	fmt.Fprintln(w, `\bottomrule`)
	fmt.Fprintln(w, `\end{tabular}`)
	return nil
}

func escapeLaTeX(s string) string {
	return strings.NewReplacer("&", `\&`, "_", `\_`, "%", `\%`, "#", `\#`).Replace(s)
}
