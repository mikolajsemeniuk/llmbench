// cmd/frontier reads metric reports from output/*.json and renders a
// LaTeX figure (pgfplots scatter) showing the cost--accuracy frontier:
// per-sample runtime in milliseconds (x-axis, log scale) versus mean
// summary-level Spearman ρ across the four SummEval dimensions
// (y-axis). Used to support the deployment thesis that LGS occupies a
// distinct point on the cost--correlation frontier.
//
// Dimensional metrics (geval_*, unieval_*) are aggregated into a
// single point per family: per-sample runtime is summed across the
// four dimension-specific runs (since a deployed dimensional scorer
// must invoke one model per dimension), and mean ρ uses the diagonal
// (each dim file's correlation against its own dimension).
//
// Output: paper/frontier.gen.tex.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
)

var (
	inputDir  string
	outputTex string
	oursLabel string
)

func main() {
	flag.StringVar(&inputDir, "input", "output", "directory containing metric JSON reports")
	flag.StringVar(&outputTex, "output", "paper/frontier.gen.tex", "path to write LaTeX figure")
	flag.StringVar(&oursLabel, "ours", "lgs", "metric base name to highlight as ours")
	flag.Parse()

	matches, err := filepath.Glob(filepath.Join(inputDir, "*.json"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	sort.Strings(matches)

	groups := map[string][]eval.Report{}
	for _, p := range matches {
		raw, err := os.ReadFile(p)
		if err != nil {
			log.Fatalf("read %s: %v", p, err)
		}
		var r eval.Report
		if err := json.Unmarshal(raw, &r); err != nil {
			log.Fatalf("decode %s: %v", p, err)
		}
		base := baseName(r.Metric)
		groups[base] = append(groups[base], r)
	}

	points := make([]Point, 0, len(groups))
	for base, reports := range groups {
		pt := pointForGroup(base, reports)
		if pt.RhoMean == 0 && pt.RuntimeMs == 0 {
			continue
		}
		points = append(points, pt)
	}
	sort.Slice(points, func(i, j int) bool { return points[i].Label < points[j].Label })

	if err := writeFile(outputTex, renderLatex(points)); err != nil {
		log.Fatalf("write tex: %v", err)
	}
	log.Printf("frontier: %d metrics → %s", len(points), outputTex)
}

type Point struct {
	Label     string
	Base      string
	RuntimeMs float64
	RhoMean   float64
	IsOurs    bool
}

// baseName collapses dimensional metric names like "geval_coherence"
// to "geval" so all four dim files for the same family map into one
// group. Non-dimensional metrics keep their full name.
func baseName(metric string) string {
	for _, dim := range []string{"_coherence", "_consistency", "_fluency", "_relevance"} {
		if strings.HasSuffix(metric, dim) {
			return strings.TrimSuffix(metric, dim)
		}
	}
	return metric
}

func pointForGroup(base string, reports []eval.Report) Point {
	pt := Point{Base: base, Label: displayName(base), IsOurs: base == oursLabel}

	if len(reports) == 1 {
		r := reports[0]
		if r.Samples == 0 {
			return pt
		}
		pt.RuntimeMs = 1000.0 * r.RuntimeSec / float64(r.Samples)
		pt.RhoMean = meanSpearman(r.SummaryLevel)
		return pt
	}

	// Dimensional metric: sum per-sample runtimes across dim files
	// (deployment must run all four scorers); mean ρ uses the
	// diagonal.
	dimOfFile := map[string]string{}
	for _, r := range reports {
		for _, d := range []string{"coherence", "consistency", "fluency", "relevance"} {
			if strings.HasSuffix(r.Metric, "_"+d) {
				dimOfFile[r.Metric] = d
			}
		}
	}

	var totalMs, sumDiag float64
	var nDiag int
	for _, r := range reports {
		if r.Samples == 0 {
			continue
		}
		totalMs += 1000.0 * r.RuntimeSec / float64(r.Samples)
		dim := dimOfFile[r.Metric]
		if dim == "" {
			continue
		}
		for _, d := range r.SummaryLevel.Dimensions {
			if d.Name == dim {
				sumDiag += d.Spearman
				nDiag++
				break
			}
		}
	}
	if nDiag == 0 {
		return pt
	}
	pt.RuntimeMs = totalMs
	pt.RhoMean = sumDiag / float64(nDiag)
	return pt
}

func meanSpearman(c eval.Correlation) float64 {
	if len(c.Dimensions) == 0 {
		return 0
	}
	var s float64
	for _, d := range c.Dimensions {
		s += d.Spearman
	}
	return s / float64(len(c.Dimensions))
}

var displayNames = map[string]string{
	"bleu":        "BLEU",
	"rouge":       "ROUGE-L",
	"chrf":        "ChrF",
	"meteor":      "METEOR",
	"smartstring": "SMART-String",
	"smartmodel":  "SMART-Model",
	"embedscorer": "EmbedScorer",
	"bertscore":   "BERTScore",
	"moverscore":  "MoverScore",
	"bartscore":   "BARTScore",
	"gptscore":    "GPTScore",
	"unieval":     "UniEval",
	"geval":       "G-Eval",
	"lgs":         "LGS",
}

func displayName(base string) string {
	if v, ok := displayNames[base]; ok {
		return v
	}
	return base
}

func renderLatex(points []Point) string {
	var b strings.Builder
	fmt.Fprintln(&b, `\begin{figure}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintln(&b, `\begin{tikzpicture}`)
	fmt.Fprintln(&b, `\begin{axis}[`)
	fmt.Fprintln(&b, `  width=0.95\linewidth, height=6.0cm,`)
	fmt.Fprintln(&b, `  xlabel={Per-sample runtime (ms, log scale)},`)
	fmt.Fprintln(&b, `  ylabel={Mean Spearman $\rho$},`)
	fmt.Fprintln(&b, `  xmode=log, log basis x=10,`)
	fmt.Fprintln(&b, `  xmin=0.05, xmax=20000,`)
	fmt.Fprintln(&b, `  ymin=0.0, ymax=0.50,`)
	fmt.Fprintln(&b, `  grid=both, grid style={gray!15},`)
	fmt.Fprintln(&b, `  tick label style={font=\scriptsize},`)
	fmt.Fprintln(&b, `  label style={font=\scriptsize},`)
	fmt.Fprintln(&b, `  every node near coord/.append style={font=\tiny}`)
	fmt.Fprintln(&b, `]`)

	// Baseline points
	fmt.Fprintln(&b, `\addplot[only marks, mark=o, mark size=2pt, color=gray!70!black,`)
	fmt.Fprintln(&b, `  nodes near coords, point meta=explicit symbolic,`)
	fmt.Fprintln(&b, `  every node near coord/.append style={anchor=west, xshift=2pt, font=\tiny, color=black}]`)
	fmt.Fprintln(&b, `  table[x=rt, y=rho, meta=label, col sep=comma, row sep=\\] {`)
	fmt.Fprintln(&b, `  rt, rho, label \\`)
	for _, p := range points {
		if p.IsOurs {
			continue
		}
		fmt.Fprintf(&b, "  %.2f, %.3f, %s \\\\\n", clampLog(p.RuntimeMs), p.RhoMean, p.Label)
	}
	fmt.Fprintln(&b, `  };`)

	// Ours, highlighted
	for _, p := range points {
		if !p.IsOurs {
			continue
		}
		fmt.Fprintf(&b, "\\addplot[only marks, mark=star, mark size=4pt, color=orange!80!black,\n")
		fmt.Fprintf(&b, "  nodes near coords, point meta=explicit symbolic,\n")
		fmt.Fprintf(&b, "  every node near coord/.append style={anchor=west, xshift=3pt, font=\\tiny\\bfseries, color=orange!50!black}]\n")
		fmt.Fprintf(&b, "  table[x=rt, y=rho, meta=label, col sep=comma, row sep=\\\\] {\n")
		fmt.Fprintf(&b, "  rt, rho, label \\\\\n")
		fmt.Fprintf(&b, "  %.2f, %.3f, %s \\\\\n", clampLog(p.RuntimeMs), p.RhoMean, p.Label)
		fmt.Fprintf(&b, "  };\n")
	}

	fmt.Fprintln(&b, `\end{axis}`)
	fmt.Fprintln(&b, `\end{tikzpicture}`)
	fmt.Fprintln(&b, `\caption{Cost--accuracy frontier on \textsc{SummEval}. Each point is one metric: x-axis is the per-sample wall-clock runtime under our reproduction (Section~\ref{sec:setup-cost}), in milliseconds on a single Apple-silicon CPU thread, on a log scale; y-axis is the mean summary-level Spearman $\rho$ across the four \textsc{SummEval} dimensions. LGS (orange star) sits in the lower-cost band of the upper-correlation cluster, distinct from the LLM-judge and fine-tuned-encoder metrics in the upper-right.}`)
	fmt.Fprintln(&b, `\label{fig:frontier}`)
	fmt.Fprintln(&b, `\end{figure}`)
	return b.String()
}

func clampLog(x float64) float64 {
	if x < 0.05 {
		return 0.05
	}
	if math.IsInf(x, 0) || math.IsNaN(x) {
		return 0.05
	}
	return x
}

func writeFile(path, content string) error {
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("mkdir %s: %w", dir, err)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	_, err = io.WriteString(f, content)
	return err
}
