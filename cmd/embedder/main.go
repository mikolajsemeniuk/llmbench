// cmd/embedder reads LGS embedder-ablation reports from a directory
// (default `ablation/`, files matching `lgs_embedder_*.json`) and
// renders a LaTeX table comparing the canonical metric (λ=BGS_LAMBDA)
// across multiple sentence-embedder backbones. Establishes that LGS
// is a metric design — not specific to one embedder — by reporting
// summary-level Spearman ρ across the four SummEval dimensions for
// each tested embedder.
//
// Output: `paper/embedders.gen.tex`.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
)

var (
	inputDir    string
	outputTex   string
	canonicalID string
)

func main() {
	flag.StringVar(&inputDir, "input", "ablation", "directory containing lgs_embedder_*.json reports")
	flag.StringVar(&outputTex, "output", "paper/embedders.gen.tex", "path to write LaTeX table")
	flag.StringVar(&canonicalID, "canonical", "nomic-embed-text", "embedder model used in the headline canonical run (gets ★)")
	flag.Parse()

	matches, err := filepath.Glob(filepath.Join(inputDir, "lgs_embedder_*.json"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	if len(matches) == 0 {
		log.Fatalf("no lgs_embedder_*.json files in %s — run `make benchmark-embedder-ablation` first", inputDir)
	}
	sort.Strings(matches)

	rows := make([]Row, 0, len(matches))
	for _, p := range matches {
		row, ok := parseRow(p, canonicalID)
		if !ok {
			log.Printf("skipping %s (unrecognised report)", p)
			continue
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		log.Fatalf("no embedder rows recognised in %s", inputDir)
	}

	sortRows(rows)

	if err := writeFile(outputTex, renderLatex(rows)); err != nil {
		log.Fatalf("write tex: %v", err)
	}
	log.Printf("embedder ablation: %d rows → %s", len(rows), outputTex)
}

// Row is one line of the embedder table.
type Row struct {
	Embedder    string
	IsCanonical bool
	Coh         float64
	Con         float64
	Flu         float64
	Rel         float64
	Mean        float64
}

func parseRow(path, canonical string) (Row, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read %s: %v", path, err)
	}
	var r eval.Report
	if err := json.Unmarshal(raw, &r); err != nil {
		log.Fatalf("decode %s: %v", path, err)
	}
	if r.Metric != "lgs" {
		return Row{}, false
	}
	embed := parseStringField(r.Norm, "embed_model")
	if embed == "" {
		return Row{}, false
	}
	row := Row{
		Embedder:    embed,
		IsCanonical: stripVersion(embed) == stripVersion(canonical),
	}
	row.Coh, row.Con, row.Flu, row.Rel = extractDims(r.SummaryLevel)
	row.Mean = (row.Coh + row.Con + row.Flu + row.Rel) / 4.0
	return row, true
}

// stripVersion drops Ollama's ":tag" suffix so "nomic-embed-text" and
// "nomic-embed-text:latest" compare equal.
func stripVersion(s string) string {
	if i := strings.Index(s, ":"); i >= 0 {
		return s[:i]
	}
	return s
}

func parseStringField(norm, key string) string {
	for _, p := range strings.Split(norm, ",") {
		p = strings.TrimSpace(p)
		prefix := key + "="
		if strings.HasPrefix(p, prefix) {
			return p[len(prefix):]
		}
	}
	return ""
}

func extractDims(c eval.Correlation) (coh, con, flu, rel float64) {
	for _, d := range c.Dimensions {
		switch d.Name {
		case "coherence":
			coh = d.Spearman
		case "consistency":
			con = d.Spearman
		case "fluency":
			flu = d.Spearman
		case "relevance":
			rel = d.Spearman
		}
	}
	return
}

func sortRows(rows []Row) {
	// Order by ascending mean ρ. Canonical row (★) appears wherever
	// the mean places it — no special promotion.
	sort.SliceStable(rows, func(i, j int) bool {
		return rows[i].Mean < rows[j].Mean
	})
}

func renderLatex(rows []Row) string {
	var b strings.Builder
	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintln(&b, `\caption{Embedder ablation: canonical LGS ($\lambda=0.5$) on the full SummEval set with four sentence-embedder backbones (Ollama). Reports summary-level Spearman $\rho$ per dimension and the mean across the four. Canonical embedder (used in the headline tables) marked $\star$.}`)
	fmt.Fprintln(&b, `\label{tab:ablation_embedder}`)
	fmt.Fprintln(&b, `\begin{tabular}{lrrrrr}`)
	fmt.Fprintln(&b, `\toprule`)
	fmt.Fprintln(&b, `Embedder & Coh & Con & Flu & Rel & Mean \\`)
	fmt.Fprintln(&b, `\midrule`)
	for _, r := range rows {
		label := embedderDisplay(r.Embedder)
		if r.IsCanonical {
			label += " $\\star$"
		}
		fmt.Fprintf(&b, "%s & %s & %s & %s & %s & %s \\\\\n",
			label,
			fmtCell(r.Coh), fmtCell(r.Con), fmtCell(r.Flu), fmtCell(r.Rel), fmtCell(r.Mean))
	}
	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
}

// embedderDisplay maps Ollama model identifiers to human-readable
// labels including parameter count for the paper.
func embedderDisplay(model string) string {
	clean := stripVersion(model)
	switch clean {
	case "nomic-embed-text":
		return `\texttt{nomic-embed-text} (137M)`
	case "mxbai-embed-large":
		return `\texttt{mxbai-embed-large} (335M)`
	case "bge-m3":
		return `\texttt{bge-m3} (567M)`
	case "all-minilm":
		return `\texttt{all-minilm} (23M)`
	}
	return `\texttt{` + clean + `}`
}

func fmtCell(x float64) string {
	s := fmt.Sprintf("%.3f", x)
	if x >= 0 && x < 1 {
		return strings.TrimPrefix(s, "0")
	}
	if x > -1 && x < 0 {
		return "-" + strings.TrimPrefix(s[1:], "0")
	}
	return s
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
