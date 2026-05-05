// cmd/ablation reads BGS ablation reports from a directory (default
// `ablation/`) and renders a LaTeX table summarising the coverage-α
// sweep on the held-out development split, with the selected α on
// the test split, plus the legacy precision-side baseline (kept as a
// deprecated comparison row) and the α=0 recall-only baseline.
//
// The canonical configuration (α selected on dev, evaluated on test)
// is flagged with a ★. The recall-only and legacy rows are separated
// by a midrule to set them apart from the dev sweep.
//
// Output target matches the rest of the paper pipeline (default
// `paper/ablation.tex`). Rendering also writes a JSON sidecar so the
// webviewer's Ablation tab can render the same data without
// re-parsing LaTeX.
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
	"strconv"
	"strings"

	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
)

var (
	inputDir         string
	outputTex        string
	outputJSON       string
	kStar            int
	lambdaStar       float64
	alphaStar        float64
	gammaStar        float64
	canonicalEnabled bool
)

func main() {
	flag.StringVar(&inputDir, "input", "ablation", "directory containing ablation JSON reports")
	flag.StringVar(&outputTex, "output", "paper/ablation.tex", "path to write LaTeX table")
	flag.StringVar(&outputJSON, "json", "paper/ablation.json", "path to write JSON sidecar (for webviewer)")
	flag.IntVar(&kStar, "k-star", 0, "canonical k* selected on dev split (0 disables ★)")
	flag.Float64Var(&lambdaStar, "lambda-star", -1.0, "canonical λ* selected on dev split (−1 disables ★)")
	flag.Float64Var(&alphaStar, "alpha-star", -1.0, "canonical α* selected on dev split (−1 disables ★)")
	flag.Float64Var(&gammaStar, "gamma-star", -1.0, "canonical γ* selected on dev split (−1 disables ★)")
	flag.Parse()
	canonicalEnabled = kStar >= 1 && lambdaStar >= 0 && alphaStar >= 0 && gammaStar >= 0

	matches, err := filepath.Glob(filepath.Join(inputDir, "*.json"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	if len(matches) == 0 {
		log.Fatalf("no JSON files in %s", inputDir)
	}
	sort.Strings(matches)

	rows := make([]Row, 0, len(matches))
	for _, p := range matches {
		row, ok := parseRow(p)
		if !ok {
			log.Printf("skipping %s (unrecognised ablation variant)", p)
			continue
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		log.Fatalf("no ablation variants recognised in %s", inputDir)
	}

	sortRows(rows)
	if canonicalEnabled {
		setCanonical(rows, kStar, lambdaStar, alphaStar, gammaStar)
	}

	if err := writeFile(outputTex, renderLatex(rows)); err != nil {
		log.Fatalf("write tex: %v", err)
	}
	if outputJSON != "" {
		raw, err := json.MarshalIndent(rows, "", "  ")
		if err != nil {
			log.Fatalf("marshal json: %v", err)
		}
		if err := writeFile(outputJSON, string(raw)); err != nil {
			log.Fatalf("write json: %v", err)
		}
	}
	log.Printf("ablation: %d rows → %s", len(rows), outputTex)
}

// Row is one line of the ablation table. Exported because the JSON
// sidecar is rendered by the webviewer.
type Row struct {
	Label           string  `json:"label"`
	Source          string  `json:"source"`
	Variant         string  `json:"variant"` // "topk" | "lead" | "coverage" | "diversity" | "combined" | "legacy" | "recall_only"
	RecallTopK      int     `json:"recall_top_k"`
	LeadBiasLambda  float64 `json:"lead_bias_lambda"`
	CoverageAlpha   float64 `json:"coverage_alpha"`
	RedundancyGamma float64 `json:"redundancy_gamma"`
	Split           string  `json:"split"` // "first50" | "last50" | "all" | "dev" | "test" | ""
	Beta            float64 `json:"beta"`
	SalienceFrac    float64 `json:"salience_frac"`
	IsCanonical     bool    `json:"is_canonical"`
	IsRecallOnly    bool    `json:"is_recall_only"`
	IsLegacy        bool    `json:"is_legacy"`
	IsTest          bool    `json:"is_test"`
	Coh             float64 `json:"coh"`
	Con             float64 `json:"con"`
	Flu             float64 `json:"flu"`
	Rel             float64 `json:"rel"`
}

func parseRow(path string) (Row, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read %s: %v", path, err)
	}
	var r eval.Report
	if err := json.Unmarshal(raw, &r); err != nil {
		log.Fatalf("decode %s: %v", path, err)
	}

	row := Row{Source: filepath.Base(path)}
	row.Coh, row.Con, row.Flu, row.Rel = extractDims(r.SummaryLevel)

	if r.Metric != "bgs" {
		return Row{}, false
	}

	// Recall-only sentinel: "no second component" baseline anchored
	// in the original ablation. Equivalent to coverage_alpha=0.
	if strings.Contains(r.Norm, "recall_only=true") {
		row.Variant = "recall_only"
		row.IsRecallOnly = true
		row.Label = "Recall only ($\\alpha=0$)"
		return row, true
	}

	// Legacy precision-side path: deprecated formulation kept for
	// the comparison row.
	if strings.Contains(r.Norm, "legacy_precision=true") {
		row.Variant = "legacy"
		row.IsLegacy = true
		row.Beta = parseField(r.Norm, "beta")
		row.SalienceFrac = parseField(r.Norm, "salience")
		// Legacy norm strings don't yet encode split, so infer from
		// the source filename. Filename convention: bgs_legacy_<split>_b<beta>.json
		// where <split> is "dev" or "test".
		suffix := ""
		switch {
		case strings.Contains(row.Source, "_dev_"):
			row.Split = "dev"
			suffix = " (dev)"
		case strings.Contains(row.Source, "_test_"):
			row.Split = "test"
			suffix = " (test)"
		}
		row.Label = fmt.Sprintf("Legacy F$_\\beta$ ($\\beta=%g$, frac=%.2f)%s", row.Beta, row.SalienceFrac, suffix)
		return row, true
	}

	// Canonical formulation: recall (top-k) · coverage^α · diversity^γ.
	// Variant depends on which knobs are non-default. The Split field
	// distinguishes dev sweep rows from the test-split row that gets ★.
	if strings.Contains(r.Norm, "coverage_alpha=") || strings.Contains(r.Norm, "top_k=") {
		row.RecallTopK = int(parseField(r.Norm, "top_k"))
		if row.RecallTopK == 0 {
			row.RecallTopK = 1
		}
		row.LeadBiasLambda = parseField(r.Norm, "lead_lambda")
		row.CoverageAlpha = parseField(r.Norm, "coverage_alpha")
		row.RedundancyGamma = parseField(r.Norm, "redundancy_gamma")
		row.Split = parseStringField(r.Norm, "split")
		row.IsTest = row.Split == "last50"

		baseRecall := row.RecallTopK == 1 && row.LeadBiasLambda == 0 &&
			row.CoverageAlpha == 0 && row.RedundancyGamma == 0
		switch {
		case baseRecall:
			row.Variant = "coverage" // pure recall baseline groups with coverage section
		case row.LeadBiasLambda > 0 && row.RecallTopK == 1 &&
			row.CoverageAlpha == 0 && row.RedundancyGamma == 0:
			row.Variant = "lead"
		case row.RecallTopK > 1 && row.LeadBiasLambda == 0 &&
			row.CoverageAlpha == 0 && row.RedundancyGamma == 0:
			row.Variant = "topk"
		case row.CoverageAlpha > 0 && row.RedundancyGamma > 0:
			row.Variant = "combined"
		case row.RedundancyGamma > 0:
			row.Variant = "diversity"
		default:
			row.Variant = "coverage"
		}

		var paramStr string
		switch {
		case baseRecall:
			paramStr = "Recall baseline ($k=1$, $\\lambda=\\alpha=\\gamma=0$)"
		case row.Variant == "lead":
			paramStr = fmt.Sprintf("Lead-bias ($\\lambda=%g$)", row.LeadBiasLambda)
		case row.Variant == "topk":
			paramStr = fmt.Sprintf("Top-$k$ recall ($k=%d$)", row.RecallTopK)
		case row.Variant == "diversity":
			paramStr = fmt.Sprintf("Diversity ($\\gamma=%g$)", row.RedundancyGamma)
		case row.Variant == "combined":
			paramStr = fmt.Sprintf("$\\alpha=%g$, $\\gamma=%g$", row.CoverageAlpha, row.RedundancyGamma)
		default:
			paramStr = fmt.Sprintf("Coverage ($\\alpha=%g$)", row.CoverageAlpha)
		}

		switch row.Split {
		case "first50":
			row.Label = paramStr + " (dev)"
		case "last50":
			row.Label = paramStr + " (test)"
		case "all":
			row.Label = paramStr + " (full set)"
		default:
			row.Label = paramStr
		}
		return row, true
	}

	// Legacy on-disk format (β / salience without legacy_precision flag),
	// produced by the pre-redesign cmd/bgs. Treat as legacy variant.
	if strings.Contains(r.Norm, "beta=") {
		row.Variant = "legacy"
		row.IsLegacy = true
		row.Beta = parseField(r.Norm, "beta")
		row.SalienceFrac = parseField(r.Norm, "salience")
		row.Label = fmt.Sprintf("Legacy F$_\\beta$ ($\\beta=%g$, frac=%.2f)", row.Beta, row.SalienceFrac)
		return row, true
	}

	return Row{}, false
}

// setCanonical marks the test-split row matching the dev-selected
// (k*, λ*, α*, γ*) combination.
func setCanonical(rows []Row, kStar int, lambdaStar, alphaStar, gammaStar float64) {
	for i := range rows {
		if rows[i].Split != "last50" {
			continue
		}
		switch rows[i].Variant {
		case "lead", "topk", "coverage", "diversity", "combined":
		default:
			continue
		}
		if rows[i].RecallTopK == kStar &&
			approxEqual(rows[i].LeadBiasLambda, lambdaStar) &&
			approxEqual(rows[i].CoverageAlpha, alphaStar) &&
			approxEqual(rows[i].RedundancyGamma, gammaStar) {
			rows[i].IsCanonical = true
			return
		}
	}
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

func parseField(norm, key string) float64 {
	for _, p := range strings.Split(norm, ",") {
		p = strings.TrimSpace(p)
		prefix := key + "="
		if !strings.HasPrefix(p, prefix) {
			continue
		}
		v, err := strconv.ParseFloat(p[len(prefix):], 64)
		if err == nil {
			return v
		}
	}
	return 0
}

func approxEqual(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-6
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

// variantRank orders the table sections. Lead bias is now the
// canonical second component, so it leads the dev / test sections;
// the rejected ablations (top-k, coverage, diversity, legacy) follow.
func variantRank(r Row) int {
	switch {
	case r.Variant == "lead" && r.Split == "first50":
		return 0
	case r.Variant == "topk" && r.Split == "first50":
		return 1
	case r.Variant == "coverage" && r.Split == "first50":
		return 2
	case r.Variant == "diversity" && r.Split == "first50":
		return 3
	case r.Variant == "combined" && r.Split == "first50":
		return 4
	case r.Variant == "legacy" && r.Split == "dev":
		return 5
	case r.Variant == "lead" && r.Split == "last50":
		return 6
	case r.Variant == "topk" && r.Split == "last50":
		return 7
	case r.Variant == "coverage" && r.Split == "last50":
		return 8
	case r.Variant == "diversity" && r.Split == "last50":
		return 9
	case r.Variant == "combined" && r.Split == "last50":
		return 10
	case r.Variant == "legacy" && r.Split == "test":
		return 11
	case r.Split == "all":
		return 12
	case r.IsRecallOnly:
		return 13
	case r.IsLegacy:
		return 14
	}
	return 15
}

func sortRows(rows []Row) {
	sort.SliceStable(rows, func(i, j int) bool {
		ri, rj := variantRank(rows[i]), variantRank(rows[j])
		if ri != rj {
			return ri < rj
		}
		// Within a section: ascending λ (lead-bias), then k (top-k),
		// then α (coverage), then γ (diversity), then β, then frac.
		if rows[i].LeadBiasLambda != rows[j].LeadBiasLambda {
			return rows[i].LeadBiasLambda < rows[j].LeadBiasLambda
		}
		if rows[i].RecallTopK != rows[j].RecallTopK {
			return rows[i].RecallTopK < rows[j].RecallTopK
		}
		if rows[i].CoverageAlpha != rows[j].CoverageAlpha {
			return rows[i].CoverageAlpha < rows[j].CoverageAlpha
		}
		if rows[i].RedundancyGamma != rows[j].RedundancyGamma {
			return rows[i].RedundancyGamma < rows[j].RedundancyGamma
		}
		if rows[i].Beta != rows[j].Beta {
			return rows[i].Beta < rows[j].Beta
		}
		return rows[i].SalienceFrac < rows[j].SalienceFrac
	})
}

func renderLatex(rows []Row) string {
	var b strings.Builder
	fmt.Fprintln(&b, `\begin{table*}[t]`)
	fmt.Fprintln(&b, `\centering`)
	fmt.Fprintln(&b, `\caption{Ablation of BGS on SummEval (summary-level Spearman $\rho$). The canonical metric is recall with an exponential lead-bias prior on source sentences: $\mathrm{score} = \mathrm{mean}_j\, \max_i\, w(i)\cdot\cos(c_j, s_i)$ with $w(i)=\exp(-\lambda \cdot i / n)$. Four hyperparameters are tested independently on a held-out development split (50 articles): $\lambda$ (lead-bias decay), $k$ (top-$k$ recall aggregation, replacing max with the mean of top-$k$ cosines), $\alpha$ (anchor-coverage exponent), $\gamma$ (within-summary-diversity exponent). The dev-selected canonical row is marked $\star$ and re-evaluated on the test split (50 articles). Lead bias is the only tested component that improves dev mean Spearman over the recall baseline ($\lambda=0$); top-$k$ recall, anchor coverage, and within-summary diversity all degrade dev mean and are reported as negative ablations. Legacy F$_\beta$ rows reproduce the original bidirectional formulation.}`)
	fmt.Fprintln(&b, `\label{tab:ablation_bgs}`)
	fmt.Fprintln(&b, `\begin{tabular}{lrrrr}`)
	fmt.Fprintln(&b, `\toprule`)
	fmt.Fprintln(&b, `Variant & Coh & Con & Flu & Rel \\`)
	fmt.Fprintln(&b, `\midrule`)
	prevRank := -1
	for _, r := range rows {
		rank := variantRank(r)
		if prevRank != -1 && rank != prevRank {
			fmt.Fprintln(&b, `\midrule`)
		}
		prevRank = rank
		label := r.Label
		if r.IsCanonical {
			label += " $\\star$"
		}
		fmt.Fprintf(&b, "%s & %s & %s & %s & %s \\\\\n",
			label,
			fmtCell(r.Coh), fmtCell(r.Con), fmtCell(r.Flu), fmtCell(r.Rel))
	}
	fmt.Fprintln(&b, `\bottomrule`)
	fmt.Fprintln(&b, `\end{tabular}`)
	fmt.Fprintln(&b, `\end{table*}`)
	return b.String()
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
