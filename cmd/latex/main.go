package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"text/template"
)

// ---------------------------------------------------------------------------
// Minimal structs — only what we need for LaTeX export.
// In production this imports from llmbench; here kept standalone so the
// exporter compiles without the full module.
// ---------------------------------------------------------------------------

type CompareReport struct {
	GeneratedAt string             `json:"generated_at"`
	ModelA      string             `json:"model_a"`
	ModelB      string             `json:"model_b"`
	Aggregate   []MetricComparison `json:"aggregate"`
	PerLevel    []LevelComparison  `json:"per_level"`
	PerTask     []TaskComparison   `json:"per_task"`
	Raw         struct {
		A Report `json:"a"`
		B Report `json:"b"`
	} `json:"raw"`
}

type MetricComparison struct {
	Name             string  `json:"name"`
	FullName         string  `json:"full_name"`
	HigherIsBetter   bool    `json:"higher_is_better"`
	ValueA           float64 `json:"value_a"`
	ValueB           float64 `json:"value_b"`
	Delta            float64 `json:"delta"`
	WilcoxonU        float64 `json:"wilcoxon_u"`
	PValue           float64 `json:"p_value"`
	PValueCorrected  float64 `json:"p_value_corrected"`
	CorrectionMethod string  `json:"correction_method"`
	Significance     string  `json:"significance"`
	EffectSize       float64 `json:"effect_size_r"`
	EffectLabel      string  `json:"effect_label"`
}

type LevelComparison struct {
	Level string  `json:"level"`
	ESRA  float64 `json:"esr_a"`
	ESRB  float64 `json:"esr_b"`
	TSAA  float64 `json:"tsa_a"`
	TSAB  float64 `json:"tsa_b"`
	CHRA  float64 `json:"chr_a"`
	CHRB  float64 `json:"chr_b"`
	RunsA int     `json:"runs_a"`
	RunsB int     `json:"runs_b"`
}

type TaskComparison struct {
	TaskID string  `json:"task_id"`
	Level  string  `json:"level"`
	ESRA   float64 `json:"esr_a"`
	ESRB   float64 `json:"esr_b"`
	Delta  float64 `json:"delta"`
}

type Report struct {
	Metadata struct {
		Provider    string `json:"provider"`
		Model       string `json:"model"`
		Timestamp   string `json:"timestamp"`
		TotalTasks  int    `json:"total_tasks"`
		RunsPerTask int    `json:"runs_per_task"`
		TotalRuns   int    `json:"total_runs"`
		Seed        int64  `json:"random_seed"`
	} `json:"metadata"`
	Metrics struct {
		ESR   float64    `json:"esr"`
		ESRCI [2]float64 `json:"esr_ci_95"`
	} `json:"aggregate"`
	RAG struct {
		MeanPrecisionAtK float64 `json:"mean_precision_at_k"`
		MeanRecallAtK    float64 `json:"mean_recall_at_k"`
		MeanMRR          float64 `json:"mean_mrr"`
		MeanNDCGAtK      float64 `json:"mean_ndcg_at_k"`
		MeanFScoreAtK    float64 `json:"mean_f1_at_k"`
	} `json:"rag_quality"`
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

var (
	flagFile   string
	flagOutput string
)

func main() {
	flag.StringVar(&flagFile, "file", "compare.json", "Path to compare JSON")
	flag.StringVar(&flagOutput, "output", "tables.tex", "Output .tex file")
	flag.Parse()

	raw, err := os.ReadFile(flagFile)
	if err != nil {
		log.Fatalf("cannot read %s: %v", flagFile, err)
	}

	var cr CompareReport
	if err := json.Unmarshal(raw, &cr); err != nil {
		log.Fatalf("cannot parse: %v", err)
	}

	funcMap := template.FuncMap{
		"pct":       func(v float64) string { return fmt.Sprintf("%.1f", v*100) },
		"f2":        func(v float64) string { return fmt.Sprintf("%.2f", v) },
		"f3":        func(v float64) string { return fmt.Sprintf("%.3f", v) },
		"f4":        func(v float64) string { return fmt.Sprintf("%.4f", v) },
		"sci":       fmtSci,
		"delta":     fmtDelta,
		"deltaPct":  fmtDeltaPct,
		"sig":       fmtSig,
		"esc":       texEscape,
		"shortName": shortModelName,
		"levelTag":  levelTag,
		"isTested":  isTested,
		"add":       func(a, b int) int { return a + b },
		"half":      func(tc []TaskComparison) int { return len(tc) / 2 },
		"taskAt":    func(tc []TaskComparison, i int) TaskComparison { return tc[i] },
		"lt":        func(a, b int) bool { return a < b },
		"seq": func(n int) []int {
			s := make([]int, n)
			for i := range s {
				s[i] = i
			}
			return s
		},
		// Helpers for per-task win/loss/tie counts.
		"wins":   func(tc []TaskComparison) int { return countDelta(tc, 1) },
		"ties":   func(tc []TaskComparison) int { return countDelta(tc, 0) },
		"losses": func(tc []TaskComparison) int { return countDelta(tc, -1) },
		// Filter tested metrics (those with Holm-Bonferroni correction).
		"tested":  filterTested,
		"scalars": filterScalars,
	}

	tpl, err := template.New("latex").Funcs(funcMap).Parse(latexTemplate)
	if err != nil {
		log.Fatalf("template error: %v", err)
	}

	f, err := os.Create(flagOutput)
	if err != nil {
		log.Fatalf("cannot create %s: %v", flagOutput, err)
	}
	defer f.Close()

	if err := tpl.Execute(f, cr); err != nil {
		log.Fatalf("template execute: %v", err)
	}
	log.Printf("wrote %s", flagOutput)
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

func texEscape(s string) string {
	r := strings.NewReplacer(
		"_", `\_`,
		"%", `\%`,
		"&", `\&`,
		"#", `\#`,
		"$", `\$`,
		"{", `\{`,
		"}", `\}`,
		"~", `\textasciitilde{}`,
	)
	return r.Replace(s)
}

func shortModelName(s string) string {
	// "ollama/qwen2.5:3b-instruct" → "Qwen2.5-3B"
	parts := strings.SplitN(s, "/", 2)
	name := parts[len(parts)-1]
	name = strings.ReplaceAll(name, ":latest", "")
	return texEscape(name)
}

func levelTag(s string) string {
	switch s {
	case "L1-diagnostic":
		return "L1"
	case "L2-repair":
		return "L2"
	case "L3-multi-step":
		return "L3"
	}
	return s
}

func fmtSci(v float64) string {
	if v == 0 {
		return "$< 10^{-15}$"
	}
	if v >= 0.001 {
		return fmt.Sprintf("%.4f", v)
	}
	exp := int(math.Floor(math.Log10(math.Abs(v))))
	mantissa := v / math.Pow(10, float64(exp))
	return fmt.Sprintf("$%.2f{\\times}10^{%d}$", mantissa, exp)
}

func fmtDelta(v float64) string {
	if v > 0 {
		return fmt.Sprintf("+%.4f", v)
	}
	if v < 0 {
		return fmt.Sprintf("%.4f", v) // minus sign included
	}
	return "0"
}

func fmtDeltaPct(v float64) string {
	pct := v * 100
	if pct > 0 {
		return fmt.Sprintf("+%.1f", pct)
	}
	if pct < 0 {
		return fmt.Sprintf("%.1f", pct)
	}
	return "0.0"
}

func fmtSig(s string) string {
	switch s {
	case "***":
		return `\sigHigh{***}`
	case "**":
		return `\sigMed{**}`
	case "*":
		return `\sigLow{*}`
	case "n.s.":
		return `n.s.`
	default:
		return "---"
	}
}

func isTested(m MetricComparison) bool {
	return m.CorrectionMethod != "" && m.CorrectionMethod != "n/a"
}

func filterTested(ms []MetricComparison) []MetricComparison {
	var out []MetricComparison
	for _, m := range ms {
		if isTested(m) {
			out = append(out, m)
		}
	}
	return out
}

func filterScalars(ms []MetricComparison) []MetricComparison {
	var out []MetricComparison
	for _, m := range ms {
		if !isTested(m) {
			out = append(out, m)
		}
	}
	return out
}

func countDelta(tc []TaskComparison, dir int) int {
	n := 0
	for _, t := range tc {
		switch {
		case dir > 0 && t.Delta > 0:
			n++
		case dir < 0 && t.Delta < 0:
			n++
		case dir == 0 && t.Delta == 0:
			n++
		}
	}
	return n
}

// ---------------------------------------------------------------------------
// LaTeX template
// ---------------------------------------------------------------------------

const latexTemplate = `%% ==========================================================================
%% Auto-generated by llmbench/cmd/latex — do not edit manually.
%% Generated: {{.GeneratedAt}}
%% Model A: {{esc .ModelA}}
%% Model B: {{esc .ModelB}}
%% ==========================================================================
%%
%% Required packages (add to your preamble):
%%   \usepackage{booktabs}
%%   \usepackage{multirow}
%%   \usepackage{xcolor}
%%   \usepackage{siunitx}
%%
%% Significance color commands (add to preamble):
%%   \newcommand{\sigHigh}[1]{\textcolor{green!60!black}{\textbf{#1}}}
%%   \newcommand{\sigMed}[1]{\textcolor{green!60!black}{\textbf{#1}}}
%%   \newcommand{\sigLow}[1]{\textcolor{orange!80!black}{#1}}
%%


%% ==========================================================================
%% TABLE 1: Aggregate Metrics — Tested (Wilcoxon + Holm-Bonferroni)
%% ==========================================================================
\begin{table*}[t]
\centering
\caption{Aggregate benchmark results comparing {{shortName .ModelA}} (A) and {{shortName .ModelB}} (B) across 60 Kubernetes diagnostic tasks. Metrics marked with $\uparrow$ are higher-is-better; $\downarrow$ lower-is-better. Statistical significance assessed via Wilcoxon rank-sum test with Holm--Bonferroni correction ($m={{len (tested .Aggregate)}}$). Effect size: rank-biserial~$r$.}
\label{tab:aggregate}
\small
\begin{tabular}{@{}llrrrcrrr@{}}
\toprule
Metric & Dir. & \multicolumn{1}{c}{A} & \multicolumn{1}{c}{B} & $\Delta$ & $p_{\text{corr}}$ & Sig. & $r$ & Effect \\
\midrule
{{- range tested .Aggregate}}
{{.Name}} & {{if .HigherIsBetter}}$\uparrow${{else}}$\downarrow${{end}} & {{f4 .ValueA}} & {{f4 .ValueB}} & {{delta .Delta}} & {{sci .PValueCorrected}} & {{sig .Significance}} & {{f3 .EffectSize}} & {{.EffectLabel}} \\
{{- end}}
\bottomrule
\end{tabular}
\end{table*}


%% ==========================================================================
%% TABLE 2: Scalar Metrics (no per-run vectors — descriptive only)
%% ==========================================================================
\begin{table}[t]
\centering
\caption{Supplementary scalar metrics. CES$=-1$ denotes zero-cost local deployment (undefined ratio). These metrics lack per-run distributions and are reported descriptively without statistical testing.}
\label{tab:scalars}
\small
\begin{tabular}{@{}llrrl@{}}
\toprule
Metric & Dir. & \multicolumn{1}{c}{A} & \multicolumn{1}{c}{B} & $\Delta$ \\
\midrule
{{- range scalars .Aggregate}}
{{- if and (ne .Name "CES") (ne .Name "CCR") (ne .Name "CTR")}}
{{.Name}} & {{if .HigherIsBetter}}$\uparrow${{else}}$\downarrow${{end}} & {{f4 .ValueA}} & {{f4 .ValueB}} & {{delta .Delta}} \\
{{- end}}
{{- end}}
\bottomrule
\end{tabular}
\end{table}


%% ==========================================================================
%% TABLE 3: Per-Level Breakdown
%% ==========================================================================
\begin{table}[t]
\centering
\caption{Per-level performance breakdown. L1: diagnostic (identify the fault), L2: repair (diagnose and fix), L3: multi-step (cross-resource resolution). $n$ = runs per model per level.}
\label{tab:perlevel}
\small
\begin{tabular}{@{}l rr rr rr r@{}}
\toprule
 & \multicolumn{2}{c}{ESR $\uparrow$} & \multicolumn{2}{c}{TSA $\uparrow$} & \multicolumn{2}{c}{CHR $\downarrow$} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Level & \multicolumn{1}{c}{A} & \multicolumn{1}{c}{B} & \multicolumn{1}{c}{A} & \multicolumn{1}{c}{B} & \multicolumn{1}{c}{A} & \multicolumn{1}{c}{B} & $n$ \\
\midrule
{{- range .PerLevel}}
{{levelTag .Level}} & {{pct .ESRA}} & {{pct .ESRB}} & {{pct .TSAA}} & {{pct .TSAB}} & {{pct .CHRA}} & {{pct .CHRB}} & {{.RunsA}} \\
{{- end}}
\bottomrule
\end{tabular}
\end{table}


%% ==========================================================================
%% TABLE 4: Per-Task ESR — Win/Loss/Tie Summary
%% ==========================================================================
\begin{table}[t]
\centering
\caption{Per-task ESR comparison ({{shortName .ModelA}} vs.\ {{shortName .ModelB}}). A \textit{win} means model~A achieved higher ESR on that task.}
\label{tab:pertask-summary}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
 & Wins (A) & Ties & Losses (A) \\
\midrule
All tasks ($n={{len .PerTask}}$) & {{wins .PerTask}} & {{ties .PerTask}} & {{losses .PerTask}} \\
\bottomrule
\end{tabular}
\end{table}


%% ==========================================================================
%% TABLE 5: RAG Quality Metrics (task design validation)
%% ==========================================================================
\begin{table}[t]
\centering
\caption{RAG retrieval quality metrics averaged over all 60~tasks. These validate the benchmark design rather than model performance: $R@K=1.0$ by construction (all relevant documents included).}
\label{tab:rag}
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & P@K & R@K & F$_1$@K & MRR & NDCG@K \\
\midrule
Benchmark & {{f3 .Raw.A.RAG.MeanPrecisionAtK}} & {{f3 .Raw.A.RAG.MeanRecallAtK}} & {{f3 .Raw.A.RAG.MeanFScoreAtK}} & {{f3 .Raw.A.RAG.MeanMRR}} & {{f3 .Raw.A.RAG.MeanNDCGAtK}} \\
\bottomrule
\end{tabular}
\end{table}


%% ==========================================================================
%% TABLE 6: Experimental Setup
%% ==========================================================================
\begin{table}[t]
\centering
\caption{Experimental configuration. Both models evaluated under identical conditions on the same hardware with deterministic sampling (temperature$=0$, seed$={{.Raw.A.Metadata.Seed}}$).}
\label{tab:setup}
\small
\begin{tabular}{@{}ll@{}}
\toprule
Parameter & Value \\
\midrule
Model A            & {{esc .ModelA}} \\
Model B            & {{esc .ModelB}} \\
Tasks              & {{.Raw.A.Metadata.TotalTasks}} (L1:20, L2:20, L3:20) \\
Runs per task      & {{.Raw.A.Metadata.RunsPerTask}} \\
Total runs / model & {{.Raw.A.Metadata.TotalRuns}} \\
Random seed        & {{.Raw.A.Metadata.Seed}} \\
Temperature        & 0 \\
Correction         & Holm--Bonferroni ($m={{len (tested .Aggregate)}}$) \\
Effect size        & Rank-biserial $r$ \\
\bottomrule
\end{tabular}
\end{table}


%% ==========================================================================
%% APPENDIX TABLE: Full Per-Task ESR
%% ==========================================================================
\begin{table*}[t]
\centering
\caption{Per-task ESR for all 60~benchmark tasks. $\Delta = \text{ESR}_A - \text{ESR}_B$; positive values favor {{shortName .ModelA}}.}
\label{tab:pertask-full}
\scriptsize
\begin{tabular}{@{}llrrr|llrrr@{}}
\toprule
Task & Lvl & A & B & $\Delta$ & Task & Lvl & A & B & $\Delta$ \\
\midrule
{{- $tasks := .PerTask}}
{{- $h := half $tasks}}
{{- range $i := seq $h}}
{{- $t := taskAt $tasks $i}}
{{- $j := add $i $h}}
{{- $t2 := taskAt $tasks $j}}
{{esc $t.TaskID}} & {{levelTag $t.Level}} & {{pct $t.ESRA}} & {{pct $t.ESRB}} & {{deltaPct $t.Delta}} & {{esc $t2.TaskID}} & {{levelTag $t2.Level}} & {{pct $t2.ESRA}} & {{pct $t2.ESRB}} & {{deltaPct $t2.Delta}} \\
{{- end}}
\bottomrule
\end{tabular}
\end{table*}
`
