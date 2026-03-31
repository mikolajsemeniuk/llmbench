package llmbench

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

type compareRawTest struct {
	name           string
	fullName       string
	higherIsBetter bool
	valueA, valueB float64
	u, pRaw        float64
	effectSize     float64
	effectLabel    string
}

// BuildCompareReport runs Wilcoxon + Holm–Bonferroni on paired run vectors from two benchmark reports.
func BuildCompareReport(a, b Report) CompareReport {
	successA, successB := binaryVec(a.Records, func(r Record) bool { return r.DiagCorrect && r.ActionCorrect }), binaryVec(b.Records, func(r Record) bool { return r.DiagCorrect && r.ActionCorrect })
	actionA, actionB := binaryVec(a.Records, func(r Record) bool { return r.ActionCorrect }), binaryVec(b.Records, func(r Record) bool { return r.ActionCorrect })
	chrA, chrB := chrVector(a.Records), chrVector(b.Records)
	latA, latB := f64Vec(a.Records, func(r Record) float64 { return r.LatencySec }), f64Vec(b.Records, func(r Record) float64 { return r.LatencySec })

	svrA, svrB := binaryVec(a.Records, func(r Record) bool { return r.JSONValid }), binaryVec(b.Records, func(r Record) bool { return r.JSONValid })
	scrA, scrB := binaryVec(a.Records, func(r Record) bool { return r.SchemaCompliant }), binaryVec(b.Records, func(r Record) bool { return r.SchemaCompliant })
	rprA, rprB := f64Vec(a.Records, func(r Record) float64 { return r.RPR }), f64Vec(b.Records, func(r Record) float64 { return r.RPR })
	cdsA, cdsB := f64Vec(a.Records, func(r Record) float64 { return r.CDS }), f64Vec(b.Records, func(r Record) float64 { return r.CDS })
	teA, teB := f64Vec(a.Records, func(r Record) float64 { return r.TE }), f64Vec(b.Records, func(r Record) float64 { return r.TE })

	rawTests := []compareRawTest{
		compareBuildRaw("ESR", "Execution Success Rate", true, a.Metrics.ESR, b.Metrics.ESR, successA, successB),
		compareBuildRaw("TSA", "Tool Selection Accuracy", true, a.Metrics.TSA, b.Metrics.TSA, actionA, actionB),
		compareBuildRaw("CHR", "Context Hallucination Rate", false, a.Metrics.CHR, b.Metrics.CHR, chrA, chrB),
		compareBuildRaw("LatP50", "Latency p50 (s)", false, a.Metrics.LatencyP50, b.Metrics.LatencyP50, latA, latB),
		compareBuildRaw("SVR", "Syntax Validation Rate", true, a.Metrics.SVR, b.Metrics.SVR, svrA, svrB),
		compareBuildRaw("SCR", "Schema Compliance Rate", true, a.Metrics.SCR, b.Metrics.SCR, scrA, scrB),
		compareBuildRaw("RPR", "Recovery Plan Rationality", true, a.Metrics.RPR, b.Metrics.RPR, rprA, rprB),
		compareBuildRaw("CDS", "Context Density Score", true, a.Metrics.CDS, b.Metrics.CDS, cdsA, cdsB),
		compareBuildRaw("TE", "Token Efficiency", true, a.Metrics.TE, b.Metrics.TE, teA, teB),
	}

	rawPs := make([]float64, len(rawTests))
	for i, rt := range rawTests {
		rawPs[i] = rt.pRaw
	}
	corrPs := HolmBonferroniCorrection(rawPs)

	const method = "holm-bonferroni"
	tested := make([]MetricComparison, len(rawTests))
	for i, rt := range rawTests {
		tested[i] = MetricComparison{
			Name: rt.name, FullName: rt.fullName, HigherIsBetter: rt.higherIsBetter,
			ValueA: rt.valueA, ValueB: rt.valueB, Delta: rt.valueA - rt.valueB,
			WilcoxonU: rt.u, PValue: rt.pRaw, PValueCorrected: corrPs[i],
			CorrectionMethod: method,
			Significance:     WilcoxonSignificanceLabel(corrPs[i]),
			EffectSize:       rt.effectSize, EffectLabel: rt.effectLabel,
		}
	}

	scalars := []MetricComparison{
		compareScalar("FCSR", "First Call Success Rate", true, a.Metrics.FCSR, b.Metrics.FCSR),
		compareScalar("DAAR", "Destructive Action Rate", false, a.Metrics.DAAR, b.Metrics.DAAR),
		compareScalar("LAE", "Latency-Action Efficiency", true, a.Metrics.LAE, b.Metrics.LAE),
		compareScalar("MTTR", "Mean Time To Recovery (s)", false, a.Metrics.MTTR, b.Metrics.MTTR),
		compareScalar("MFS", "Multi-Step Faithfulness", true, a.Metrics.MFS, b.Metrics.MFS),
		compareScalar("CES", "Cost Efficiency Score", true, a.Metrics.CES, b.Metrics.CES),
		compareScalar("CTR", "Context Truncation Rate", false, a.Metrics.CTR, b.Metrics.CTR),
		compareScalar("CCR", "Context Compression Ratio", true, a.Metrics.CCR, b.Metrics.CCR),
		compareScalar("LatP95", "Latency p95 (s)", false, a.Metrics.LatencyP95, b.Metrics.LatencyP95),
		compareScalar("LatP99", "Latency p99 (s)", false, a.Metrics.LatencyP99, b.Metrics.LatencyP99),
	}

	cr := CompareReport{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		ModelA:      a.Metadata.Provider + "/" + a.Metadata.Model,
		ModelB:      b.Metadata.Provider + "/" + b.Metadata.Model,
		Aggregate:   append(tested, scalars...),
		PerLevel:    compareBuildPerLevel(a, b),
		PerTask:     compareBuildPerTask(a, b),
	}
	cr.Raw.A = a
	cr.Raw.B = b
	return cr
}

func compareBuildRaw(name, fullName string, hib bool, va, vb float64, sA, sB []float64) compareRawTest {
	u, p := WilcoxonRankSum(sA, sB)
	r := compareRankBiserialR(u, len(sA), len(sB))
	return compareRawTest{name, fullName, hib, va, vb, u, p, r, compareRankEffectLabel(math.Abs(r))}
}

func compareScalar(name, fullName string, hib bool, va, vb float64) MetricComparison {
	return MetricComparison{
		Name: name, FullName: fullName, HigherIsBetter: hib,
		ValueA: va, ValueB: vb, Delta: va - vb,
		CorrectionMethod: "n/a", Significance: "n/a", EffectLabel: "n/a",
	}
}

func compareBuildPerLevel(a, b Report) []LevelComparison {
	idxA := make(map[string]LevelMetrics)
	for _, l := range a.PerLevel {
		idxA[l.Name] = l
	}

	idxB := make(map[string]LevelMetrics)
	for _, l := range b.PerLevel {
		idxB[l.Name] = l
	}

	order := []string{"L1-diagnostic", "L2-repair", "L3-multi-step"}
	var out []LevelComparison
	for _, lv := range order {
		la, oka := idxA[lv]
		lb, okb := idxB[lv]
		if !oka && !okb {
			continue
		}
		out = append(out, LevelComparison{
			Level: lv, ESRA: la.ESR, ESRB: lb.ESR, TSAA: la.TSA, TSAB: lb.TSA,
			CHRA: la.CHR, CHRB: lb.CHR, RunsA: la.Runs, RunsB: lb.Runs,
		})
	}

	return out
}

func compareBuildPerTask(a, b Report) []TaskComparison {
	idxB := make(map[string]Summary)
	for _, s := range b.Summaries {
		idxB[s.TaskID] = s
	}

	var out []TaskComparison
	for _, sa := range a.Summaries {
		sb := idxB[sa.TaskID]
		out = append(out, TaskComparison{
			TaskID: sa.TaskID, Level: sa.Level, ESRA: sa.ESR, ESRB: sb.ESR, Delta: sa.ESR - sb.ESR,
		})
	}

	sort.Slice(out, func(i, j int) bool { return out[i].TaskID < out[j].TaskID })
	return out
}

func binaryVec(recs []Record, pred func(Record) bool) []float64 {
	v := make([]float64, len(recs))
	for i, r := range recs {
		if pred(r) {
			v[i] = 1.0
		}
	}

	return v
}

func f64Vec(recs []Record, ext func(Record) float64) []float64 {
	v := make([]float64, len(recs))
	for i, r := range recs {
		v[i] = ext(r)
	}

	return v
}

func chrVector(recs []Record) []float64 {
	v := make([]float64, len(recs))
	for i, r := range recs {
		if r.TotalEntities > 0 {
			v[i] = float64(r.Hallucinations) / float64(r.TotalEntities)
		}
	}

	return v
}

func compareRankBiserialR(u float64, na, nb int) float64 {
	d := float64(na * nb)
	if d == 0 {
		return 0
	}

	return 1.0 - (2*u)/d
}

func compareRankEffectLabel(absR float64) string {
	switch {
	case absR >= 0.50:
		return "large"
	case absR >= 0.30:
		return "medium"
	case absR >= 0.10:
		return "small"
	default:
		return "negligible"
	}
}

// BuildEvaluationReport aggregates raw run results into a full Report (JSON-serializable).
func BuildEvaluationReport(
	seed int64,
	runsPerTask int,
	providerName, modelName string,
	modelDigest, modelFamily, modelQuant string,
	tasks []Task,
	results []Result,
	records []Record,
	totalCostUSD float64,
	contextWindow int,
) Report {
	totalRuns := len(results)

	var (
		successCount, actionCorrectCount     int
		totalHallucinated, totalEntities     int
		destructiveCount                     int
		latencies, successLatencies          []float64
		jsonValidCount, schemaCompliantCount int
		rprSum                               float64
		groundedStepsSum, totalStepsSum      int
		cdsRelevantSum, cdsTotalSum          int
		truncatedCount                       int
	)

	for _, r := range results {
		success := r.DiagnosisCorrect && r.ActionCorrect
		if success {
			successCount++
			successLatencies = append(successLatencies, r.LatencySec)
		}
		if r.ActionCorrect {
			actionCorrectCount++
		}
		totalHallucinated += r.HallucinatedArgs
		totalEntities += r.TotalArgs
		if r.DestructiveHit {
			destructiveCount++
		}
		latencies = append(latencies, r.LatencySec)
		if r.JSONValid {
			jsonValidCount++
		}
		if r.SchemaCompliant {
			schemaCompliantCount++
		}
		rprSum += r.RPRScore
		groundedStepsSum += r.GroundedSteps
		totalStepsSum += r.TotalSteps
		cdsRelevantSum += r.ContextRelevantWords
		cdsTotalSum += r.ContextTotalWords
		if r.Truncated {
			truncatedCount++
		}
	}

	esr := ExecutionSuccessRate(successCount, totalRuns)
	tsa := ToolSelectionAccuracy(actionCorrectCount, totalRuns)
	chr := ContextHallucinationRate(totalHallucinated, totalEntities)
	daar := DestructiveActionAttemptRate(destructiveCount, totalRuns)

	firstCallSuccesses := 0
	for _, task := range tasks {
		for _, r := range results {
			if r.TaskID == task.ID && r.RunIndex == 0 {
				if r.DiagnosisCorrect && r.ActionCorrect {
					firstCallSuccesses++
				}
				break
			}
		}
	}
	fcsr := FirstCallSuccessRate(firstCallSuccesses, len(tasks))

	p50 := LatencyPercentile(latencies, 50)
	p95 := LatencyPercentile(latencies, 95)
	p99 := LatencyPercentile(latencies, 99)
	lae := LatencyToActionEfficiency(esr, p50)
	mttr := MeanTimeToRecovery(successLatencies)

	rng := rand.New(rand.NewSource(seed))
	ci := BootstrapConfidenceInterval(successCount, totalRuns, 10000, 0.05, rng.Float64)

	svr := SyntaxValidationRate(jsonValidCount, totalRuns)
	scr := SchemaComplianceRate(schemaCompliantCount, totalRuns)

	teAgg := 0.0
	teN := 0
	for _, rec := range records {
		if rec.TE > 0 {
			teAgg += rec.TE
			teN++
		}
	}
	if teN > 0 {
		teAgg /= float64(teN)
	}

	cds := ContextDensityScore(cdsRelevantSum, cdsTotalSum)
	rpr := 0.0
	if totalRuns > 0 {
		rpr = rprSum / float64(totalRuns)
	}
	mfs := MultiStepFaithfulnessScore(groundedStepsSum, totalStepsSum)
	ces := CostEfficiencyScore(successCount, totalCostUSD)
	ctr := ContextTruncationRate(truncatedCount, totalRuns)
	ccr := MeasureCCR(ManifestTokenCount())

	return Report{
		Metadata: Metadata{
			Provider: providerName, Model: modelName,
			ModelDigest: modelDigest, ModelFamily: modelFamily, ModelQuant: modelQuant,
			Timestamp:  time.Now().UTC().Format(time.RFC3339),
			TotalTasks: len(tasks), RunsPerTask: runsPerTask, TotalRuns: totalRuns,
			Seed: seed, ContextWindow: contextWindow,
		},
		Metrics: Metrics{
			ESR: esr, ESRCI: ci, TSA: tsa, CHR: chr, DAAR: daar,
			FCSR: fcsr, LAE: lae, MTTR: mttr,
			LatencyP50: p50, LatencyP95: p95, LatencyP99: p99,
			SVR: svr, SCR: scr, TE: teAgg, CDS: cds,
			RPR: rpr, MFS: mfs, CES: ces, CTR: ctr, CCR: ccr,
		},
		PerLevel:  EvaluatePerLevelMetrics(tasks, results),
		RAG:       AggregateRAGQualityMetrics(tasks),
		Summaries: SummarizePerTaskResults(tasks, results),
		Records:   records,
	}
}

// EvaluatePerLevelMetrics aggregates ESR/TSA/CHR by task level.
func EvaluatePerLevelMetrics(tasks []Task, results []Result) []LevelMetrics {
	taskLevel := make(map[string]TaskLevel, len(tasks))
	for _, t := range tasks {
		taskLevel[t.ID] = t.Level
	}
	type acc struct{ success, action, hall, ent, total int }
	m := make(map[string]*acc)
	for _, r := range results {
		lv := string(taskLevel[r.TaskID])
		a, ok := m[lv]
		if !ok {
			a = &acc{}
			m[lv] = a
		}
		a.total++
		if r.DiagnosisCorrect && r.ActionCorrect {
			a.success++
		}
		if r.ActionCorrect {
			a.action++
		}
		a.hall += r.HallucinatedArgs
		a.ent += r.TotalArgs
	}
	order := []string{string(LevelDiagnostic), string(LevelRepair), string(LevelMultiStep)}
	var out []LevelMetrics
	for _, lv := range order {
		v, ok := m[lv]
		if !ok {
			continue
		}
		out = append(out, LevelMetrics{
			Name: lv,
			ESR:  ExecutionSuccessRate(v.success, v.total),
			TSA:  ToolSelectionAccuracy(v.action, v.total),
			CHR:  ContextHallucinationRate(v.hall, v.ent),
			Runs: v.total,
		})
	}
	return out
}

// AggregateRAGQualityMetrics averages per-task RAG IR metrics.
func AggregateRAGQualityMetrics(tasks []Task) RAGQualityMetrics {
	var sumP, sumR, sumMRR, sumNDCG float64
	for _, t := range tasks {
		p, r, mrr, n := ComputeTaskRAGMetrics(t)
		sumP += p
		sumR += r
		sumMRR += mrr
		sumNDCG += n
	}
	nt := float64(len(tasks))
	mp, mr := sumP/nt, sumR/nt
	return RAGQualityMetrics{
		MeanPrecisionAtK: mp, MeanRecallAtK: mr,
		MeanMRR: sumMRR / nt, MeanNDCGAtK: sumNDCG / nt,
		MeanFScoreAtK: RAGFScoreAtK(mp, mr, 1.0),
	}
}

// SummarizePerTaskResults builds per-task summaries from flat results.
func SummarizePerTaskResults(tasks []Task, results []Result) []Summary {
	out := make([]Summary, 0, len(tasks))
	for _, t := range tasks {
		var success, action, hall, ent, total int
		var lat float64
		for _, r := range results {
			if r.TaskID != t.ID {
				continue
			}
			total++
			if r.DiagnosisCorrect && r.ActionCorrect {
				success++
			}
			if r.ActionCorrect {
				action++
			}
			hall += r.HallucinatedArgs
			ent += r.TotalArgs
			lat += r.LatencySec
		}
		ml := 0.0
		if total > 0 {
			ml = lat / float64(total)
		}
		out = append(out, Summary{
			TaskID: t.ID, Level: string(t.Level),
			ESR:        ExecutionSuccessRate(success, total),
			TSA:        ToolSelectionAccuracy(action, total),
			CHR:        ContextHallucinationRate(hall, ent),
			MeanLatSec: ml,
		})
	}
	return out
}

// PrintReportSummary writes a human-readable summary to w (e.g. stdout).
func PrintReportSummary(w io.Writer, r Report, outputPath string) {
	sep := strings.Repeat("=", 60)
	fmt.Fprintf(w, "\n%s\nBENCHMARK REPORT\n%s\n", sep, sep)
	fmt.Fprintf(w, "Provider:    %s\nModel:       %s\n", r.Metadata.Provider, r.Metadata.Model)
	fmt.Fprintf(w, "Tasks:       %d\nRuns/task:   %d\nTotal runs:  %d\n", r.Metadata.TotalTasks, r.Metadata.RunsPerTask, r.Metadata.TotalRuns)

	a := r.Metrics
	fmt.Fprintln(w, "\n--- CORE ---")
	fmt.Fprintf(w, "ESR=%.3f [CI %.3f,%.3f]  TSA=%.3f  CHR=%.3f  DAAR=%.3f  FCSR=%.3f  LAE=%.4f\n",
		a.ESR, a.ESRCI[0], a.ESRCI[1], a.TSA, a.CHR, a.DAAR, a.FCSR, a.LAE)
	fmt.Fprintln(w, "\n--- EXTENDED ---")
	fmt.Fprintf(w, "SVR=%.3f  SCR=%.3f  TE=%.3f  CDS=%.3f  RPR=%.3f  MFS=%.3f\n", a.SVR, a.SCR, a.TE, a.CDS, a.RPR, a.MFS)
	fmt.Fprintf(w, "CES=%.2f  CTR=%.3f  CCR=%.3f\n", a.CES, a.CTR, a.CCR)
	fmt.Fprintln(w, "\n--- LATENCY ---")
	fmt.Fprintf(w, "p50=%.2fs  p95=%.2fs  p99=%.2fs  MTTR=%.2fs\n", a.LatencyP50, a.LatencyP95, a.LatencyP99, a.MTTR)

	fmt.Fprintln(w, "\n--- PER-LEVEL ---")
	for _, m := range r.PerLevel {
		fmt.Fprintf(w, "  %-18s ESR=%.3f TSA=%.3f CHR=%.3f (n=%d)\n", m.Name, m.ESR, m.TSA, m.CHR, m.Runs)
	}
	fmt.Fprintln(w, "\n--- RAG QUALITY ---")
	fmt.Fprintf(w, "  P@K=%.3f R@K=%.3f F1@K=%.3f MRR=%.3f NDCG@K=%.3f\n",
		r.RAG.MeanPrecisionAtK, r.RAG.MeanRecallAtK, r.RAG.MeanFScoreAtK, r.RAG.MeanMRR, r.RAG.MeanNDCGAtK)
	fmt.Fprintf(w, "\nSaved: %s\n", outputPath)
}

// SanitizeMetricsForJSON replaces Inf/NaN with -1 for JSON-safe export (local/zero-cost).
func SanitizeMetricsForJSON(m *Metrics) {
	sanitize := func(v *float64) {
		if math.IsInf(*v, 0) || math.IsNaN(*v) {
			*v = -1
		}
	}
	sanitize(&m.CES)
	sanitize(&m.LAE)
	sanitize(&m.MTTR)
}

// WriteReportJSON writes report to path with sanitized metrics.
func WriteReportJSON(path string, r Report) error {
	SanitizeMetricsForJSON(&r.Metrics)
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// CountTasksByLevel returns how many tasks have the given level.
func CountTasksByLevel(tasks []Task, level TaskLevel) int {
	n := 0
	for _, t := range tasks {
		if t.Level == level {
			n++
		}
	}
	return n
}

// ReportView holds either a single benchmark Report or a CompareReport for HTML/CLI consumers.
type ReportView struct {
	IsCompare bool
	Single    Report
	Compare   CompareReport
}

// ParseReportFileJSON unmarshals JSON, detecting compare format by the presence of "model_a".
func ParseReportFileJSON(raw []byte) (ReportView, error) {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(raw, &probe); err != nil {
		return ReportView{}, fmt.Errorf("invalid JSON: %w", err)
	}
	if _, ok := probe["model_a"]; ok {
		var cr CompareReport
		if err := json.Unmarshal(raw, &cr); err != nil {
			return ReportView{}, fmt.Errorf("compare report: %w", err)
		}
		return ReportView{IsCompare: true, Compare: cr}, nil
	}
	var sr Report
	if err := json.Unmarshal(raw, &sr); err != nil {
		return ReportView{}, fmt.Errorf("single report: %w", err)
	}
	return ReportView{IsCompare: false, Single: sr}, nil
}

// FormatHTMLDelta formats a delta with sign; negative uses Unicode minus (U+2212) for clean HTML.
func FormatHTMLDelta(v float64) string {
	abs := math.Abs(v)
	if v > 0 {
		return fmt.Sprintf("+%.4f", abs)
	}
	if v < 0 {
		return fmt.Sprintf("−%.4f", abs)
	}
	return "0.0000"
}
