package main

import (
	"encoding/json"
	"flag"
	"log"
	"math"
	"os"
	"sort"
	"time"

	"github.com/mikolajsemeniuk/llmbench"
)

var (
	flagA      string
	flagB      string
	flagOutput string
)

func main() {
	flag.StringVar(&flagA, "a", "", "Path to first benchmark JSON (required)")
	flag.StringVar(&flagB, "b", "", "Path to second benchmark JSON (required)")
	flag.StringVar(&flagOutput, "output", "compare.json", "Output path for comparison JSON")
	flag.Parse()

	if flagA == "" || flagB == "" {
		log.Fatal("both -a and -b are required")
	}

	read := func(path string) llmbench.Report {
		data, err := os.ReadFile(path)
		if err != nil {
			log.Fatalf("cannot read %s: %v", path, err)
		}
		var r llmbench.Report
		if err := json.Unmarshal(data, &r); err != nil {
			log.Fatalf("cannot parse %s: %v", path, err)
		}
		return r
	}

	reportA := read(flagA)
	reportB := read(flagB)

	out := buildCompare(reportA, reportB)
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		log.Fatalf("cannot marshal output: %v", err)
	}
	if err := os.WriteFile(flagOutput, data, 0644); err != nil {
		log.Fatalf("cannot write %s: %v", flagOutput, err)
	}
}

type rawTestResult struct {
	name           string
	fullName       string
	higherIsBetter bool
	valueA, valueB float64
	u, pRaw        float64
	effectSize     float64
	effectLabel    string
}

func buildCompare(a, b llmbench.Report) llmbench.CompareReport {
	// Core vectors.
	successA, successB := binaryVec(a.Records, func(r llmbench.Record) bool { return r.DiagCorrect && r.ActionCorrect }), binaryVec(b.Records, func(r llmbench.Record) bool { return r.DiagCorrect && r.ActionCorrect })
	actionA, actionB := binaryVec(a.Records, func(r llmbench.Record) bool { return r.ActionCorrect }), binaryVec(b.Records, func(r llmbench.Record) bool { return r.ActionCorrect })
	chrA, chrB := chrVector(a.Records), chrVector(b.Records)
	latA, latB := f64Vec(a.Records, func(r llmbench.Record) float64 { return r.LatencySec }), f64Vec(b.Records, func(r llmbench.Record) float64 { return r.LatencySec })

	// Extended vectors (new).
	svrA, svrB := binaryVec(a.Records, func(r llmbench.Record) bool { return r.JSONValid }), binaryVec(b.Records, func(r llmbench.Record) bool { return r.JSONValid })
	scrA, scrB := binaryVec(a.Records, func(r llmbench.Record) bool { return r.SchemaCompliant }), binaryVec(b.Records, func(r llmbench.Record) bool { return r.SchemaCompliant })
	rprA, rprB := f64Vec(a.Records, func(r llmbench.Record) float64 { return r.RPR }), f64Vec(b.Records, func(r llmbench.Record) float64 { return r.RPR })
	cdsA, cdsB := f64Vec(a.Records, func(r llmbench.Record) float64 { return r.CDS }), f64Vec(b.Records, func(r llmbench.Record) float64 { return r.CDS })
	teA, teB := f64Vec(a.Records, func(r llmbench.Record) float64 { return r.TE }), f64Vec(b.Records, func(r llmbench.Record) float64 { return r.TE })

	// m=9 Holm-Bonferroni family.
	rawTests := []rawTestResult{
		buildRaw("ESR", "Execution Success Rate", true, a.Metrics.ESR, b.Metrics.ESR, successA, successB),
		buildRaw("TSA", "Tool Selection Accuracy", true, a.Metrics.TSA, b.Metrics.TSA, actionA, actionB),
		buildRaw("CHR", "Context Hallucination Rate", false, a.Metrics.CHR, b.Metrics.CHR, chrA, chrB),
		buildRaw("LatP50", "Latency p50 (s)", false, a.Metrics.LatencyP50, b.Metrics.LatencyP50, latA, latB),
		buildRaw("SVR", "Syntax Validation Rate", true, a.Metrics.SVR, b.Metrics.SVR, svrA, svrB),
		buildRaw("SCR", "Schema Compliance Rate", true, a.Metrics.SCR, b.Metrics.SCR, scrA, scrB),
		buildRaw("RPR", "Recovery Plan Rationality", true, a.Metrics.RPR, b.Metrics.RPR, rprA, rprB),
		buildRaw("CDS", "Context Density Score", true, a.Metrics.CDS, b.Metrics.CDS, cdsA, cdsB),
		buildRaw("TE", "Token Efficiency", true, a.Metrics.TE, b.Metrics.TE, teA, teB),
	}

	rawPs := make([]float64, len(rawTests))
	for i, rt := range rawTests {
		rawPs[i] = rt.pRaw
	}
	corrPs := llmbench.HolmBonferroniCorrection(rawPs)

	const method = "holm-bonferroni"
	tested := make([]llmbench.MetricComparison, len(rawTests))
	for i, rt := range rawTests {
		tested[i] = llmbench.MetricComparison{
			Name: rt.name, FullName: rt.fullName, HigherIsBetter: rt.higherIsBetter,
			ValueA: rt.valueA, ValueB: rt.valueB, Delta: rt.valueA - rt.valueB,
			WilcoxonU: rt.u, PValue: rt.pRaw, PValueCorrected: corrPs[i],
			CorrectionMethod: method,
			Significance:     llmbench.WilcoxonSignificanceLabel(corrPs[i]),
			EffectSize:       rt.effectSize, EffectLabel: rt.effectLabel,
		}
	}

	scalars := []llmbench.MetricComparison{
		scalarCmp("FCSR", "First Call Success Rate", true, a.Metrics.FCSR, b.Metrics.FCSR),
		scalarCmp("DAAR", "Destructive Action Rate", false, a.Metrics.DAAR, b.Metrics.DAAR),
		scalarCmp("LAE", "Latency-Action Efficiency", true, a.Metrics.LAE, b.Metrics.LAE),
		scalarCmp("MTTR", "Mean Time To Recovery (s)", false, a.Metrics.MTTR, b.Metrics.MTTR),
		scalarCmp("MFS", "Multi-Step Faithfulness", true, a.Metrics.MFS, b.Metrics.MFS),
		scalarCmp("CES", "Cost Efficiency Score", true, a.Metrics.CES, b.Metrics.CES),
		scalarCmp("CTR", "Context Truncation Rate", false, a.Metrics.CTR, b.Metrics.CTR),
		scalarCmp("CCR", "Context Compression Ratio", true, a.Metrics.CCR, b.Metrics.CCR),
		scalarCmp("LatP95", "Latency p95 (s)", false, a.Metrics.LatencyP95, b.Metrics.LatencyP95),
		scalarCmp("LatP99", "Latency p99 (s)", false, a.Metrics.LatencyP99, b.Metrics.LatencyP99),
	}

	cr := llmbench.CompareReport{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		ModelA:      a.Metadata.Provider + "/" + a.Metadata.Model,
		ModelB:      b.Metadata.Provider + "/" + b.Metadata.Model,
		Aggregate:   append(tested, scalars...),
		PerLevel:    buildPerLevel(a, b),
		PerTask:     buildPerTask(a, b),
	}
	cr.Raw.A = a
	cr.Raw.B = b
	return cr
}

func buildRaw(name, fullName string, hib bool, va, vb float64, sA, sB []float64) rawTestResult {
	u, p := llmbench.WilcoxonRankSum(sA, sB)
	r := rankBiserialR(u, len(sA), len(sB))
	return rawTestResult{name, fullName, hib, va, vb, u, p, r, effectLabel(math.Abs(r))}
}

func scalarCmp(name, fullName string, hib bool, va, vb float64) llmbench.MetricComparison {
	return llmbench.MetricComparison{
		Name: name, FullName: fullName, HigherIsBetter: hib,
		ValueA: va, ValueB: vb, Delta: va - vb,
		CorrectionMethod: "n/a", Significance: "n/a", EffectLabel: "n/a",
	}
}

func buildPerLevel(a, b llmbench.Report) []llmbench.LevelComparison {
	idxA := make(map[string]llmbench.LevelMetrics)
	for _, l := range a.PerLevel {
		idxA[l.Name] = l
	}
	idxB := make(map[string]llmbench.LevelMetrics)
	for _, l := range b.PerLevel {
		idxB[l.Name] = l
	}
	order := []string{"L1-diagnostic", "L2-repair", "L3-multi-step"}
	var out []llmbench.LevelComparison
	for _, lv := range order {
		la, oka := idxA[lv]
		lb, okb := idxB[lv]
		if !oka && !okb {
			continue
		}
		out = append(out, llmbench.LevelComparison{
			Level: lv, ESRA: la.ESR, ESRB: lb.ESR, TSAA: la.TSA, TSAB: lb.TSA,
			CHRA: la.CHR, CHRB: lb.CHR, RunsA: la.Runs, RunsB: lb.Runs,
		})
	}
	return out
}

func buildPerTask(a, b llmbench.Report) []llmbench.TaskComparison {
	idxB := make(map[string]llmbench.Summary)
	for _, s := range b.Summaries {
		idxB[s.TaskID] = s
	}
	var out []llmbench.TaskComparison
	for _, sa := range a.Summaries {
		sb := idxB[sa.TaskID]
		out = append(out, llmbench.TaskComparison{
			TaskID: sa.TaskID, Level: sa.Level, ESRA: sa.ESR, ESRB: sb.ESR, Delta: sa.ESR - sb.ESR,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].TaskID < out[j].TaskID })
	return out
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

func binaryVec(recs []llmbench.Record, pred func(llmbench.Record) bool) []float64 {
	v := make([]float64, len(recs))
	for i, r := range recs {
		if pred(r) {
			v[i] = 1.0
		}
	}
	return v
}

func f64Vec(recs []llmbench.Record, ext func(llmbench.Record) float64) []float64 {
	v := make([]float64, len(recs))
	for i, r := range recs {
		v[i] = ext(r)
	}
	return v
}

func chrVector(recs []llmbench.Record) []float64 {
	v := make([]float64, len(recs))
	for i, r := range recs {
		if r.TotalEntities > 0 {
			v[i] = float64(r.Hallucinations) / float64(r.TotalEntities)
		}
	}
	return v
}

func rankBiserialR(u float64, na, nb int) float64 {
	d := float64(na * nb)
	if d == 0 {
		return 0
	}
	return 1.0 - (2*u)/d
}

func effectLabel(absR float64) string {
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
