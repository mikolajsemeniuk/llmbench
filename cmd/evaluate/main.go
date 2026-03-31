package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/llmbench"
)

type Provider interface {
	Name() string
	ChatCompletion(ctx context.Context, prompt string) (llmbench.Response, error)
}

var (
	flagProvider      string
	flagModel         string
	flagURL           string
	flagRuns          int
	flagOutput        string
	flagSeed          int64
	flagAPIKey        string
	flagContextWindow int
)

func newProvider() Provider {
	switch strings.ToLower(flagProvider) {
	case "ollama":
		return llmbench.NewOllamaProvider(flagURL, flagModel, 0, flagSeed)
	// case "anthropic":
	// 	return llmbench.NewAnthropicProvider(flagAPIKey, flagModel, 0, flagSeed)
	default:
		log.Fatalf("unknown provider %q — supported: ollama, anthropic", flagProvider)
		return nil
	}
}

func main() {
	flag.StringVar(&flagProvider, "provider", "ollama", "Model provider: ollama | anthropic")
	flag.StringVar(&flagModel, "model", "qwen2.5:3b-instruct", "Model identifier")
	flag.StringVar(&flagURL, "url", "http://localhost:11434", "Base URL for Ollama server")
	flag.IntVar(&flagRuns, "runs", 10, "Number of independent runs per task")
	flag.StringVar(&flagOutput, "output", "results.json", "Path for the JSON report")
	flag.Int64Var(&flagSeed, "seed", 42, "Random seed for bootstrap CI")
	flag.StringVar(&flagAPIKey, "api-key", "", "API key for API providers")
	flag.IntVar(&flagContextWindow, "context-window", 0, "Model context window in tokens (0 = use provider default)")
	flag.Parse()

	provider := newProvider()
	tasks := llmbench.BenchmarkTasks()
	totalRuns := len(tasks) * flagRuns

	parts := strings.SplitN(provider.Name(), "/", 2)
	providerName, modelName := parts[0], parts[0]
	if len(parts) == 2 {
		modelName = parts[1]
	}
	pricing := llmbench.KnownPricing(providerName, modelName)
	ctxWindow := flagContextWindow
	if ctxWindow == 0 {
		ctxWindow = pricing.ContextWindow
	}

	fmt.Println("=== LLMBench: K8s MCP Benchmark ===")
	fmt.Printf("Provider:    %s\n", provider.Name())
	fmt.Printf("Tasks:       %d (L1=%d, L2=%d, L3=%d)\n",
		len(tasks),
		countLevel(tasks, llmbench.LevelDiagnostic),
		countLevel(tasks, llmbench.LevelRepair),
		countLevel(tasks, llmbench.LevelMultiStep),
	)
	fmt.Printf("Runs/task:   %d\n", flagRuns)
	fmt.Printf("Total runs:  %d\n", totalRuns)
	fmt.Printf("Seed:        %d\n", flagSeed)
	fmt.Println()

	var modelDigest, modelFamily, modelQuant string
	if op, ok := provider.(*llmbench.OllamaProvider); ok {
		if info, err := op.ModelInfo(); err == nil && info.Digest != "" {
			modelDigest = info.Digest
			modelFamily = info.Details.Family
			modelQuant = info.Details.QuantizationLevel
			if len(modelDigest) > 12 {
				fmt.Printf("Digest:      %s\n", modelDigest[:12])
			}
			fmt.Printf("Family:      %s\n", modelFamily)
			fmt.Printf("Quant:       %s\n", modelQuant)
			fmt.Println()
		} else if err != nil {
			log.Printf("Warning: cannot fetch Ollama model info: %v", err)
		}
	}

	var (
		results      []llmbench.Result
		records      []llmbench.Record
		totalCostUSD float64
	)

	ctx := context.Background()

	for _, task := range tasks {
		fmt.Printf("[%s] %s\n", task.ID, task.Description)
		prompt := llmbench.BuildPrompt(task)

		for run := 0; run < flagRuns; run++ {
			start := time.Now()
			resp, err := provider.ChatCompletion(ctx, prompt)
			latency := time.Since(start).Seconds()
			resp.LatencySec = latency

			if err != nil {
				fmt.Printf("  Run %d/%d: ERROR (%v)\n", run+1, flagRuns, err)
				results = append(results, llmbench.Result{
					TaskID: task.ID, RunIndex: run, LatencySec: latency,
					TotalArgs: len(task.GroundTruth.ContextEntities), HallucinatedArgs: len(task.GroundTruth.ContextEntities),
					TotalSteps: len(task.GroundTruth.DiagnosisGroups),
				})
				records = append(records, llmbench.Record{
					TaskID: task.ID, RunIndex: run, LatencySec: latency,
					TotalEntities: len(task.GroundTruth.ContextEntities), Hallucinations: len(task.GroundTruth.ContextEntities),
					Error: err.Error(),
				})
				continue
			}

			eval := llmbench.EvaluateResponseFull(resp.Text, task, resp.PromptTokens, ctxWindow)
			eval.TaskID = task.ID
			eval.RunIndex = run
			eval.LatencySec = latency
			eval.PromptTokens = resp.PromptTokens
			eval.CompletionTokens = resp.CompletionTokens
			results = append(results, eval)

			totalCostUSD += pricing.RunCostUSD(resp.PromptTokens, resp.CompletionTokens)

			tokPerSec := 0.0
			if resp.CompletionTokens > 0 && latency > 0 {
				tokPerSec = float64(resp.CompletionTokens) / latency
			}
			te := llmbench.ComputeTokenEfficiency(resp.Text, resp.CompletionTokens)
			cds := 0.0
			if eval.ContextTotalWords > 0 {
				cds = float64(eval.ContextRelevantWords) / float64(eval.ContextTotalWords)
			}
			mfs := 0.0
			if eval.TotalSteps > 0 {
				mfs = float64(eval.GroundedSteps) / float64(eval.TotalSteps)
			}

			records = append(records, llmbench.Record{
				TaskID: task.ID, RunIndex: run, LatencySec: latency,
				DiagCorrect: eval.DiagnosisCorrect, ActionCorrect: eval.ActionCorrect,
				Hallucinations: eval.HallucinatedArgs, TotalEntities: eval.TotalArgs,
				Destructive:  eval.DestructiveHit,
				PromptTokens: resp.PromptTokens, CompletionTokens: resp.CompletionTokens,
				TokensPerSec: tokPerSec,
				JSONValid:    eval.JSONValid, SchemaCompliant: eval.SchemaCompliant,
				RPR: eval.RPRScore, MFS: mfs, CDS: cds, TE: te,
				Truncated: eval.Truncated,
			})

			status := "FAIL"
			if eval.DiagnosisCorrect && eval.ActionCorrect {
				status = "PASS"
			}
			fmt.Printf("  Run %d/%d: %s (%.1fs, %d tok)\n", run+1, flagRuns, status, latency, resp.CompletionTokens)
		}
	}

	report := buildReport(provider, modelDigest, modelFamily, modelQuant, tasks, results, records, totalCostUSD, ctxWindow)
	printReport(report)
	saveReport(report)
}

func buildReport(
	provider Provider,
	digest, family, quant string,
	tasks []llmbench.Task,
	results []llmbench.Result,
	records []llmbench.Record,
	totalCostUSD float64,
	contextWindow int,
) llmbench.Report {
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

	esr := llmbench.ExecutionSuccessRate(successCount, totalRuns)
	tsa := llmbench.ToolSelectionAccuracy(actionCorrectCount, totalRuns)
	chr := llmbench.ContextHallucinationRate(totalHallucinated, totalEntities)
	daar := llmbench.DestructiveActionAttemptRate(destructiveCount, totalRuns)

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
	fcsr := llmbench.FirstCallSuccessRate(firstCallSuccesses, len(tasks))

	p50 := llmbench.LatencyPercentile(latencies, 50)
	p95 := llmbench.LatencyPercentile(latencies, 95)
	p99 := llmbench.LatencyPercentile(latencies, 99)
	lae := llmbench.LatencyToActionEfficiency(esr, p50)
	mttr := llmbench.MeanTimeToRecovery(successLatencies)

	rng := rand.New(rand.NewSource(flagSeed))
	ci := llmbench.BootstrapConfidenceInterval(successCount, totalRuns, 10000, 0.05, rng.Float64)

	svr := llmbench.SyntaxValidationRate(jsonValidCount, totalRuns)
	scr := llmbench.SchemaComplianceRate(schemaCompliantCount, totalRuns)

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

	cds := llmbench.ContextDensityScore(cdsRelevantSum, cdsTotalSum)
	rpr := 0.0
	if totalRuns > 0 {
		rpr = rprSum / float64(totalRuns)
	}
	mfs := llmbench.MultiStepFaithfulnessScore(groundedStepsSum, totalStepsSum)
	ces := llmbench.CostEfficiencyScore(successCount, totalCostUSD)
	ctr := llmbench.ContextTruncationRate(truncatedCount, totalRuns)
	ccr := llmbench.MeasureCCR(llmbench.ManifestTokenCount())

	parts := strings.SplitN(provider.Name(), "/", 2)
	provName, modName := parts[0], parts[0]
	if len(parts) == 2 {
		modName = parts[1]
	}

	return llmbench.Report{
		Metadata: llmbench.Metadata{
			Provider: provName, Model: modName,
			ModelDigest: digest, ModelFamily: family, ModelQuant: quant,
			Timestamp:  time.Now().UTC().Format(time.RFC3339),
			TotalTasks: len(tasks), RunsPerTask: flagRuns, TotalRuns: totalRuns,
			Seed: flagSeed, ContextWindow: contextWindow,
		},
		Metrics: llmbench.Metrics{
			ESR: esr, ESRCI: ci, TSA: tsa, CHR: chr, DAAR: daar,
			FCSR: fcsr, LAE: lae, MTTR: mttr,
			LatencyP50: p50, LatencyP95: p95, LatencyP99: p99,
			SVR: svr, SCR: scr, TE: teAgg, CDS: cds,
			RPR: rpr, MFS: mfs, CES: ces, CTR: ctr, CCR: ccr,
		},
		PerLevel:  computePerLevel(tasks, results),
		RAG:       computeRAGMetrics(tasks),
		Summaries: computePerTask(tasks, results),
		Records:   records,
	}
}

func computePerLevel(tasks []llmbench.Task, results []llmbench.Result) []llmbench.LevelMetrics {
	taskLevel := make(map[string]llmbench.TaskLevel, len(tasks))
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
	order := []string{string(llmbench.LevelDiagnostic), string(llmbench.LevelRepair), string(llmbench.LevelMultiStep)}
	var out []llmbench.LevelMetrics
	for _, lv := range order {
		v, ok := m[lv]
		if !ok {
			continue
		}
		out = append(out, llmbench.LevelMetrics{
			Name: lv,
			ESR:  llmbench.ExecutionSuccessRate(v.success, v.total),
			TSA:  llmbench.ToolSelectionAccuracy(v.action, v.total),
			CHR:  llmbench.ContextHallucinationRate(v.hall, v.ent),
			Runs: v.total,
		})
	}
	return out
}

func computeRAGMetrics(tasks []llmbench.Task) llmbench.RAGQualityMetrics {
	var sumP, sumR, sumMRR, sumNDCG float64
	for _, t := range tasks {
		p, r, m, n := llmbench.ComputeTaskRAGMetrics(t)
		sumP += p
		sumR += r
		sumMRR += m
		sumNDCG += n
	}
	nt := float64(len(tasks))
	mp, mr := sumP/nt, sumR/nt
	return llmbench.RAGQualityMetrics{
		MeanPrecisionAtK: mp, MeanRecallAtK: mr,
		MeanMRR: sumMRR / nt, MeanNDCGAtK: sumNDCG / nt,
		MeanFScoreAtK: llmbench.RAGFScoreAtK(mp, mr, 1.0),
	}
}

func computePerTask(tasks []llmbench.Task, results []llmbench.Result) []llmbench.Summary {
	out := make([]llmbench.Summary, 0, len(tasks))
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
		out = append(out, llmbench.Summary{
			TaskID: t.ID, Level: string(t.Level),
			ESR:        llmbench.ExecutionSuccessRate(success, total),
			TSA:        llmbench.ToolSelectionAccuracy(action, total),
			CHR:        llmbench.ContextHallucinationRate(hall, ent),
			MeanLatSec: ml,
		})
	}
	return out
}

func printReport(r llmbench.Report) {
	sep := strings.Repeat("=", 60)
	fmt.Printf("\n%s\nBENCHMARK REPORT\n%s\n", sep, sep)
	fmt.Printf("Provider:    %s\nModel:       %s\n", r.Metadata.Provider, r.Metadata.Model)
	fmt.Printf("Tasks:       %d\nRuns/task:   %d\nTotal runs:  %d\n", r.Metadata.TotalTasks, r.Metadata.RunsPerTask, r.Metadata.TotalRuns)

	a := r.Metrics
	fmt.Println("\n--- CORE ---")
	fmt.Printf("ESR=%.3f [CI %.3f,%.3f]  TSA=%.3f  CHR=%.3f  DAAR=%.3f  FCSR=%.3f  LAE=%.4f\n",
		a.ESR, a.ESRCI[0], a.ESRCI[1], a.TSA, a.CHR, a.DAAR, a.FCSR, a.LAE)
	fmt.Println("\n--- EXTENDED ---")
	fmt.Printf("SVR=%.3f  SCR=%.3f  TE=%.3f  CDS=%.3f  RPR=%.3f  MFS=%.3f\n", a.SVR, a.SCR, a.TE, a.CDS, a.RPR, a.MFS)
	fmt.Printf("CES=%.2f  CTR=%.3f  CCR=%.3f\n", a.CES, a.CTR, a.CCR)
	fmt.Println("\n--- LATENCY ---")
	fmt.Printf("p50=%.2fs  p95=%.2fs  p99=%.2fs  MTTR=%.2fs\n", a.LatencyP50, a.LatencyP95, a.LatencyP99, a.MTTR)

	fmt.Println("\n--- PER-LEVEL ---")
	for _, m := range r.PerLevel {
		fmt.Printf("  %-18s ESR=%.3f TSA=%.3f CHR=%.3f (n=%d)\n", m.Name, m.ESR, m.TSA, m.CHR, m.Runs)
	}
	fmt.Println("\n--- RAG QUALITY ---")
	fmt.Printf("  P@K=%.3f R@K=%.3f F1@K=%.3f MRR=%.3f NDCG@K=%.3f\n",
		r.RAG.MeanPrecisionAtK, r.RAG.MeanRecallAtK, r.RAG.MeanFScoreAtK, r.RAG.MeanMRR, r.RAG.MeanNDCGAtK)
	fmt.Printf("\nSaved: %s\n", flagOutput)
}

func saveReport(r llmbench.Report) {
	sanitizeMetrics(&r.Metrics)
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		log.Printf("marshal error: %v", err)
		return
	}
	if err := os.WriteFile(flagOutput, data, 0644); err != nil {
		log.Printf("write error: %v", err)
	}
}

func sanitizeMetrics(m *llmbench.Metrics) {
	sanitize := func(v *float64) {
		if math.IsInf(*v, 0) || math.IsNaN(*v) {
			*v = -1 // sentinel: -1 means "not applicable" (local provider, zero cost)
		}
	}
	sanitize(&m.CES)
	sanitize(&m.LAE)
	sanitize(&m.MTTR)
}

func countLevel(tasks []llmbench.Task, level llmbench.TaskLevel) int {
	n := 0
	for _, t := range tasks {
		if t.Level == level {
			n++
		}
	}

	return n
}
