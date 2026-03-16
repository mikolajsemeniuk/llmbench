package main

// LLMBench: K8s MCP Benchmark Runner
//
// Runs the full benchmark suite against a local Ollama model, evaluates responses
// using deterministic keyword-based ground truth, and reports metrics required
// by ACM TOIS, IP&M, IEEE Access, and ESWA reviewers.
//
// Usage:
//   ollama pull qwen2.5:3b-instruct
//   go run ./cmd/qwen_esr/ -model qwen2.5:3b-instruct -runs 5 -output results.json

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/llmbench"
)

type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type OllamaResponse struct {
	Response        string `json:"response"`
	Done            bool   `json:"done"`
	TotalDuration   int64  `json:"total_duration"`
	PromptEvalCount int    `json:"prompt_eval_count"`
	EvalCount       int    `json:"eval_count"`
	EvalDuration    int64  `json:"eval_duration"`
}

// ---------------------------------------------------------------------------
// Report types — structured output for reproducibility
// ---------------------------------------------------------------------------

type BenchmarkReport struct {
	Metadata  ReportMetadata          `json:"metadata"`
	Aggregate AggregateMetrics        `json:"aggregate"`
	PerLevel  map[string]LevelMetrics `json:"per_level"`
	RAG       RAGQualityMetrics       `json:"rag_quality"`
	PerTask   []TaskSummary           `json:"per_task"`
	Runs      []RunRecord             `json:"runs"`
}

type ReportMetadata struct {
	Model       string `json:"model"`
	Timestamp   string `json:"timestamp"`
	TotalTasks  int    `json:"total_tasks"`
	RunsPerTask int    `json:"runs_per_task"`
	TotalRuns   int    `json:"total_runs"`
	Seed        int64  `json:"random_seed"`
}

type AggregateMetrics struct {
	ESR        float64    `json:"esr"`
	ESRCI      [2]float64 `json:"esr_ci_95"`
	TSA        float64    `json:"tsa"`
	CHR        float64    `json:"chr"`
	DAAR       float64    `json:"daar"`
	FCSR       float64    `json:"fcsr"`
	LAE        float64    `json:"lae"`
	MTTR       float64    `json:"mttr_sec"`
	LatencyP50 float64    `json:"latency_p50_sec"`
	LatencyP95 float64    `json:"latency_p95_sec"`
	LatencyP99 float64    `json:"latency_p99_sec"`
}

type LevelMetrics struct {
	ESR  float64 `json:"esr"`
	TSA  float64 `json:"tsa"`
	CHR  float64 `json:"chr"`
	Runs int     `json:"runs"`
}

type RAGQualityMetrics struct {
	MeanPrecisionAtK float64 `json:"mean_precision_at_k"`
	MeanRecallAtK    float64 `json:"mean_recall_at_k"`
	MeanMRR          float64 `json:"mean_mrr"`
	MeanNDCGAtK      float64 `json:"mean_ndcg_at_k"`
	MeanFScoreAtK    float64 `json:"mean_f1_at_k"`
}

type TaskSummary struct {
	TaskID     string  `json:"task_id"`
	Level      string  `json:"level"`
	ESR        float64 `json:"esr"`
	TSA        float64 `json:"tsa"`
	CHR        float64 `json:"chr"`
	MeanLatSec float64 `json:"mean_latency_sec"`
}

type RunRecord struct {
	TaskID           string  `json:"task_id"`
	RunIndex         int     `json:"run_index"`
	LatencySec       float64 `json:"latency_sec"`
	DiagCorrect      bool    `json:"diagnosis_correct"`
	ActionCorrect    bool    `json:"action_correct"`
	Hallucinations   int     `json:"hallucinations"`
	TotalEntities    int     `json:"total_entities"`
	Destructive      bool    `json:"destructive_action"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TokensPerSec     float64 `json:"tokens_per_sec"`
	Error            string  `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// Main benchmark loop
// ---------------------------------------------------------------------------

var (
	flagModel  string
	flagURL    string
	flagRuns   int
	flagOutput string
	flagSeed   int64
)

func main() {
	flag.StringVar(&flagModel, "model", "qwen2.5:3b-instruct", "Ollama model name")
	flag.StringVar(&flagURL, "url", "http://localhost:11434/api/generate", "Ollama API URL")
	flag.IntVar(&flagRuns, "runs", 5, "Number of independent runs per task")
	flag.StringVar(&flagOutput, "output", "results.json", "Output JSON file path")
	flag.Int64Var(&flagSeed, "seed", 42, "Random seed for bootstrap CI (reproducibility)")
	flag.Parse()

	tasks := llmbench.BenchmarkTasks()
	totalRuns := len(tasks) * flagRuns

	fmt.Println("=== LLMBench: K8s MCP Benchmark ===")
	fmt.Printf("Model:       %s\n", flagModel)
	fmt.Printf("Tasks:       %d (L1=%d, L2=%d, L3=%d)\n",
		len(tasks), countLevel(tasks, llmbench.LevelDiagnostic),
		countLevel(tasks, llmbench.LevelRepair), countLevel(tasks, llmbench.LevelMultiStep))
	fmt.Printf("Runs/task:   %d\n", flagRuns)
	fmt.Printf("Total runs:  %d\n", totalRuns)
	fmt.Printf("Seed:        %d\n", flagSeed)
	fmt.Println()

	// Pre-flight: verify Ollama connectivity
	fmt.Print("Checking Ollama connectivity... ")
	if _, err := callOllama(flagURL, flagModel, "ping"); err != nil {
		log.Fatalf("FAILED\n\nCannot reach Ollama at %s: %v\n\nEnsure Ollama is running:\n  ollama serve\n  ollama pull %s\n",
			flagURL, err, flagModel)
	}
	fmt.Println("OK")
	fmt.Println()

	var (
		results []llmbench.TaskResult
		runs    []RunRecord
	)

	for _, task := range tasks {
		fmt.Printf("[%s] %s\n", task.ID, task.Description)
		prompt := llmbench.BuildPrompt(task)

		for run := 0; run < flagRuns; run++ {
			start := time.Now()
			ollamaResp, err := callOllama(flagURL, flagModel, prompt)
			latency := time.Since(start).Seconds()

			if err != nil {
				fmt.Printf("  Run %d/%d: ERROR (%v)\n", run+1, flagRuns, err)
				results = append(results, llmbench.TaskResult{
					TaskID:           task.ID,
					RunIndex:         run,
					LatencySec:       latency,
					TotalArgs:        len(task.GroundTruth.ContextEntities),
					HallucinatedArgs: len(task.GroundTruth.ContextEntities),
				})
				runs = append(runs, RunRecord{
					TaskID:         task.ID,
					RunIndex:       run,
					LatencySec:     latency,
					TotalEntities:  len(task.GroundTruth.ContextEntities),
					Hallucinations: len(task.GroundTruth.ContextEntities),
					Error:          err.Error(),
				})
				continue
			}

			eval := llmbench.EvaluateResponse(ollamaResp.Response, task.GroundTruth)
			eval.TaskID = task.ID
			eval.RunIndex = run
			eval.LatencySec = latency
			eval.PromptTokens = ollamaResp.PromptEvalCount
			eval.CompletionTokens = ollamaResp.EvalCount
			results = append(results, eval)

			tokPerSec := 0.0
			if ollamaResp.EvalDuration > 0 {
				tokPerSec = float64(ollamaResp.EvalCount) / (float64(ollamaResp.EvalDuration) / 1e9)
			}
			runs = append(runs, RunRecord{
				TaskID:           task.ID,
				RunIndex:         run,
				LatencySec:       latency,
				DiagCorrect:      eval.DiagnosisCorrect,
				ActionCorrect:    eval.ActionCorrect,
				Hallucinations:   eval.HallucinatedArgs,
				TotalEntities:    eval.TotalArgs,
				Destructive:      eval.DestructiveHit,
				PromptTokens:     ollamaResp.PromptEvalCount,
				CompletionTokens: ollamaResp.EvalCount,
				TokensPerSec:     tokPerSec,
			})

			status := "FAIL"
			if eval.DiagnosisCorrect && eval.ActionCorrect {
				status = "PASS"
			}
			fmt.Printf("  Run %d/%d: %s (%.1fs, %d tok)\n",
				run+1, flagRuns, status, latency, ollamaResp.EvalCount)
		}
	}

	report := computeReport(tasks, results, runs)
	printReport(report)
	saveReport(report)
}

// ---------------------------------------------------------------------------
// Ollama HTTP client
// ---------------------------------------------------------------------------

func callOllama(url, model, prompt string) (OllamaResponse, error) {
	reqBody := OllamaRequest{Model: model, Prompt: prompt, Stream: false}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return OllamaResponse{}, fmt.Errorf("marshal: %w", err)
	}

	resp, err := http.Post(url, "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		return OllamaResponse{}, fmt.Errorf("http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return OllamaResponse{}, fmt.Errorf("status %d: %s", resp.StatusCode, string(body))
	}

	var ollamaResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return OllamaResponse{}, fmt.Errorf("decode: %w", err)
	}
	return ollamaResp, nil
}

// ---------------------------------------------------------------------------
// Metrics computation
// ---------------------------------------------------------------------------

func computeReport(tasks []llmbench.Task, results []llmbench.TaskResult, runs []RunRecord) BenchmarkReport {
	totalRuns := len(results)

	var (
		successCount       int
		actionCorrectCount int
		totalHallucinated  int
		totalEntities      int
		destructiveCount   int
		latencies          []float64
		successLatencies   []float64
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
	}

	esr := llmbench.ExecutionSuccessRate(successCount, totalRuns)
	tsa := llmbench.ToolSelectionAccuracy(actionCorrectCount, totalRuns)
	chr := llmbench.ContextHallucinationRate(totalHallucinated, totalEntities)
	daar := llmbench.DestructiveActionAttemptRate(destructiveCount, totalRuns)

	// FCSR: success on the first run of each task
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

	return BenchmarkReport{
		Metadata: ReportMetadata{
			Model:       flagModel,
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			TotalTasks:  len(tasks),
			RunsPerTask: flagRuns,
			TotalRuns:   totalRuns,
			Seed:        flagSeed,
		},
		Aggregate: AggregateMetrics{
			ESR: esr, ESRCI: ci,
			TSA: tsa, CHR: chr, DAAR: daar,
			FCSR: fcsr, LAE: lae, MTTR: mttr,
			LatencyP50: p50, LatencyP95: p95, LatencyP99: p99,
		},
		PerLevel: computePerLevel(tasks, results),
		RAG:      computeRAGMetrics(tasks),
		PerTask:  computePerTask(tasks, results),
		Runs:     runs,
	}
}

type levelAccum struct {
	success, actionCorrect, hallucinated, entities, total int
}

func computePerLevel(tasks []llmbench.Task, results []llmbench.TaskResult) map[string]LevelMetrics {
	taskLevel := make(map[string]llmbench.TaskLevel, len(tasks))
	for _, t := range tasks {
		taskLevel[t.ID] = t.Level
	}

	accum := make(map[string]*levelAccum)
	for _, r := range results {
		level := string(taskLevel[r.TaskID])
		a, ok := accum[level]
		if !ok {
			a = &levelAccum{}
			accum[level] = a
		}
		a.total++
		if r.DiagnosisCorrect && r.ActionCorrect {
			a.success++
		}
		if r.ActionCorrect {
			a.actionCorrect++
		}
		a.hallucinated += r.HallucinatedArgs
		a.entities += r.TotalArgs
	}

	result := make(map[string]LevelMetrics, len(accum))
	for level, a := range accum {
		result[level] = LevelMetrics{
			ESR:  llmbench.ExecutionSuccessRate(a.success, a.total),
			TSA:  llmbench.ToolSelectionAccuracy(a.actionCorrect, a.total),
			CHR:  llmbench.ContextHallucinationRate(a.hallucinated, a.entities),
			Runs: a.total,
		}
	}
	return result
}

func computeRAGMetrics(tasks []llmbench.Task) RAGQualityMetrics {
	var sumP, sumR, sumMRR, sumNDCG float64
	for _, t := range tasks {
		p, r, m, n := llmbench.ComputeTaskRAGMetrics(t)
		sumP += p
		sumR += r
		sumMRR += m
		sumNDCG += n
	}
	nt := float64(len(tasks))
	meanP := sumP / nt
	meanR := sumR / nt
	return RAGQualityMetrics{
		MeanPrecisionAtK: meanP,
		MeanRecallAtK:    meanR,
		MeanMRR:          sumMRR / nt,
		MeanNDCGAtK:      sumNDCG / nt,
		MeanFScoreAtK:    llmbench.RAGFScoreAtK(meanP, meanR, 1.0),
	}
}

func computePerTask(tasks []llmbench.Task, results []llmbench.TaskResult) []TaskSummary {
	summaries := make([]TaskSummary, 0, len(tasks))
	for _, t := range tasks {
		var success, actionCorrect, hallucinated, entities, total int
		var totalLat float64
		for _, r := range results {
			if r.TaskID != t.ID {
				continue
			}
			total++
			if r.DiagnosisCorrect && r.ActionCorrect {
				success++
			}
			if r.ActionCorrect {
				actionCorrect++
			}
			hallucinated += r.HallucinatedArgs
			entities += r.TotalArgs
			totalLat += r.LatencySec
		}
		meanLat := 0.0
		if total > 0 {
			meanLat = totalLat / float64(total)
		}
		summaries = append(summaries, TaskSummary{
			TaskID:     t.ID,
			Level:      string(t.Level),
			ESR:        llmbench.ExecutionSuccessRate(success, total),
			TSA:        llmbench.ToolSelectionAccuracy(actionCorrect, total),
			CHR:        llmbench.ContextHallucinationRate(hallucinated, entities),
			MeanLatSec: meanLat,
		})
	}
	return summaries
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

func printReport(r BenchmarkReport) {
	sep := strings.Repeat("=", 60)
	fmt.Printf("\n%s\n", sep)
	fmt.Println("BENCHMARK REPORT")
	fmt.Println(sep)
	fmt.Printf("Model:       %s\n", r.Metadata.Model)
	fmt.Printf("Tasks:       %d\n", r.Metadata.TotalTasks)
	fmt.Printf("Runs/task:   %d\n", r.Metadata.RunsPerTask)
	fmt.Printf("Total runs:  %d\n", r.Metadata.TotalRuns)
	fmt.Printf("Timestamp:   %s\n", r.Metadata.Timestamp)

	a := r.Aggregate
	fmt.Println("\n--- AGGREGATE METRICS ---")
	fmt.Printf("ESR  (Execution Success Rate):     %.3f  [95%% CI: %.3f, %.3f]\n", a.ESR, a.ESRCI[0], a.ESRCI[1])
	fmt.Printf("TSA  (Tool Selection Accuracy):    %.3f\n", a.TSA)
	fmt.Printf("CHR  (Context Hallucination Rate): %.3f\n", a.CHR)
	fmt.Printf("DAAR (Destructive Action Rate):    %.3f\n", a.DAAR)
	fmt.Printf("FCSR (First Call Success Rate):    %.3f\n", a.FCSR)
	fmt.Printf("LAE  (Latency-Action Efficiency):  %.4f\n", a.LAE)

	fmt.Println("\n--- LATENCY ---")
	fmt.Printf("p50: %.2fs  |  p95: %.2fs  |  p99: %.2fs\n", a.LatencyP50, a.LatencyP95, a.LatencyP99)
	fmt.Printf("MTTR (successful runs): %.2fs\n", a.MTTR)

	fmt.Println("\n--- PER-LEVEL BREAKDOWN ---")
	for _, level := range []string{
		string(llmbench.LevelDiagnostic),
		string(llmbench.LevelRepair),
		string(llmbench.LevelMultiStep),
	} {
		if m, ok := r.PerLevel[level]; ok {
			fmt.Printf("  %-18s  ESR=%.3f  TSA=%.3f  CHR=%.3f  (n=%d)\n",
				level, m.ESR, m.TSA, m.CHR, m.Runs)
		}
	}

	fmt.Println("\n--- RAG QUALITY (task design validation) ---")
	fmt.Printf("  P@K=%.3f  |  R@K=%.3f  |  F1@K=%.3f  |  MRR=%.3f  |  NDCG@K=%.3f\n",
		r.RAG.MeanPrecisionAtK, r.RAG.MeanRecallAtK,
		r.RAG.MeanFScoreAtK, r.RAG.MeanMRR, r.RAG.MeanNDCGAtK)

	fmt.Println("\n--- PER-TASK SUMMARY ---")
	for _, ts := range r.PerTask {
		fmt.Printf("  %-14s %-18s ESR=%.2f  TSA=%.2f  CHR=%.2f  lat=%.1fs\n",
			ts.TaskID, ts.Level, ts.ESR, ts.TSA, ts.CHR, ts.MeanLatSec)
	}

	fmt.Printf("\nResults saved to: %s\n", flagOutput)
}

func saveReport(r BenchmarkReport) {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		log.Printf("Error marshaling report: %v", err)
		return
	}
	if err := os.WriteFile(flagOutput, data, 0644); err != nil {
		log.Printf("Error writing %s: %v", flagOutput, err)
		return
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func countLevel(tasks []llmbench.Task, level llmbench.TaskLevel) int {
	n := 0
	for _, t := range tasks {
		if t.Level == level {
			n++
		}
	}
	return n
}
