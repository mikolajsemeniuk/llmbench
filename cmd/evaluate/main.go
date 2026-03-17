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
	Model   string        `json:"model"`
	Prompt  string        `json:"prompt"`
	Stream  bool          `json:"stream"`
	Options OllamaOptions `json:"options"`
}

type OllamaOptions struct {
	Temperature float64 `json:"temperature"`
	Seed        int64   `json:"seed"`
}

type OllamaResponse struct {
	Response        string `json:"response"`
	Done            bool   `json:"done"`
	TotalDuration   int64  `json:"total_duration"`
	PromptEvalCount int    `json:"prompt_eval_count"`
	EvalCount       int    `json:"eval_count"`
	EvalDuration    int64  `json:"eval_duration"`
}

type OllamaShowRequest struct {
	Name string `json:"name"`
}

type OllamaShowResponse struct {
	Digest  string `json:"digest"`
	Details struct {
		Format            string `json:"format"`
		Family            string `json:"family"`
		ParameterSize     string `json:"parameter_size"`
		QuantizationLevel string `json:"quantization_level"`
	} `json:"details"`
}

// ---------------------------------------------------------------------------
// Report types — structured output for reproducibility
// ---------------------------------------------------------------------------

type Report struct {
	Metadata  Metadata          `json:"metadata"`
	Metrics   Metrics           `json:"aggregate"`
	PerLevel  []LevelMetrics    `json:"per_level"`
	RAG       RAGQualityMetrics `json:"rag_quality"`
	Summaries []Summary         `json:"per_task"`
	Records   []Record          `json:"runs"`
}

type Metadata struct {
	Model       string `json:"model"`
	ModelDigest string `json:"model_digest"`
	ModelFamily string `json:"model_family"`
	ModelQuant  string `json:"model_quantization"`
	Timestamp   string `json:"timestamp"`
	TotalTasks  int    `json:"total_tasks"`
	RunsPerTask int    `json:"runs_per_task"`
	TotalRuns   int    `json:"total_runs"`
	Seed        int64  `json:"random_seed"`
}

type Metrics struct {
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
	Name string  `json:"name"`
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

type Summary struct {
	TaskID     string  `json:"task_id"`
	Level      string  `json:"level"`
	ESR        float64 `json:"esr"`
	TSA        float64 `json:"tsa"`
	CHR        float64 `json:"chr"`
	MeanLatSec float64 `json:"mean_latency_sec"`
}

type Record struct {
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
	model    string
	flagURL  string
	flagRuns int
	output   string
	seed     int64
)

func main() {
	flag.StringVar(&model, "model", "qwen2.5:3b-instruct", "Ollama model name")
	flag.StringVar(&flagURL, "url", "http://localhost:11434/api/generate", "Ollama API URL")
	flag.IntVar(&flagRuns, "runs", 5, "Number of independent runs per task")
	flag.StringVar(&output, "output", "results.json", "Output JSON file path")
	flag.Int64Var(&seed, "seed", 42, "Random seed for bootstrap CI (reproducibility)")
	flag.Parse()

	tasks := llmbench.BenchmarkTasks()
	totalRuns := len(tasks) * flagRuns

	fmt.Println("=== LLMBench: K8s MCP Benchmark ===")
	fmt.Printf("Model:       %s\n", model)
	fmt.Printf("Tasks:       %d (L1=%d, L2=%d, L3=%d)\n",
		len(tasks), countLevel(tasks, llmbench.LevelDiagnostic),
		countLevel(tasks, llmbench.LevelRepair), countLevel(tasks, llmbench.LevelMultiStep))
	fmt.Printf("Runs/task:   %d\n", flagRuns)
	fmt.Printf("Total runs:  %d\n", totalRuns)
	fmt.Printf("Seed:        %d\n", seed)
	fmt.Println()

	modelInfo := fetchModelInfo(flagURL, model)
	if modelInfo.Digest != "" {
		fmt.Printf("Digest:      %s\n", modelInfo.Digest[:12])
		fmt.Printf("Family:      %s\n", modelInfo.Details.Family)
		fmt.Printf("Quant:       %s\n", modelInfo.Details.QuantizationLevel)
	}
	fmt.Println()

	var (
		results []llmbench.Result
		runs    []Record
	)

	for _, task := range tasks {
		fmt.Printf("[%s] %s\n", task.ID, task.Description)
		prompt := llmbench.BuildPrompt(task)

		for run := 0; run < flagRuns; run++ {
			start := time.Now()
			ollamaResp, err := callOllama(flagURL, model, prompt)
			latency := time.Since(start).Seconds()

			if err != nil {
				fmt.Printf("  Run %d/%d: ERROR (%v)\n", run+1, flagRuns, err)
				result := llmbench.Result{
					TaskID:           task.ID,
					RunIndex:         run,
					LatencySec:       latency,
					TotalArgs:        len(task.GroundTruth.ContextEntities),
					HallucinatedArgs: len(task.GroundTruth.ContextEntities),
				}
				results = append(results, result)

				record := Record{
					TaskID:         task.ID,
					RunIndex:       run,
					LatencySec:     latency,
					TotalEntities:  len(task.GroundTruth.ContextEntities),
					Hallucinations: len(task.GroundTruth.ContextEntities),
					Error:          err.Error(),
				}
				runs = append(runs, record)

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

			record := Record{
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
			}
			runs = append(runs, record)

			status := "FAIL"
			if eval.DiagnosisCorrect && eval.ActionCorrect {
				status = "PASS"
			}

			fmt.Printf("  Run %d/%d: %s (%.1fs, %d tok)\n", run+1, flagRuns, status, latency, ollamaResp.EvalCount)
		}
	}

	report := newReport(tasks, results, runs, modelInfo)
	printReport(report)
	saveReport(report)
}

func callOllama(url, model, prompt string) (OllamaResponse, error) {
	in := OllamaRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
		Options: OllamaOptions{
			Temperature: 0,
			Seed:        int64(seed),
		},
	}

	body, err := json.Marshal(in)
	if err != nil {
		return OllamaResponse{}, fmt.Errorf("marshal: %w", err)
	}

	res, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return OllamaResponse{}, fmt.Errorf("http: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(res.Body)
		return OllamaResponse{}, fmt.Errorf("status %d: %s", res.StatusCode, string(body))
	}

	var out OllamaResponse
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return OllamaResponse{}, fmt.Errorf("decode: %w", err)
	}

	return out, nil
}

func fetchModelInfo(url, model string) OllamaShowResponse {
	showURL := strings.TrimSuffix(url, "/api/generate") + "/api/show"
	body, _ := json.Marshal(OllamaShowRequest{Name: model})
	res, err := http.Post(showURL, "application/json", bytes.NewReader(body))
	if err != nil {
		log.Printf("Warning: cannot fetch model info: %v", err)
		return OllamaShowResponse{}
	}
	defer res.Body.Close()

	var out OllamaShowResponse
	json.NewDecoder(res.Body).Decode(&out)
	return out
}

// ---------------------------------------------------------------------------
// Metrics computation
// ---------------------------------------------------------------------------

func newReport(tasks []llmbench.Task, results []llmbench.Result, runs []Record, modelInfo OllamaShowResponse) Report {
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

	rng := rand.New(rand.NewSource(seed))
	ci := llmbench.BootstrapConfidenceInterval(successCount, totalRuns, 10000, 0.05, rng.Float64)

	return Report{
		Metadata: Metadata{
			Model:       model,
			ModelDigest: modelInfo.Digest,
			ModelFamily: modelInfo.Details.Family,
			ModelQuant:  modelInfo.Details.QuantizationLevel,
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			TotalTasks:  len(tasks),
			RunsPerTask: flagRuns,
			TotalRuns:   totalRuns,
			Seed:        seed,
		},
		Metrics: Metrics{
			ESR: esr, ESRCI: ci,
			TSA: tsa, CHR: chr, DAAR: daar,
			FCSR: fcsr, LAE: lae, MTTR: mttr,
			LatencyP50: p50, LatencyP95: p95, LatencyP99: p99,
		},
		PerLevel:  computePerLevel(tasks, results),
		RAG:       computeRAGMetrics(tasks),
		Summaries: computePerTask(tasks, results),
		Records:   runs,
	}
}

type levelAccum struct {
	success, actionCorrect, hallucinated, entities, total int
}

func computePerLevel(tasks []llmbench.Task, results []llmbench.Result) []LevelMetrics {
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

	order := []string{
		string(llmbench.LevelDiagnostic),
		string(llmbench.LevelRepair),
		string(llmbench.LevelMultiStep),
	}

	var out []LevelMetrics
	for _, level := range order {
		v, ok := accum[level]
		if !ok {
			continue
		}

		lm := LevelMetrics{
			Name: level,
			ESR:  llmbench.ExecutionSuccessRate(v.success, v.total),
			TSA:  llmbench.ToolSelectionAccuracy(v.actionCorrect, v.total),
			CHR:  llmbench.ContextHallucinationRate(v.hallucinated, v.entities),
			Runs: v.total,
		}

		out = append(out, lm)
	}

	return out
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

func computePerTask(tasks []llmbench.Task, results []llmbench.Result) []Summary {
	summaries := make([]Summary, 0, len(tasks))
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
		summaries = append(summaries, Summary{
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

func printReport(r Report) {
	sep := strings.Repeat("=", 60)
	fmt.Printf("\n%s\n", sep)
	fmt.Println("BENCHMARK REPORT")
	fmt.Println(sep)
	fmt.Printf("Model:       %s\n", r.Metadata.Model)
	fmt.Printf("Tasks:       %d\n", r.Metadata.TotalTasks)
	fmt.Printf("Runs/task:   %d\n", r.Metadata.RunsPerTask)
	fmt.Printf("Total runs:  %d\n", r.Metadata.TotalRuns)
	fmt.Printf("Timestamp:   %s\n", r.Metadata.Timestamp)

	a := r.Metrics
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
	for _, m := range r.PerLevel {
		fmt.Printf("  %-18s  ESR=%.3f  TSA=%.3f  CHR=%.3f  (n=%d)\n",
			m.Name, m.ESR, m.TSA, m.CHR, m.Runs)
	}

	fmt.Println("\n--- RAG QUALITY (task design validation) ---")
	fmt.Printf("  P@K=%.3f  |  R@K=%.3f  |  F1@K=%.3f  |  MRR=%.3f  |  NDCG@K=%.3f\n",
		r.RAG.MeanPrecisionAtK, r.RAG.MeanRecallAtK,
		r.RAG.MeanFScoreAtK, r.RAG.MeanMRR, r.RAG.MeanNDCGAtK)

	fmt.Println("\n--- PER-TASK SUMMARY ---")
	for _, ts := range r.Summaries {
		fmt.Printf("  %-14s %-18s ESR=%.2f  TSA=%.2f  CHR=%.2f  lat=%.1fs\n",
			ts.TaskID, ts.Level, ts.ESR, ts.TSA, ts.CHR, ts.MeanLatSec)
	}

	fmt.Printf("\nResults saved to: %s\n", output)
}

func saveReport(r Report) {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		log.Printf("Error marshaling report: %v", err)
		return
	}

	if err := os.WriteFile(output, data, 0644); err != nil {
		log.Printf("Error writing %s: %v", output, err)
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
