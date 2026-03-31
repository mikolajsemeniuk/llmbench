package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/llmbench"
)

type Provider interface {
	Name() string
	ChatCompletion(ctx context.Context, prompt string) (llmbench.Response, error)
}

func main() {
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
	flag.StringVar(&flagProvider, "provider", "ollama", "Model provider: ollama | anthropic")
	flag.StringVar(&flagModel, "model", "qwen2.5:3b-instruct", "Model identifier")
	flag.StringVar(&flagURL, "url", "http://localhost:11434", "Base URL for Ollama server")
	flag.IntVar(&flagRuns, "runs", 10, "Number of independent runs per task")
	flag.StringVar(&flagOutput, "output", "results.json", "Path for the JSON report")
	flag.Int64Var(&flagSeed, "seed", 42, "Random seed for bootstrap CI")
	flag.StringVar(&flagAPIKey, "api-key", "", "API key for API providers")
	flag.IntVar(&flagContextWindow, "context-window", 0, "Model context window in tokens (0 = use provider default)")
	flag.Parse()

	provider := newProvider(flagProvider, flagURL, flagModel, flagSeed, flagAPIKey)
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
		llmbench.CountTasksByLevel(tasks, llmbench.LevelDiagnostic),
		llmbench.CountTasksByLevel(tasks, llmbench.LevelRepair),
		llmbench.CountTasksByLevel(tasks, llmbench.LevelMultiStep),
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

	report := llmbench.BuildEvaluationReport(
		flagSeed, flagRuns,
		providerName, modelName,
		modelDigest, modelFamily, modelQuant,
		tasks, results, records, totalCostUSD, ctxWindow,
	)
	llmbench.PrintReportSummary(os.Stdout, report, flagOutput)
	if err := llmbench.WriteReportJSON(flagOutput, report); err != nil {
		log.Printf("write error: %v", err)
	}
}

func newProvider(flagProvider, flagURL, flagModel string, flagSeed int64, flagAPIKey string) Provider {
	switch strings.ToLower(flagProvider) {
	case "ollama":
		return llmbench.NewOllamaProvider(flagURL, flagModel, 0, flagSeed)
	default:
		log.Fatalf("unknown provider %q — supported: ollama, anthropic", flagProvider)
		return nil
	}
}
