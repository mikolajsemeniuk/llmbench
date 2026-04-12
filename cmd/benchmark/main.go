package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/llmbench"
)

var (
	input    string
	output   string
	host     string
	embed    string
	judge    string
	reranker string
)

type result struct {
	Method      string        `json:"method"`
	MeanScore   float64       `json:"mean_score"`
	TotalTime   time.Duration `json:"total_time_ns"`
	AvgTime     time.Duration `json:"avg_time_ns"`
	TotalTimeMs float64       `json:"total_time_ms"`
	AvgTimeMs   float64       `json:"avg_time_ms"`
	Scores      []float64     `json:"scores,omitempty"`
}

func main() {
	flag.StringVar(&input, "input", "testdata/samples.json", "path to samples JSON file")
	flag.StringVar(&output, "output", "", "path to write benchmark JSON (empty = skip)")
	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL")
	flag.StringVar(&embed, "embed", "nomic-embed-text", "Ollama embedding model")
	flag.StringVar(&judge, "judge", "qwen2.5:7b-instruct-q4_K_M", "Ollama model for G-Eval")
	flag.StringVar(&reranker, "reranker", "http://localhost:8010", "Host API Rerankera na Dockerze")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	samples, err := llmbench.NewSamples(input)
	if err != nil {
		log.Fatalf("load samples: %v", err)
	}

	n := len(samples)
	fmt.Fprintf(os.Stderr, "=== llmbench benchmark (%d samples) ===\n\n", n)

	var results []result

	// BLEU
	results = append(results, runSync("BLEU", samples, func(s llmbench.Sample) (float64, error) {
		return llmbench.BLEU(s.Reference, s.Candidate), nil
	}))

	// ROUGE-L
	results = append(results, runSync("ROUGE-L", samples, func(s llmbench.Sample) (float64, error) {
		return llmbench.ROUGEL(s.Reference, s.Candidate), nil
	}))

	// BERTScore
	bert := llmbench.NewBERTScorer(host, embed)
	results = append(results, runAsync(ctx, "BERTScore", samples, func(ctx context.Context, s llmbench.Sample) (float64, error) {
		return bert.Score(ctx, s.Reference, s.Candidate)
	}))

	// CrossEncoder (bidirectional)
	ce := llmbench.NewCrossEncoderScorer(reranker)
	results = append(results, runAsync(ctx, "CrossEnc", samples, func(ctx context.Context, s llmbench.Sample) (float64, error) {
		return ce.BidirectionalScore(ctx, s.Reference, s.Candidate)
	}))

	// Sentence Coverage F1
	sc := llmbench.NewSentenceCoverageScorer(host, embed)
	results = append(results, runAsync(ctx, "SentCov", samples, func(ctx context.Context, s llmbench.Sample) (float64, error) {
		return sc.Score(ctx, s.Reference, s.Candidate)
	}))

	// Hybrid (all 4 signals)
	hybrid := llmbench.NewHybridScorer(host, embed, reranker)
	results = append(results, runAsync(ctx, "Hybrid", samples, func(ctx context.Context, s llmbench.Sample) (float64, error) {
		return hybrid.Score(ctx, s.Question, s.Reference, s.Candidate)
	}))

	// Hybrid without cross-encoder (ablation — embeddings only)
	hybridNoCE := llmbench.NewHybridNoCrossScorer(host, embed)
	results = append(results, runAsync(ctx, "Hybrid-noCE", samples, func(ctx context.Context, s llmbench.Sample) (float64, error) {
		return hybridNoCE.Score(ctx, s.Question, s.Reference, s.Candidate)
	}))

	// G-Eval
	geval := llmbench.NewGEval(host, judge)
	results = append(results, runAsync(ctx, "G-Eval", samples, func(ctx context.Context, s llmbench.Sample) (float64, error) {
		return geval.Score(ctx, s.Question, s.Reference, s.Candidate)
	}))

	// ── Summary table ──
	fmt.Fprintf(os.Stderr, "\n%-12s  %10s  %10s  %10s\n", "METHOD", "MEAN SCORE", "TOTAL (ms)", "AVG (ms)")
	fmt.Fprintf(os.Stderr, "%-12s  %10s  %10s  %10s\n", "------", "----------", "----------", "-------")
	for _, r := range results {
		fmt.Fprintf(os.Stderr, "%-12s  %10.4f  %10.2f  %10.4f\n",
			r.Method, r.MeanScore, r.TotalTimeMs, r.AvgTimeMs)
	}

	// ── Correlation with G-Eval ──
	gevalIdx := len(results) - 1
	gevalScores := results[gevalIdx].Scores
	if len(gevalScores) > 0 {
		fmt.Fprintf(os.Stderr, "\n%-12s  %10s  %10s\n", "METHOD", "Pearson r", "Spearman ρ")
		fmt.Fprintf(os.Stderr, "%-12s  %10s  %10s\n", "------", "---------", "----------")
		for i, r := range results {
			if i == gevalIdx || len(r.Scores) != len(gevalScores) {
				continue
			}
			p := llmbench.PearsonCorrelation(r.Scores, gevalScores)
			s := llmbench.SpearmanCorrelation(r.Scores, gevalScores)
			fmt.Fprintf(os.Stderr, "%-12s  %10.4f  %10.4f\n", r.Method, p, s)
		}
	}

	// ── Score distributions ──
	fmt.Fprintf(os.Stderr, "\n=== Score Distributions ===\n")
	buckets := []string{"[0.0-0.1)", "[0.1-0.2)", "[0.2-0.3)", "[0.3-0.4)", "[0.4-0.5)", "[0.5-0.6)", "[0.6-0.7)", "[0.7-0.8)", "[0.8-0.9)", "[0.9-1.0]"}
	for _, r := range results {
		counts := make([]int, 10)
		for _, s := range r.Scores {
			b := int(s * 10)
			if b >= 10 {
				b = 9
			}
			if b < 0 {
				b = 0
			}
			counts[b]++
		}
		fmt.Fprintf(os.Stderr, "\n%s:\n", r.Method)
		for i, label := range buckets {
			bar := strings.Repeat("█", counts[i])
			fmt.Fprintf(os.Stderr, "  %s %3d %s\n", label, counts[i], bar)
		}
	}

	if output == "" {
		return
	}
	out, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		log.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(output, append(out, '\n'), 0o644); err != nil {
		log.Fatalf("write: %v", err)
	}
	fmt.Fprintf(os.Stderr, "\nbenchmark results written to %s\n", output)
}

func runSync(method string, samples []llmbench.Sample, fn func(llmbench.Sample) (float64, error)) result {
	fmt.Fprintf(os.Stderr, "[%s] running...\n", method)
	start := time.Now()
	scores := make([]float64, len(samples))
	var sum float64
	for i, s := range samples {
		score, err := fn(s)
		if err != nil {
			log.Fatalf("%s: sample %s: %v", method, s.ID, err)
		}
		scores[i] = score
		sum += score
	}

	total := time.Since(start)
	mean := sum / float64(len(samples))
	avg := total / time.Duration(len(samples))
	return result{
		Method: method, MeanScore: mean,
		TotalTime: total, AvgTime: avg,
		TotalTimeMs: float64(total.Microseconds()) / 1000.0,
		AvgTimeMs:   float64(avg.Microseconds()) / 1000.0,
		Scores:      scores,
	}
}

func runAsync(ctx context.Context, method string, samples []llmbench.Sample, fn func(context.Context, llmbench.Sample) (float64, error)) result {
	fmt.Fprintf(os.Stderr, "[%s] running...\n", method)
	start := time.Now()
	scores := make([]float64, len(samples))
	var sum float64
	for i, s := range samples {
		score, err := fn(ctx, s)
		if err != nil {
			log.Fatalf("%s: sample %s: %v", method, s.ID, err)
		}
		scores[i] = score
		sum += score
	}
	total := time.Since(start)
	mean := sum / float64(len(samples))
	avg := total / time.Duration(len(samples))
	return result{
		Method: method, MeanScore: mean,
		TotalTime: total, AvgTime: avg,
		TotalTimeMs: float64(total.Microseconds()) / 1000.0,
		AvgTimeMs:   float64(avg.Microseconds()) / 1000.0,
		Scores:      scores,
	}
}
