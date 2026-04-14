package llmbench

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type GEval struct {
	ctx      context.Context
	Provider *Ollama
	Model    string
}

func NewGEval(ctx context.Context, host, model string) *GEval {
	return &GEval{
		ctx:      ctx,
		Provider: NewOllama(host),
		Model:    model,
	}
}

const gevalPrompt = `You are an expert evaluator. Given a question, a reference answer (ground truth), and a candidate answer produced by an AI model, rate the candidate on two dimensions.

QUESTION:
%s

REFERENCE ANSWER:
%s

CANDIDATE ANSWER:
%s

Rate the candidate answer on each dimension using an integer from 1 to 5:

1. CORRECTNESS (1-5): Are all facts in the candidate answer accurate compared to the reference?
   1 = mostly wrong, 2 = several errors, 3 = some errors, 4 = minor issues, 5 = fully correct

2. COMPLETENESS (1-5): Does the candidate cover the key information from the reference?
   1 = mostly missing, 2 = significant gaps, 3 = partial, 4 = nearly complete, 5 = fully complete

Respond ONLY with two integers separated by a space. Example: 4 3
Do not include any other text.`

// score asks the LLM to rate the candidate and returns a normalised score
// in [0, 1] (average of correctness and completeness, each divided by 5).
func (g *GEval) score(ctx context.Context, question, reference, candidate string) (float64, error) {
	prompt := fmt.Sprintf(gevalPrompt, question, reference, candidate)

	input := ChatInput{
		Model:  g.Model,
		Prompt: prompt,
		Stream: false,
		Options: ChatOptions{
			Temperature: 0,
			Seed:        42,
		},
	}
	res, err := g.Provider.Chat(ctx, input)
	if err != nil {
		return 0, fmt.Errorf("geval: completion: %w", err)
	}

	correctness, completeness, err := parseGEvalResponse(res.Response)
	if err != nil {
		return 0, fmt.Errorf("geval: parse response %q: %w", res.Response, err)
	}

	score := (float64(correctness) + float64(completeness)) / 10.0
	return score, nil
}

// Score implements Scorer.
func (g *GEval) Score(entries []Entry) (ScoreOutput, error) {
	var out ScoreOutput
	total := 0
	for _, e := range entries {
		total += len(e.MachineSummaries)
	}
	done := 0
	for _, e := range entries {
		for mi, mach := range e.MachineSummaries {
			best := 0.0
			for _, ref := range e.HumanSummaries {
				s, err := g.score(g.ctx, e.Text, ref, mach)
				if err != nil {
					return ScoreOutput{}, err
				}
				if s > best {
					best = s
				}
			}
			out.Scores = append(out.Scores, best)
			out.Relevance = append(out.Relevance, e.Relevance[mi])
			out.Coherence = append(out.Coherence, e.Coherence[mi])
			out.Fluency = append(out.Fluency, e.Fluency[mi])
			out.Consistency = append(out.Consistency, e.Consistency[mi])
			done++
			fmt.Fprintf(os.Stderr, "\r[G-Eval] %d/%d", done, total)
		}
	}
	return out, nil
}

// parseGEvalResponse extracts two integers from the LLM output.
// Handles formats like "4 3", "4, 3", "Correctness: 4\nCompleteness: 3".
func parseGEvalResponse(raw string) (int, int, error) {
	nums := extractInts(raw)
	if len(nums) >= 2 {
		c, k := clampScore(nums[0]), clampScore(nums[1])
		return c, k, nil
	}

	if len(nums) == 1 {
		c := clampScore(nums[0])
		return c, c, nil
	}

	return 0, 0, fmt.Errorf("no integers found in response")
}

func extractInts(s string) []int {
	var result []int
	for _, field := range strings.Fields(s) {
		field = strings.TrimRight(field, ".,;:!?")
		n, err := strconv.Atoi(field)
		if err == nil {
			result = append(result, n)
		}
	}

	return result
}

func clampScore(n int) int {
	if n < 1 {
		return 1
	}

	if n > 5 {
		return 5
	}

	return n
}
