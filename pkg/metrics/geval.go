package metrics

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

// GEval implements a simplified G-Eval (Liu et al., 2023):
// "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment".
type GEval struct {
	Provider *Ollama
	Model    string

	// Temperature and Seed control the LLM-judge sampling. Defaults
	// (0, 42) match the canonical greedy-decoding configuration. Set
	// Temperature>0 and vary Seed across runs to obtain independent
	// G-Eval samples for variance estimation (see -runs flag in
	// cmd/geval).
	Temperature float64
	Seed        int64
}

func NewGEval(host, model string) *GEval {
	return &GEval{Provider: NewOllama(host), Model: model, Seed: 42}
}

type GEvalDimension struct {
	Name     string
	MinScore int
	MaxScore int
	build    func(source, candidate string) string
}

var GEvalDimensions = map[string]GEvalDimension{
	"coherence": {
		Name: "coherence", MinScore: 1, MaxScore: 5,
		build: func(source, candidate string) string {
			return fmt.Sprintf(coherencePrompt, source, candidate)
		},
	},
	"consistency": {
		Name: "consistency", MinScore: 1, MaxScore: 5,
		build: func(source, candidate string) string {
			return fmt.Sprintf(consistencyPrompt, source, candidate)
		},
	},
	"fluency": {
		Name: "fluency", MinScore: 1, MaxScore: 3,
		build: func(_, candidate string) string {
			return fmt.Sprintf(fluencyPrompt, candidate)
		},
	},
	"relevance": {
		Name: "relevance", MinScore: 1, MaxScore: 5,
		build: func(source, candidate string) string {
			return fmt.Sprintf(relevancePrompt, source, candidate)
		},
	},
}

// GEvalResult carries the parsed score plus metadata for diagnostics.
type GEvalResult struct {
	NormalizedScore float64 // [0, 1]
	RawScore        int     // raw integer from model, in [MinScore, MaxScore]
	UsedFallback    bool    // true when parser couldn't extract a clean integer
	RawResponse     string  // model's actual response text
}

// Score evaluates the candidate summary against the source document on the
// given SummEval dimension. Returns a normalized score in [0, 1].
//
// On unparseable responses, returns the dimension midpoint as a fallback
// rather than failing — this keeps the run going and preserves N for
// correlation. Use ScoreDetailed if you need to track fallback frequency.
func (g *GEval) Score(ctx context.Context, dimension, source, candidate string) (float64, error) {
	res, err := g.ScoreDetailed(ctx, dimension, source, candidate)
	if err != nil {
		return 0, err
	}

	return res.NormalizedScore, nil
}

// ScoreDetailed returns the parsed score plus metadata about how it was extracted.
// Use this when you want to track fallback frequency (e.g., warn if >5% of
// responses required fallback).
func (g *GEval) ScoreDetailed(ctx context.Context, dimension, source, candidate string) (GEvalResult, error) {
	dim, ok := GEvalDimensions[dimension]
	if !ok {
		return GEvalResult{}, fmt.Errorf("geval: unknown dimension %q", dimension)
	}

	in := ChatInput{
		Model:  g.Model,
		Prompt: dim.build(source, candidate),
		Stream: false,
		Options: ChatOptions{
			Temperature: g.Temperature,
			Seed:        g.Seed,
		},
	}
	res, err := g.Provider.Chat(ctx, in)
	if err != nil {
		return GEvalResult{}, fmt.Errorf("geval: completion: %w", err)
	}

	raw, fallback := parseGEvalScore(res.Response, dim.MinScore, dim.MaxScore)
	out := GEvalResult{
		NormalizedScore: float64(raw-dim.MinScore) / float64(dim.MaxScore-dim.MinScore),
		RawScore:        raw,
		UsedFallback:    fallback,
		RawResponse:     res.Response,
	}

	return out, nil
}

// parseGEvalScore extracts a score from the model response. Handles:
//   - Plain integers: "4", "Score: 4"
//   - Trailing punctuation: "4.", "4,", "(4)"
//   - Ranges treated as ambiguous, returns lower bound: "1-2" -> 1
//   - Echo of prompt header: "- Fluency (1-2):" -> fallback to midpoint
//
// Returns (score, usedFallback). When usedFallback is true, the score is
// the midpoint of the dimension's valid range — not derived from the
// response — so the caller can decide whether to log a warning.
func parseGEvalScore(raw string, minVal, maxVal int) (int, bool) {
	// Strip the prompt-echo prefix if present (e.g. "- Fluency (1-3):").
	cleaned := stripPromptEcho(raw)

	for _, field := range strings.Fields(cleaned) {
		field = strings.TrimFunc(field, func(r rune) bool {
			return r == '.' || r == ',' || r == ';' || r == ':' ||
				r == '!' || r == '?' || r == '*' ||
				r == '(' || r == ')' || r == '[' || r == ']' ||
				r == '"' || r == '\''
		})
		if field == "" {
			continue
		}

		if n, err := strconv.Atoi(field); err == nil {
			return clamp(n, minVal, maxVal), false
		}

		// Range like "1-2" or "1-3" — take lower bound.
		if idx := strings.Index(field, "-"); idx > 0 && idx < len(field)-1 {
			lo, errLo := strconv.Atoi(field[:idx])
			hi, errHi := strconv.Atoi(field[idx+1:])
			if errLo == nil && errHi == nil && lo >= minVal && hi <= maxVal && lo <= hi {
				// This is a valid range from the model. Take the lower bound
				// conservatively.
				return lo, false
			}
		}
	}

	// Fallback: midpoint of valid range. Caller is informed via UsedFallback.
	return (minVal + maxVal) / 2, true
}

// stripPromptEcho removes the "- Coherence:" / "- Fluency (1-3):" / etc.
// prefix that small models sometimes echo back.
func stripPromptEcho(raw string) string {
	prefixes := []string{
		"- Coherence", "- Consistency", "- Fluency", "- Relevance",
		"Coherence", "Consistency", "Fluency", "Relevance",
	}

	trimmed := strings.TrimSpace(raw)
	lower := strings.ToLower(trimmed)
	for _, p := range prefixes {
		pl := strings.ToLower(p)
		if !strings.HasPrefix(lower, pl) {
			continue
		}

		rest := trimmed[len(p):]
		// Skip optional "(1-3):" / "(1-5):" header.
		rest = strings.TrimSpace(rest)
		if strings.HasPrefix(rest, "(") {
			if end := strings.Index(rest, ")"); end != -1 {
				rest = rest[end+1:]
			}
		}

		rest = strings.TrimLeft(rest, " :\t\n")
		return rest
	}

	return trimmed
}

func clamp(n, minVal, maxVal int) int {
	if n < minVal {
		return minVal
	}

	if n > maxVal {
		return maxVal
	}

	return n
}

// Prompts adapted from Liu et al. 2023, Appendix A.

const coherencePrompt = `You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. The summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.

Source Text:

%s

Summary:

%s

Evaluation Form (scores ONLY):

- Coherence:`

const consistencyPrompt = `You will be given a news article. You will then be given one summary written for this article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully.

Evaluation Criteria:

Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Summaries containing hallucinated facts should be penalized.

Evaluation Steps:

1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency on a scale of 1 to 5 based on the Evaluation Criteria.

Source Text:

%s

Summary:

%s

Evaluation Form (scores ONLY):

- Consistency:`

const fluencyPrompt = `You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully.

Evaluation Criteria:

Fluency (1-3) - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
3: Good. The summary has few or no errors and is easy to read and follow.

Summary:

%s

Evaluation Form (scores ONLY):

- Fluency (1-3):`

const relevancePrompt = `You will be given one summary written for a news article. Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Summaries which contain redundancies and excess information should be penalized.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a score for relevance on a scale of 1 to 5 based on the Evaluation Criteria.

Source Text:

%s

Summary:

%s

Evaluation Form (scores ONLY):

- Relevance:`
