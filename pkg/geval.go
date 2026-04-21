package llmbench

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

// GEval implements a simplified G-Eval (Liu et al., 2023):
// "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment".
//
// Reference-free: scores a candidate summary against the source document
// on a single SummEval dimension. Use one instance per dimension; the
// cmd/geval binary runs one dimension per invocation.
//
// Simplifications vs. the original paper:
//   - Greedy decoding (temperature=0, seed=42) instead of n=20 samples
//     with probability-weighted aggregation via token logprobs.
//   - Fixed CoT steps from the paper's appendix A rather than auto-generated.
type GEval struct {
	Provider *Ollama
	Model    string
}

func NewGEval(host, model string) *GEval {
	return &GEval{Provider: NewOllama(host), Model: model}
}

// GEvalDimension describes a single SummEval dimension prompt.
type GEvalDimension struct {
	Name     string
	MinScore int
	MaxScore int
	build    func(source, candidate string) string
}

// GEvalDimensions holds the 4 canonical SummEval dimensions.
// Fluency is on 1-3 scale per Liu et al. 2023; the others are 1-5.
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

// Score evaluates the candidate summary against the source document on the
// given SummEval dimension. Returns a normalized score in [0, 1].
func (g *GEval) Score(ctx context.Context, dimension, source, candidate string) (float64, error) {
	dim, ok := GEvalDimensions[dimension]
	if !ok {
		return 0, fmt.Errorf("geval: unknown dimension %q", dimension)
	}

	res, err := g.Provider.Chat(ctx, ChatInput{
		Model:  g.Model,
		Prompt: dim.build(source, candidate),
		Stream: false,
		Options: ChatOptions{
			Temperature: 0,
			Seed:        42,
		},
	})
	if err != nil {
		return 0, fmt.Errorf("geval: completion: %w", err)
	}

	raw, err := parseGEvalScore(res.Response, dim.MinScore, dim.MaxScore)
	if err != nil {
		return 0, fmt.Errorf("geval: parse response %q: %w", res.Response, err)
	}

	return float64(raw-dim.MinScore) / float64(dim.MaxScore-dim.MinScore), nil
}

// parseGEvalScore extracts the first integer from the model response
// and clamps it to the dimension's valid range.
func parseGEvalScore(raw string, minVal, maxVal int) (int, error) {
	for _, field := range strings.Fields(raw) {
		field = strings.TrimFunc(field, func(r rune) bool {
			return r == '.' || r == ',' || r == ';' || r == ':' ||
				r == '!' || r == '?' || r == '-' || r == '*' ||
				r == '(' || r == ')' || r == '[' || r == ']'
		})
		n, err := strconv.Atoi(field)
		if err != nil {
			continue
		}
		if n < minVal {
			return minVal, nil
		}
		if n > maxVal {
			return maxVal, nil
		}
		return n, nil
	}
	return 0, fmt.Errorf("no integers found")
}

// Prompts below are adapted from Liu et al. 2023, "G-Eval: NLG Evaluation
// using GPT-4 with Better Human Alignment", Appendix A.

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
