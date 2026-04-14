package llmbench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// BARTScorer computes BARTScore using log-probability of generating
// the candidate text conditioned on the reference text.
// It uses Ollama's OpenAI-compatible /v1/completions endpoint with logprobs.
type BARTScorer struct {
	ctx   context.Context
	Host  string
	Model string
}

func NewBARTScorer(ctx context.Context, host, model string) *BARTScorer {
	return &BARTScorer{ctx: ctx, Host: host, Model: model}
}

const bartSep = "\nSummary:\n"

// score computes BARTScore: average log P(candidate_token | reference, preceding tokens).
// Higher score = model finds candidate more likely given reference = better summary.
func (b *BARTScorer) score(ctx context.Context, reference, candidate string) (float64, error) {
	prompt := reference + bartSep + candidate

	logprobs, err := b.completionLogprobs(ctx, prompt)
	if err != nil {
		return 0, err
	}

	// Find where the candidate portion starts in the token stream.
	// We look for the separator tokens and take everything after.
	sepLower := strings.ToLower(bartSep)
	accumulated := ""
	startIdx := -1
	for i, tok := range logprobs.Tokens {
		accumulated += tok
		if startIdx == -1 && strings.Contains(strings.ToLower(accumulated), strings.TrimSpace(sepLower)) {
			startIdx = i + 1
		}
	}

	if startIdx == -1 || startIdx >= len(logprobs.TokenLogprobs) {
		return 0, nil
	}

	// Average log probability of candidate tokens.
	var sum float64
	count := 0
	for i := startIdx; i < len(logprobs.TokenLogprobs); i++ {
		sum += logprobs.TokenLogprobs[i]
		count++
	}
	if count == 0 {
		return 0, nil
	}
	return sum / float64(count), nil
}

// maxScore returns the best BARTScore of candidate against all references.
func (b *BARTScorer) maxScore(ctx context.Context, references []string, candidate string) (float64, error) {
	// BARTScore is negative (log probs), so "best" = closest to 0 = maximum.
	first := true
	best := 0.0
	for _, ref := range references {
		s, err := b.score(ctx, ref, candidate)
		if err != nil {
			return 0, err
		}
		if first || s > best {
			best = s
			first = false
		}
	}
	return best, nil
}

// Score implements Scorer: scores every (entry, machine-summary) pair against all human summaries.
func (b *BARTScorer) Score(entries []Entry) (ScoreOutput, error) {
	var out ScoreOutput
	total := 0
	for _, e := range entries {
		total += len(e.MachineSummaries)
	}
	done := 0
	for _, e := range entries {
		for mi, mach := range e.MachineSummaries {
			s, err := b.maxScore(b.ctx, e.HumanSummaries, mach)
			if err != nil {
				return ScoreOutput{}, err
			}
			out.Scores = append(out.Scores, s)
			out.Relevance = append(out.Relevance, e.Relevance[mi])
			out.Coherence = append(out.Coherence, e.Coherence[mi])
			out.Fluency = append(out.Fluency, e.Fluency[mi])
			out.Consistency = append(out.Consistency, e.Consistency[mi])
			done++
			fmt.Fprintf(os.Stderr, "\r[BARTScore] %d/%d", done, total)
		}
	}
	return out, nil
}

// --- Ollama OpenAI-compatible completions with logprobs ---

type completionRequest struct {
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"`
	Echo      bool   `json:"echo"`
	Logprobs  int    `json:"logprobs"`
}

type completionResponse struct {
	Choices []struct {
		Logprobs struct {
			Tokens        []string  `json:"tokens"`
			TokenLogprobs []float64 `json:"token_logprobs"`
		} `json:"logprobs"`
	} `json:"choices"`
}

type logprobsResult struct {
	Tokens        []string
	TokenLogprobs []float64
}

func (b *BARTScorer) completionLogprobs(ctx context.Context, prompt string) (*logprobsResult, error) {
	body, err := json.Marshal(completionRequest{
		Model:     b.Model,
		Prompt:    prompt,
		MaxTokens: 0,
		Echo:      true,
		Logprobs:  1,
	})
	if err != nil {
		return nil, err
	}

	url := strings.TrimRight(b.Host, "/") + "/v1/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bartscore: request: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("bartscore: read body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bartscore: status %d: %s", resp.StatusCode, string(data))
	}

	var cr completionResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return nil, fmt.Errorf("bartscore: decode: %w", err)
	}

	if len(cr.Choices) == 0 {
		return nil, fmt.Errorf("bartscore: empty response")
	}

	lp := cr.Choices[0].Logprobs
	return &logprobsResult{
		Tokens:        lp.Tokens,
		TokenLogprobs: lp.TokenLogprobs,
	}, nil
}
