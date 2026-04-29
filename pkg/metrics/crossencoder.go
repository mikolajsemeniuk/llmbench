package metrics

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
)

type CrossEncoderScorer struct {
	Host string
}

func NewCrossEncoderScorer(host string) *CrossEncoderScorer {
	return &CrossEncoderScorer{Host: host}
}

type rerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n"`
}

type rerankResult struct {
	Index          int     `json:"index"`
	CorpusID       int     `json:"corpus_id"`
	Score          float64 `json:"score"`
	RelevanceScore float64 `json:"relevance_score"`
}

// Score computes the cross-encoder relevance score between reference and candidate.
// If the raw score is outside [0, 1], it is normalized via sigmoid.
func (c *CrossEncoderScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	score, err := c.rawScore(ctx, reference, candidate)
	if err != nil {
		return 0, err
	}

	return normalizeScore(score), nil
}

func (c *CrossEncoderScorer) BidirectionalScore(ctx context.Context, reference, candidate string) (float64, error) {
	fwd, err := c.rawScore(ctx, reference, candidate)
	if err != nil {
		return 0, fmt.Errorf("cross-encoder forward: %w", err)
	}

	bwd, err := c.rawScore(ctx, candidate, reference)
	if err != nil {
		return 0, fmt.Errorf("cross-encoder backward: %w", err)
	}

	fwdNorm := normalizeScore(fwd)
	bwdNorm := normalizeScore(bwd)

	if fwdNorm+bwdNorm == 0 {
		return 0, nil
	}
	return 2 * fwdNorm * bwdNorm / (fwdNorm + bwdNorm), nil
}

func (c *CrossEncoderScorer) rawScore(ctx context.Context, query, text string) (float64, error) {
	in := rerankRequest{
		Query:     query,
		Documents: []string{text},
		TopN:      1,
	}

	body, err := json.Marshal(in)
	if err != nil {
		return 0, fmt.Errorf("cross-encoder: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Host+"/v1/rerank", bytes.NewReader(body))
	if err != nil {
		return 0, fmt.Errorf("cross-encoder: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, fmt.Errorf("cross-encoder: http: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(res.Body)
		return 0, fmt.Errorf("cross-encoder: status %d: %s", res.StatusCode, string(raw))
	}

	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return 0, fmt.Errorf("cross-encoder: read body: %w", err)
	}

	var wrapped struct {
		Results []rerankResult `json:"results"`
	}
	if err := json.Unmarshal(raw, &wrapped); err == nil && len(wrapped.Results) > 0 {
		r := wrapped.Results[0]
		if r.RelevanceScore != 0 {
			return r.RelevanceScore, nil
		}

		return r.Score, nil
	}

	var results []rerankResult
	if err := json.Unmarshal(raw, &results); err == nil && len(results) > 0 {
		r := results[0]
		if r.RelevanceScore != 0 {
			return r.RelevanceScore, nil
		}
		return r.Score, nil
	}

	return 0, fmt.Errorf("cross-encoder: could not parse response: %s", string(raw))
}

func normalizeScore(x float64) float64 {
	if x >= 0 && x <= 1 {
		return x
	}

	return sigmoid(x)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
