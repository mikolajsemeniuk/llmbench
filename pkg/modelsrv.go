package llmbench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// ModelServer is a client for the Python modelserver container.
// It provides access to BERTScore, MoverScore, UniEval, GPTScore, BARTScore
// through a unified HTTP API.
type ModelServer struct {
	Host string // e.g. "http://localhost:9200"
}

func NewModelServer(host string) *ModelServer {
	return &ModelServer{Host: host}
}

type modelServerRequest struct {
	Reference string `json:"reference"`
	Candidate string `json:"candidate"`
	Dimension string `json:"dimension,omitempty"`
}

type modelServerResponse struct {
	Score     float64 `json:"score"`
	Precision float64 `json:"precision,omitempty"`
	Recall    float64 `json:"recall,omitempty"`
	Error     string  `json:"error,omitempty"`
}

func (m *ModelServer) post(ctx context.Context, endpoint string, req modelServerRequest) (modelServerResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return modelServerResponse{}, fmt.Errorf("modelserver: marshal: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.Host+endpoint, bytes.NewReader(body))
	if err != nil {
		return modelServerResponse{}, fmt.Errorf("modelserver: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return modelServerResponse{}, fmt.Errorf("modelserver: http: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(res.Body)
		return modelServerResponse{}, fmt.Errorf("modelserver %s: status %d: %s", endpoint, res.StatusCode, string(raw))
	}

	var out modelServerResponse
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return modelServerResponse{}, fmt.Errorf("modelserver: decode: %w", err)
	}
	if out.Error != "" {
		return out, fmt.Errorf("modelserver: %s", out.Error)
	}
	return out, nil
}

// ── BERTScore ─────────────────────────────────────────────────────────

// BERTScorer wraps the canonical token-level BERTScore F1 (Zhang et al. 2020)
// using RoBERTa-large via the Python bert-score library.
type BERTScorer struct {
	Server *ModelServer
}

func NewBERTScorer(host string) *BERTScorer {
	return &BERTScorer{Server: NewModelServer(host)}
}

func (b *BERTScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	res, err := b.Server.post(ctx, "/bertscore", modelServerRequest{
		Reference: reference,
		Candidate: candidate,
	})
	if err != nil {
		return 0, fmt.Errorf("bertscore: %w", err)
	}
	return res.Score, nil
}

// ── MoverScore ────────────────────────────────────────────────────────

// MoverScorer wraps Word Mover's Distance with contextual RoBERTa embeddings
// (Zhao et al. 2019).
type MoverScorer struct {
	Server *ModelServer
}

func NewMoverScorer(host string) *MoverScorer {
	return &MoverScorer{Server: NewModelServer(host)}
}

func (m *MoverScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	res, err := m.Server.post(ctx, "/moverscore", modelServerRequest{
		Reference: reference,
		Candidate: candidate,
	})
	if err != nil {
		return 0, fmt.Errorf("moverscore: %w", err)
	}
	return res.Score, nil
}

// ── UniEval ───────────────────────────────────────────────────────────

// UniEvalScorer wraps the T5-based Boolean QA evaluator (Zhong et al. 2022).
// Each instance is bound to a specific dimension; cmd/unieval runs one
// dimension per invocation.
type UniEvalScorer struct {
	Server    *ModelServer
	Dimension string
}

func NewUniEvalScorer(host, dimension string) *UniEvalScorer {
	if dimension == "" {
		dimension = "overall"
	}
	return &UniEvalScorer{Server: NewModelServer(host), Dimension: dimension}
}

func (u *UniEvalScorer) Score(ctx context.Context, reference, candidate string) (float64, error) {
	res, err := u.Server.post(ctx, "/unieval", modelServerRequest{
		Reference: reference,
		Candidate: candidate,
		Dimension: u.Dimension,
	})
	if err != nil {
		return 0, fmt.Errorf("unieval: %w", err)
	}
	return res.Score, nil
}
