package llmbench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

// ModelServer is a client for the Python modelserver container.
// It provides access to BERTScore (canonical), MoverScore, and UniEval
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

// BERTScoreCanonical computes canonical token-level BERTScore F1
// using RoBERTa-large via the Python bert-score library.
func (m *ModelServer) BERTScoreCanonical(ctx context.Context, reference, candidate string) (float64, error) {
	resp, err := m.post(ctx, "/bertscore", modelServerRequest{
		Reference: reference,
		Candidate: candidate,
	})
	if err != nil {
		return 0, err
	}
	return resp.Score, nil
}

// MoverScore computes Word Mover's Distance with contextual RoBERTa embeddings.
// Returns similarity score in [0, 1].
func (m *ModelServer) MoverScore(ctx context.Context, reference, candidate string) (float64, error) {
	resp, err := m.post(ctx, "/moverscore", modelServerRequest{
		Reference: reference,
		Candidate: candidate,
	})
	if err != nil {
		return 0, err
	}
	return resp.Score, nil
}

// UniEval computes the UniEval score using a T5-based Boolean QA framework.
// Dimension can be: "coherence", "consistency", "fluency", "relevance",
// "overall", or "all".
func (m *ModelServer) UniEval(ctx context.Context, reference, candidate, dimension string) (float64, error) {
	if dimension == "" {
		dimension = "overall"
	}
	resp, err := m.post(ctx, "/unieval", modelServerRequest{
		Reference: reference,
		Candidate: candidate,
		Dimension: dimension,
	})
	if err != nil {
		return 0, err
	}
	return resp.Score, nil
}

// --- Scorer wrappers ---

type BERTScorer struct {
	ctx    context.Context
	server *ModelServer
}

func NewBERTScorer(ctx context.Context, host string) *BERTScorer {
	return &BERTScorer{ctx: ctx, server: NewModelServer(host)}
}

func (b *BERTScorer) Score(entries []Entry) (ScoreOutput, error) {
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
				s, err := b.server.BERTScoreCanonical(b.ctx, ref, mach)
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
			fmt.Fprintf(os.Stderr, "\r[BERTScore] %d/%d", done, total)
		}
	}
	return out, nil
}

type MoverScorer struct {
	ctx    context.Context
	server *ModelServer
}

func NewMoverScorer(ctx context.Context, host string) *MoverScorer {
	return &MoverScorer{ctx: ctx, server: NewModelServer(host)}
}

func (m *MoverScorer) Score(entries []Entry) (ScoreOutput, error) {
	var out ScoreOutput
	total := 0
	for _, e := range entries {
		total += len(e.MachineSummaries)
	}
	done := 0
	for _, e := range entries {
		for en, mach := range e.MachineSummaries {
			best := 0.0
			for _, ref := range e.HumanSummaries {
				s, err := m.server.MoverScore(m.ctx, ref, mach)
				if err != nil {
					return ScoreOutput{}, err
				}
				if s > best {
					best = s
				}
			}
			out.Scores = append(out.Scores, best)
			out.Relevance = append(out.Relevance, e.Relevance[en])
			out.Coherence = append(out.Coherence, e.Coherence[en])
			out.Fluency = append(out.Fluency, e.Fluency[en])
			out.Consistency = append(out.Consistency, e.Consistency[en])
			done++
			fmt.Fprintf(os.Stderr, "\r[MoverScore] %d/%d", done, total)
		}
	}
	return out, nil
}

type UniEvalScorer struct {
	ctx       context.Context
	server    *ModelServer
	dimension string
}

func NewUniEvalScorer(ctx context.Context, host, dimension string) *UniEvalScorer {
	return &UniEvalScorer{ctx: ctx, server: NewModelServer(host), dimension: dimension}
}

func (u *UniEvalScorer) Score(entries []Entry) (ScoreOutput, error) {
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
				s, err := u.server.UniEval(u.ctx, ref, mach, u.dimension)
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
			fmt.Fprintf(os.Stderr, "\r[UniEval/%s] %d/%d", u.dimension, done, total)
		}
	}
	return out, nil
}
