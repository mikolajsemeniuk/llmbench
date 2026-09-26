package metrics

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// ModelServer is the HTTP client for the Python `cmd/modelsrv` Flask
// app. It serves all metrics that need a heavyweight transformer model
// behind a unified API: BERTScore, MoverScore, UniEval, GPTScore,
// BARTScore. Each metric lives in its own file (bertscorer.go,
// moverscorer.go, unieval.go, gptscorer.go, bartscorer.go) and uses
// ModelServer through `post`.
type ModelServer struct {
	Host string // e.g. "http://localhost:9200"
}

func NewModelServer(host string) *ModelServer {
	return &ModelServer{Host: host}
}

type modelServerRequest struct {
	Reference string `json:"reference,omitempty"`
	Candidate string `json:"candidate"`
	Source    string `json:"source,omitempty"` // for UniEval coherence/consistency
	Dimension string `json:"dimension,omitempty"`
}

type modelServerResponse struct {
	Score     float64 `json:"score"`
	Precision float64 `json:"precision,omitempty"`
	Recall    float64 `json:"recall,omitempty"`
	Error     string  `json:"error,omitempty"`
}

func (m *ModelServer) post(ctx context.Context, endpoint string, req modelServerRequest) (modelServerResponse, error) {
	var out modelServerResponse
	if err := m.postJSON(ctx, endpoint, req, &out); err != nil {
		return modelServerResponse{}, err
	}
	if out.Error != "" {
		return out, fmt.Errorf("modelserver: %s", out.Error)
	}
	return out, nil
}

// postJSON sends req as JSON to endpoint and decodes the response into
// out. Used directly by metrics whose wire format does not fit
// modelServerRequest (CCM sends a batch of candidates per source).
func (m *ModelServer) postJSON(ctx context.Context, endpoint string, req, out any) error {
	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("modelserver: marshal: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.Host+endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("modelserver: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("modelserver: http: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(res.Body)
		return fmt.Errorf("modelserver %s: status %d: %s", endpoint, res.StatusCode, string(raw))
	}

	if err := json.NewDecoder(res.Body).Decode(out); err != nil {
		return fmt.Errorf("modelserver: decode: %w", err)
	}
	return nil
}
