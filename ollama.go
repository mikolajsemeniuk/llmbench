package llmbench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// OllamaProvider implements llmbench.Provider for any model served via
// the Ollama HTTP API (qwen2.5, deepseek-r1, llama3, mistral, …).
//
// Usage:
//
//	p := providers.NewOllamaProvider(providers.OllamaConfig{
//	    Model: "qwen2.5:3b-instruct",
//	})
//	resp, err := p.Complete(ctx, prompt)
type OllamaProvider struct {
	// BaseURL is the root URL of the Ollama server, e.g.
	// "http://localhost:11434". Defaults to that value.
	Host string

	// Model is the Ollama model tag, e.g. "qwen2.5:3b-instruct" or
	// "deepseek-r1:7b". Required — no default.
	Model string

	// Temperature is passed to the model. Must be 0 for reproducibility
	// in benchmark runs. Defaults to 0.
	Temperature float64

	// Seed is passed to Ollama for deterministic sampling. Defaults to 42.
	Seed int64
}

// NewOllamaProvider constructs an OllamaProvider with sensible defaults.
func NewOllamaProvider(host, model string, temperature float64, seed int64) *OllamaProvider {
	if host == "" {
		host = "http://localhost:11434"
	}

	if seed == 0 {
		seed = 42
	}

	return &OllamaProvider{
		Host:        host,
		Model:       model,
		Seed:        seed,
		Temperature: temperature,
	}
}

// Name returns a stable identifier written into every report.
// Format: "ollama/<model-tag>", e.g. "ollama/qwen2.5:3b-instruct".
func (p *OllamaProvider) Name() string {
	return "ollama/" + p.Model
}

// Complete sends prompt to the locally running Ollama instance and returns
// the completion. Stream is always false so the full response arrives in a
// single JSON object — simpler to decode and avoids partial-read errors.
func (p *OllamaProvider) ChatCompletion(ctx context.Context, prompt string) (Response, error) {
	var in struct {
		Model   string `json:"model"`
		Prompt  string `json:"prompt"`
		Stream  bool   `json:"stream"`
		Options struct {
			Temperature float64 `json:"temperature"`
			Seed        int64   `json:"seed"`
		} `json:"options"`
	}

	in.Model = p.Model
	in.Prompt = prompt
	in.Stream = false
	in.Options.Temperature = p.Temperature
	in.Options.Seed = p.Seed

	body, err := json.Marshal(in)
	if err != nil {
		return Response{}, fmt.Errorf("ollama: marshal request: %w", err)
	}

	url := p.Host + "/api/generate"
	reader := bytes.NewReader(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, reader)
	if err != nil {
		return Response{}, fmt.Errorf("ollama: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return Response{}, fmt.Errorf("ollama: http: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(res.Body)
		return Response{}, fmt.Errorf("ollama: status %d: %s", res.StatusCode, string(raw))
	}

	var payload struct {
		Response        string `json:"response"`
		Done            bool   `json:"done"`
		PromptEvalCount int    `json:"prompt_eval_count"`
		EvalCount       int    `json:"eval_count"`
		EvalDuration    int64  `json:"eval_duration"`
	}
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		return Response{}, fmt.Errorf("ollama: decode response: %w", err)
	}

	out := Response{
		Text:             payload.Response,
		PromptTokens:     payload.PromptEvalCount,
		CompletionTokens: payload.EvalCount,
	}

	return out, nil
}

// ModelInfo fetches metadata (digest, family, quantization) from /api/show.
// Called once at startup to populate report.Metadata for reproducibility.
func (p *OllamaProvider) ModelInfo() (OllamaModelInfo, error) {
	url := p.Host + "/api/show"

	body, _ := json.Marshal(map[string]string{"name": p.Model})
	reader := bytes.NewReader(body)
	res, err := http.Post(url, "application/json", reader)
	if err != nil {
		return OllamaModelInfo{}, fmt.Errorf("ollama: model info: %w", err)
	}
	defer res.Body.Close()

	var out OllamaModelInfo
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return OllamaModelInfo{}, fmt.Errorf("ollama: decode model info: %w", err)
	}

	return out, nil
}

// OllamaModelInfo holds the subset of /api/show fields used in report metadata.
type OllamaModelInfo struct {
	Digest  string `json:"digest"`
	Details struct {
		Family            string `json:"family"`
		ParameterSize     string `json:"parameter_size"`
		QuantizationLevel string `json:"quantization_level"`
	} `json:"details"`
}
