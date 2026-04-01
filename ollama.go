package llmbench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const defaultSeed = 42

type Ollama struct {
	Host        string
	Model       string
	Temperature float64
	Seed        int64
}

func NewOllama(host, model string, temperature float64, seed int64) *Ollama {
	if host == "" {
		host = "http://localhost:11434"
	}

	if seed == 0 {
		seed = defaultSeed
	}

	return &Ollama{
		Host:        host,
		Model:       model,
		Seed:        seed,
		Temperature: temperature,
	}
}

type Response struct {
	Text string `json:"text"`
}

func (o *Ollama) Chat(ctx context.Context, prompt string) (Response, error) {
	var in struct {
		Model   string `json:"model"`
		Prompt  string `json:"prompt"`
		Options struct {
			Temperature float64 `json:"temperature"`
			Seed        int64   `json:"seed"`
		} `json:"options"`
	}

	in.Model = o.Model
	in.Prompt = prompt
	in.Options.Temperature = o.Temperature
	in.Options.Seed = o.Seed

	body, err := json.Marshal(in)
	if err != nil {
		return Response{}, fmt.Errorf("ollama: marshal request: %w", err)
	}

	url := o.Host + "/api/generate"
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
		Response string `json:"response"`
		Done     bool   `json:"done"`
	}
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		return Response{}, fmt.Errorf("ollama: decode response: %w", err)
	}

	out := Response{
		Text: payload.Response,
	}

	return out, nil
}

func (o *Ollama) Embed(ctx context.Context, text string) ([]float64, error) {
	var in struct {
		Model string `json:"model"`
		Input string `json:"input"`
	}
	in.Model = o.Model
	in.Input = text

	body, err := json.Marshal(in)
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal embed request: %w", err)
	}

	url := o.Host + "/api/embed"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: build embed request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: embed http: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("ollama: embed status %d: %s", res.StatusCode, string(raw))
	}

	var payload struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("ollama: decode embed response: %w", err)
	}

	return payload.Embeddings[0], nil
}
