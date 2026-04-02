package llmbench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type Ollama struct {
	Host string
}

func NewOllama(host string) *Ollama {
	return &Ollama{Host: host}
}

type ChatInput struct {
	Model   string      `json:"model"`
	Prompt  string      `json:"prompt"`
	Options ChatOptions `json:"options"`
}

type ChatOptions struct {
	Temperature float64 `json:"temperature"`
	Seed        int64   `json:"seed"`
}

type ChatOutput struct {
	Response string `json:"response"`
}

func (o *Ollama) Chat(ctx context.Context, in ChatInput) (ChatOutput, error) {
	input, err := json.Marshal(in)
	if err != nil {
		return ChatOutput{}, fmt.Errorf("ollama: marshal request: %w", err)
	}

	url := o.Host + "/api/generate"
	body := bytes.NewReader(input)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return ChatOutput{}, fmt.Errorf("ollama: build request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return ChatOutput{}, fmt.Errorf("ollama: http: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return ChatOutput{}, fmt.Errorf("ollama: status %d", res.StatusCode)
	}

	var out ChatOutput
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return ChatOutput{}, fmt.Errorf("ollama: decode response: %w", err)
	}

	return out, nil
}

type EmbedInput struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type EmbedOutput struct {
	Embeddings [][]float64 `json:"embeddings"`
}

func (o *Ollama) Embed(ctx context.Context, in EmbedInput) (EmbedOutput, error) {
	body, err := json.Marshal(in)
	if err != nil {
		return EmbedOutput{}, fmt.Errorf("ollama: marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.Host+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return EmbedOutput{}, fmt.Errorf("ollama: build embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return EmbedOutput{}, fmt.Errorf("ollama: embed http: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(res.Body)
		return EmbedOutput{}, fmt.Errorf("ollama: embed status %d: %s", res.StatusCode, string(raw))
	}

	var out EmbedOutput
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return EmbedOutput{}, fmt.Errorf("ollama: decode embed response: %w", err)
	}

	return out, nil
}
