package metrics

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// LMServer is the HTTP client for the Python `cmd/lmsrv` app: batched
// teacher-forced log-probabilities from a small causal LM, and NLI
// entailment probabilities. Kept separate from ModelServer because
// lmsrv holds only two small models and is the only server the
// counterfactual-margin metric needs -- a reviewer can reproduce
// cmd/cfm without the five-model baseline server.
type LMServer struct {
	Host   string
	Client *http.Client
}

func NewLMServer(host string) *LMServer {
	return &LMServer{Host: host, Client: &http.Client{}}
}

type lmScoreRequest struct {
	Context string   `json:"context"`
	Targets []string `json:"targets"`
}

type lmScoreResponse struct {
	MeanLogprob   []float64 `json:"mean_logprob"`
	SumLogprob    []float64 `json:"sum_logprob"`
	Tokens        []int     `json:"tokens"`
	ContextTokens int       `json:"context_tokens"`
	Error         string    `json:"error,omitempty"`
}

// ScoreDetail is Score plus the token accounting a hardware-independent
// cost axis needs: how many tokens the LM actually read (context, once
// per call, plus every target) and how many sequences were scored.
type ScoreDetail struct {
	MeanLogprob   []float64
	SumLogprob    []float64
	Tokens        []int
	ContextTokens int
	TargetTokens  int
	Sequences     int
}

// Score returns the mean per-token log-probability of every target,
// conditioned on a shared context (empty for unconditional scoring).
// The context is encoded once per call, so callers should batch every
// target that shares a context -- for cmd/cfm that is all candidates of
// one article plus all of their perturbations.
func (l *LMServer) Score(ctx context.Context, conditioning string, targets []string) ([]float64, error) {
	d, err := l.ScoreDetail(ctx, conditioning, targets)
	if err != nil {
		return nil, err
	}
	return d.MeanLogprob, nil
}

// ScoreDetail is Score with the token counts retained.
func (l *LMServer) ScoreDetail(ctx context.Context, conditioning string, targets []string) (ScoreDetail, error) {
	var out lmScoreResponse
	if err := l.post(ctx, "/lmscore", lmScoreRequest{Context: conditioning, Targets: targets}, &out); err != nil {
		return ScoreDetail{}, err
	}
	if len(out.MeanLogprob) != len(targets) {
		return ScoreDetail{}, fmt.Errorf("lmsrv: got %d scores for %d targets", len(out.MeanLogprob), len(targets))
	}
	d := ScoreDetail{
		MeanLogprob:   out.MeanLogprob,
		SumLogprob:    out.SumLogprob,
		Tokens:        out.Tokens,
		ContextTokens: out.ContextTokens,
		Sequences:     len(targets),
	}
	for _, n := range out.Tokens {
		d.TargetTokens += n
	}
	return d, nil
}

type lmPair struct {
	Context string `json:"context"`
	Target  string `json:"target"`
}

type lmPairsRequest struct {
	Pairs []lmPair `json:"pairs"`
}

// ScorePairs scores a batch of (context, target) pairs whose contexts
// differ. Judge-style prompts embed the candidate, so nothing can share
// an encoded prefix and batching over prompts is the only saving left.
func (l *LMServer) ScorePairs(ctx context.Context, contexts, targets []string) (ScoreDetail, error) {
	if len(contexts) != len(targets) {
		return ScoreDetail{}, fmt.Errorf("lmsrv: %d contexts vs %d targets", len(contexts), len(targets))
	}
	pairs := make([]lmPair, len(contexts))
	for i := range contexts {
		pairs[i] = lmPair{Context: contexts[i], Target: targets[i]}
	}
	var out lmScoreResponse
	if err := l.post(ctx, "/lmpairs", lmPairsRequest{Pairs: pairs}, &out); err != nil {
		return ScoreDetail{}, err
	}
	if len(out.MeanLogprob) != len(pairs) {
		return ScoreDetail{}, fmt.Errorf("lmsrv: got %d scores for %d pairs", len(out.MeanLogprob), len(pairs))
	}
	d := ScoreDetail{
		MeanLogprob:   out.MeanLogprob,
		SumLogprob:    out.SumLogprob,
		Tokens:        out.Tokens,
		ContextTokens: out.ContextTokens,
		Sequences:     len(pairs),
	}
	for _, n := range out.Tokens {
		d.TargetTokens += n
	}
	return d, nil
}

type nliPair struct {
	Premise    string `json:"premise"`
	Hypothesis string `json:"hypothesis"`
}

type nliRequest struct {
	Pairs []nliPair `json:"pairs"`
}

type nliResponse struct {
	Entailment    []float64 `json:"entailment"`
	Contradiction []float64 `json:"contradiction"`
	Error         string    `json:"error,omitempty"`
}

// NLI returns entailment and contradiction probabilities for each
// (premise, hypothesis) pair.
func (l *LMServer) NLI(ctx context.Context, premises, hypotheses []string) (ent, con []float64, err error) {
	if len(premises) != len(hypotheses) {
		return nil, nil, fmt.Errorf("lmsrv: %d premises vs %d hypotheses", len(premises), len(hypotheses))
	}
	pairs := make([]nliPair, len(premises))
	for i := range premises {
		pairs[i] = nliPair{Premise: premises[i], Hypothesis: hypotheses[i]}
	}
	var out nliResponse
	if err := l.post(ctx, "/nli", nliRequest{Pairs: pairs}, &out); err != nil {
		return nil, nil, err
	}
	return out.Entailment, out.Contradiction, nil
}

func (l *LMServer) post(ctx context.Context, endpoint string, req, out any) error {
	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("lmsrv: marshal: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, l.Host+endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("lmsrv: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := l.Client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("lmsrv: http: %w", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(res.Body)
		return fmt.Errorf("lmsrv %s: status %d: %s", endpoint, res.StatusCode, string(raw))
	}
	if err := json.NewDecoder(res.Body).Decode(out); err != nil {
		return fmt.Errorf("lmsrv: decode: %w", err)
	}
	return nil
}
