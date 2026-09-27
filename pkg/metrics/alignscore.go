package metrics

import (
	"context"
	"fmt"
)

// AlignScore (Zha et al., ACL 2023) is the published state of the art
// for reference-free factual consistency, used here as the consistency
// baseline LNC has to face. The model server runs AlignScore-large in the
// official "nli_sp" mode (/alignscore); the source and each candidate are
// split into sentences here with SplitSentences instead of NLTK.
type AlignScore struct {
	Server *ModelServer
}

func NewAlignScore(host string) *AlignScore { return &AlignScore{Server: NewModelServer(host)} }

type alignScoreRequest struct {
	SourceSents []string   `json:"source_sents"`
	Candidates  [][]string `json:"candidates"`
}

type alignScoreResponse struct {
	Scores []float64 `json:"scores"`
	Error  string    `json:"error,omitempty"`
}

// ScoreArticle scores all rendered candidates of one article in a single
// request, so the source is chunked once.
func (a *AlignScore) ScoreArticle(ctx context.Context, source string, candidates []string) ([]float64, error) {
	req := alignScoreRequest{SourceSents: SplitSentences(source)}
	for _, c := range candidates {
		req.Candidates = append(req.Candidates, SplitSentences(c))
	}
	var res alignScoreResponse
	if err := a.Server.postJSON(ctx, "/alignscore", req, &res); err != nil {
		return nil, fmt.Errorf("alignscore: %w", err)
	}
	if res.Error != "" {
		return nil, fmt.Errorf("alignscore: %s", res.Error)
	}
	if len(res.Scores) != len(candidates) {
		return nil, fmt.Errorf("alignscore: expected %d scores, got %d", len(candidates), len(res.Scores))
	}
	return res.Scores, nil
}
