package metrics

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// PSC — Paraphrased-Source Conditioning. CCM showed that likelihood
// signals log P(y|x) are strongly tied to extractiveness (ρ_copy ≈ .74):
// with the source in context, a causal LM predicts spans copied verbatim
// from it almost deterministically, so copying earns a free likelihood
// bonus and even perturbation margins reward it. PSC removes the verbatim
// surface the copy mechanism latches onto: the source is rewritten ONCE
// per article by an instruction LLM into a meaning-preserving paraphrase
// x̃, and every CCM signal is recomputed as log P(· | x̃). A candidate
// that copies x still has to be supported by the facts of x̃, but no
// longer matches it token for token.
//
// Scoring reuses CCM unchanged (same perturbations, same seed, same LM),
// so PSC vs CCM is a paired comparison that differs ONLY in the
// conditioning text.
type Paraphraser struct {
	LLM   *Ollama
	Model string

	// ChunkSents is how many source sentences are rewritten per LLM
	// call. Small chunks keep the rewrite faithful and complete; whole-
	// article prompts tend to summarise instead of paraphrase.
	ChunkSents int

	Seed int64

	// Workers is how many chunks are rewritten concurrently (Ollama
	// batches parallel requests when OLLAMA_NUM_PARALLEL > 1).
	Workers int
}

func NewParaphraser(host, model string) *Paraphraser {
	return &Paraphraser{LLM: NewOllama(host), Model: model, ChunkSents: 3, Seed: 42, Workers: 4}
}

const pscPrompt = `Rewrite the following news text so that it states exactly the same facts in different words.
Rules:
- Keep every name, number, date and claim. Do not add, remove or reinterpret any information.
- Change the wording and sentence structure; avoid reusing any run of more than three consecutive words from the original.
- Output only the rewritten text, with no preamble or notes.

Text:
%s

Rewritten text:`

// Paraphrase rewrites source chunk by chunk (temperature 0, fixed seed)
// and returns the concatenated paraphrase.
func (p *Paraphraser) Paraphrase(ctx context.Context, source string) (string, error) {
	sents := SplitSentences(source)
	if len(sents) == 0 {
		return "", fmt.Errorf("psc: empty source")
	}
	var chunks []string
	for i := 0; i < len(sents); i += p.ChunkSents {
		chunks = append(chunks, strings.Join(sents[i:min(i+p.ChunkSents, len(sents))], " "))
	}
	parts := make([]string, len(chunks))
	errs := make([]error, len(chunks))
	sem := make(chan struct{}, max(1, p.Workers))
	var wg sync.WaitGroup
	for i, chunk := range chunks {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			res, err := p.LLM.Chat(ctx, ChatInput{
				Model:   p.Model,
				Prompt:  fmt.Sprintf(pscPrompt, chunk),
				Options: ChatOptions{Temperature: 0, Seed: p.Seed},
			})
			if err != nil {
				errs[i] = fmt.Errorf("psc: paraphrase chunk %d: %w", i, err)
				return
			}
			out := cleanParaphrase(res.Response)
			if out == "" {
				// Never drop content: fall back to the original chunk.
				out = chunk
			}
			parts[i] = out
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return "", err
		}
	}
	return strings.Join(parts, " "), nil
}

// cleanParaphrase strips the preambles instruction models sometimes add
// despite being told not to ("Here is the rewritten text:").
func cleanParaphrase(s string) string {
	s = strings.TrimSpace(s)
	if first, rest, ok := strings.Cut(s, "\n"); ok {
		f := strings.ToLower(strings.TrimSpace(first))
		if strings.HasPrefix(f, "here") || strings.HasPrefix(f, "rewritten") || strings.HasSuffix(f, ":") {
			s = strings.TrimSpace(rest)
		}
	}
	return strings.Join(strings.Fields(s), " ")
}

// BigramOverlap is the fraction of b's lowercase token bigrams that
// also occur in a — used to report how much verbatim surface the
// paraphrase still shares with the original source.
func BigramOverlap(a, b string) float64 {
	set := map[string]bool{}
	ta, tb := strings.Fields(strings.ToLower(a)), strings.Fields(strings.ToLower(b))
	for i := 0; i+1 < len(ta); i++ {
		set[ta[i]+" "+ta[i+1]] = true
	}
	if len(tb) < 2 {
		return 0
	}
	hit := 0
	for i := 0; i+1 < len(tb); i++ {
		if set[tb[i]+" "+tb[i+1]] {
			hit++
		}
	}
	return float64(hit) / float64(len(tb)-1)
}
