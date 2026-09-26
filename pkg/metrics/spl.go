package metrics

import (
	"context"
	"fmt"
	"math"
	"strings"
	"unicode"
)

// SPL — Splice-Point Likelihood. On SummEval the median candidate copies
// 92% of its bigrams from the source: most "summaries" are verbatim
// source fragments glued together. Inside a copied fragment a causal LM
// that has the source in context predicts every token almost for free,
// which is why plain log P(y|x) tracks copy rate (ρ ≈ .74, see CCM).
// The places where quality is actually decided are the SPLICE POINTS:
// the first word of every fragment and every word that is not a
// continuation of a source bigram. A clean splice ("Sterling's wife
// sued V. Stiviano ...") is a plausible continuation; a broken one
// ("Donald Sterling, NBA team last year.") is not.
//
// SPL therefore scores only splice words:
//
//	w_i is a splice word  ⇔  i = 0  or  (w_{i-1}, w_i) is not a source bigram
//	score = mean over splice words of log P(w_i | x, w_<i)
//
// under the same small causal LM as CCM (one source pass per article,
// no perturbations). Word log-probability is the sum over the LM tokens
// whose first non-space character falls in the word. A candidate that
// is one verbatim source sentence has a single splice word, its first.
type SPL struct {
	Server *ModelServer
}

func NewSPL(host string) *SPL { return &SPL{Server: NewModelServer(host)} }

// SPLWord is one word of the candidate with its LM evidence.
type SPLWord struct {
	Word   string  `json:"w"`
	Splice bool    `json:"s"`
	Cond   float64 `json:"c"` // log P(word | x, prefix)
	Uncond float64 `json:"u"` // log P(word | prefix)
}

type SPLResult struct {
	Words []SPLWord `json:"words"`
}

func (r SPLResult) agg(splice bool, f func(SPLWord) float64) (mean, minv float64, n int) {
	minv = math.Inf(1)
	for _, w := range r.Words {
		if w.Splice != splice {
			continue
		}
		v := f(w)
		mean += v
		minv = min(minv, v)
		n++
	}
	if n == 0 {
		return 0, 0, 0
	}
	return mean / float64(n), minv, n
}

// Score is the canonical SPL: mean conditional log-probability of the
// splice words.
func (r SPLResult) Score() float64 {
	m, _, _ := r.agg(true, func(w SPLWord) float64 { return w.Cond })
	return m
}

// PMIScore uses log P(w|x,·) − log P(w|·) at splice words: how much the
// source, rather than language-model fluency, supports each splice.
// Ablation.
func (r SPLResult) PMIScore() float64 {
	m, _, _ := r.agg(true, func(w SPLWord) float64 { return w.Cond - w.Uncond })
	return m
}

// MinScore is the worst splice. Ablation.
func (r SPLResult) MinScore() float64 {
	_, m, n := r.agg(true, func(w SPLWord) float64 { return w.Cond })
	if n == 0 {
		return 0
	}
	return m
}

// InsideScore is the mean over words INSIDE copied fragments — the part
// SPL deliberately discards; a control expected to track copying.
func (r SPLResult) InsideScore() float64 {
	m, _, _ := r.agg(false, func(w SPLWord) float64 { return w.Cond })
	return m
}

// splWord is a word span over the candidate's runes.
type splWord struct {
	start, end int
	lower      string
}

// splWords tokenises like cmd/confound (lowercase, the punctuation
// . , ! ? ; : ( ) " ' split off as words), keeping rune spans.
func splWords(text string) []splWord {
	const punct = `.,!?;:()"'`
	r := []rune(text)
	var out []splWord
	start := -1
	flush := func(i int) {
		if start >= 0 {
			out = append(out, splWord{start, i, strings.ToLower(string(r[start:i]))})
			start = -1
		}
	}
	for i, c := range r {
		switch {
		case unicode.IsSpace(c):
			flush(i)
		case strings.ContainsRune(punct, c):
			flush(i)
			out = append(out, splWord{i, i + 1, string(c)})
		default:
			if start < 0 {
				start = i
			}
		}
	}
	flush(len(r))
	return out
}

func splBigrams(text string) map[string]bool {
	ws := splWords(text)
	set := make(map[string]bool, len(ws))
	for i := 1; i < len(ws); i++ {
		set[ws[i-1].lower+" "+ws[i].lower] = true
	}
	return set
}

type splRequest struct {
	Source     string   `json:"source"`
	Candidates []string `json:"candidates"`
}

type splResponse struct {
	Cond    [][]float64 `json:"cond"`
	Uncond  [][]float64 `json:"uncond"`
	Offsets [][][2]int  `json:"offsets"`
	Error   string      `json:"error,omitempty"`
}

// ScoreArticle scores the rendered candidates of one article in a
// single request.
func (m *SPL) ScoreArticle(ctx context.Context, source string, candidates []string) ([]SPLResult, error) {
	var res splResponse
	if err := m.Server.postJSON(ctx, "/tokenlogprobs", splRequest{Source: source, Candidates: candidates}, &res); err != nil {
		return nil, fmt.Errorf("spl: %w", err)
	}
	if res.Error != "" {
		return nil, fmt.Errorf("spl: %s", res.Error)
	}
	if len(res.Cond) != len(candidates) || len(res.Offsets) != len(candidates) {
		return nil, fmt.Errorf("spl: expected %d candidates, got %d", len(candidates), len(res.Cond))
	}

	src := splBigrams(source)
	out := make([]SPLResult, len(candidates))
	for c, text := range candidates {
		ws := splWords(text)
		words := make([]SPLWord, len(ws))
		for i, w := range ws {
			words[i] = SPLWord{Word: w.lower, Splice: i == 0 || !src[ws[i-1].lower+" "+w.lower]}
		}
		runes := []rune(text)
		offs := res.Offsets[c]
		n := min(len(offs), len(res.Cond[c]), len(res.Uncond[c]))
		wi := 0
		for t := 0; t < n; t++ {
			// Anchor the token at its first non-space rune; a pure-space
			// token belongs to the word that follows it.
			p := offs[t][0]
			for p < offs[t][1] && p < len(runes) && unicode.IsSpace(runes[p]) {
				p++
			}
			for wi < len(ws) && ws[wi].end <= p {
				wi++
			}
			if wi == len(ws) {
				break
			}
			words[wi].Cond += res.Cond[c][t]
			words[wi].Uncond += res.Uncond[c][t]
		}
		out[c] = SPLResult{Words: words}
	}
	return out, nil
}
