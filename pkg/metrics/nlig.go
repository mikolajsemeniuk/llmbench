package metrics

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

// NLIG — NLI grounding. Each summary sentence is checked for
// entailment against every source window (single sentences and pairs of
// adjacent sentences) with an off-the-shelf NLI cross-encoder
// (DeBERTa-v3-large MNLI/FEVER/ANLI/LingNLI/WANLI, served at /nli):
//
//	g(c_j) = max_w  P(entail | w, c_j) − P(contradict | w, c_j)
//	score  = mean_j g(c_j)
//
// This is the SummaC-ZS construction (Laban et al., TACL 2022) and is
// not claimed as new; it is here as the consistency component answering
// "cosine is not entailment" (reviewers #4, #6), and because entailment
// holds for paraphrase, so unlike likelihood it should not pay a bonus
// for verbatim copying.
type NLIG struct {
	Server *ModelServer

	// Window is the largest number of adjacent source sentences joined
	// into one premise (1 = sentences only).
	Window int

	// TopK, when > 0, scores each summary sentence only against the K
	// premises with the highest word overlap (share of the sentence's
	// lowercase words found in the premise) instead of every premise —
	// a cost cut from O(|premises|) to O(K) NLI calls per sentence.
	TopK int
}

func NewNLIG(host string) *NLIG {
	return &NLIG{Server: NewModelServer(host), Window: 2}
}

// NLISentence is the best evidence found for one summary sentence.
type NLISentence struct {
	MaxEC  float64 `json:"max_ec"`  // max_w (entail − contradict)
	MaxE   float64 `json:"max_e"`   // max_w entail
	MaxC   float64 `json:"max_c"`   // max_w contradict
	ArgWin int     `json:"arg_win"` // premise index of MaxEC
}

type NLIResult struct {
	Sents []NLISentence `json:"sents"`
}

// Score is the canonical SummaC-ZS aggregate: mean over summary
// sentences of the best entail-minus-contradict.
func (r NLIResult) Score() float64 {
	if len(r.Sents) == 0 {
		return 0
	}
	var s float64
	for _, x := range r.Sents {
		s += x.MaxEC
	}
	return s / float64(len(r.Sents))
}

// MinScore is the weakest-sentence variant (one unsupported sentence
// is enough to make a summary inconsistent). Ablation.
func (r NLIResult) MinScore() float64 {
	if len(r.Sents) == 0 {
		return 0
	}
	m := r.Sents[0].MaxEC
	for _, x := range r.Sents[1:] {
		m = min(m, x.MaxEC)
	}
	return m
}

// MaxContradiction is the strongest contradiction any source window
// raises against any summary sentence (negated, higher = better).
// Ablation.
func (r NLIResult) MaxContradiction() float64 {
	var m float64
	for _, x := range r.Sents {
		m = max(m, x.MaxC)
	}
	return -m
}

// Premises returns the source windows used as NLI premises.
func (g *NLIG) Premises(source string) []string {
	sents := SplitSentences(source)
	var out []string
	for w := 1; w <= max(1, g.Window); w++ {
		for i := 0; i+w <= len(sents); i++ {
			out = append(out, strings.Join(sents[i:i+w], " "))
		}
	}
	return out
}

type nliRequest struct {
	Premises   []string `json:"premises"`
	Hypotheses []string `json:"hypotheses"`
}

type nliResponse struct {
	Probs [][][3]float64 `json:"probs"`
	Flat  [][3]float64   `json:"flat"`
	Error string         `json:"error,omitempty"`
}

type nliPairsRequest struct {
	Pairs [][2]string `json:"pairs"`
}

func nliWordSet(s string) map[string]bool {
	set := map[string]bool{}
	for _, w := range splWords(s) {
		if len(w.lower) > 2 {
			set[w.lower] = true
		}
	}
	return set
}

// topPremises returns the indices of the k premises covering the most of
// hypothesis h's words (ties: earlier premise first).
func topPremises(h map[string]bool, prem []map[string]bool, k int) []int {
	type sc struct {
		i int
		v int
	}
	all := make([]sc, len(prem))
	for i, p := range prem {
		for w := range h {
			if p[w] {
				all[i].v++
			}
		}
		all[i].i = i
	}
	sort.SliceStable(all, func(a, b int) bool { return all[a].v > all[b].v })
	out := make([]int, 0, k)
	for _, x := range all[:min(k, len(all))] {
		out = append(out, x.i)
	}
	return out
}

// ScoreArticle scores all rendered candidates of one article in a
// single request (every candidate sentence against every premise).
func (g *NLIG) ScoreArticle(ctx context.Context, source string, candidates []string) ([]NLIResult, error) {
	prem := g.Premises(source)
	var hyp []string
	owner := []int{}
	for i, c := range candidates {
		for _, s := range SplitSentences(c) {
			if len([]rune(s)) < 4 {
				continue
			}
			hyp = append(hyp, s)
			owner = append(owner, i)
		}
	}
	out := make([]NLIResult, len(candidates))
	if len(hyp) == 0 || len(prem) == 0 {
		return out, nil
	}

	if g.TopK > 0 {
		return g.scoreSparse(ctx, prem, hyp, owner, len(candidates))
	}

	var res nliResponse
	if err := g.Server.postJSON(ctx, "/nli", nliRequest{Premises: prem, Hypotheses: hyp}, &res); err != nil {
		return nil, fmt.Errorf("nlig: %w", err)
	}
	if res.Error != "" {
		return nil, fmt.Errorf("nlig: %s", res.Error)
	}
	if len(res.Probs) != len(hyp) {
		return nil, fmt.Errorf("nlig: expected %d rows, got %d", len(hyp), len(res.Probs))
	}
	for h, row := range res.Probs {
		wins := make([]int, len(row))
		for i := range wins {
			wins[i] = i
		}
		out[owner[h]].Sents = append(out[owner[h]].Sents, bestEvidence(row, wins))
	}
	return out, nil
}

func bestEvidence(row [][3]float64, wins []int) NLISentence {
	best := NLISentence{MaxEC: -2}
	for j, p := range row {
		if ec := p[0] - p[2]; ec > best.MaxEC {
			best.MaxEC, best.ArgWin = ec, wins[j]
		}
		best.MaxE = max(best.MaxE, p[0])
		best.MaxC = max(best.MaxC, p[2])
	}
	return best
}

// scoreSparse scores each hypothesis against its TopK premises only.
func (g *NLIG) scoreSparse(ctx context.Context, prem, hyp []string, owner []int, nCand int) ([]NLIResult, error) {
	pw := make([]map[string]bool, len(prem))
	for i, p := range prem {
		pw[i] = nliWordSet(p)
	}
	var pairs [][2]string
	sel := make([][]int, len(hyp))
	for h, x := range hyp {
		sel[h] = topPremises(nliWordSet(x), pw, g.TopK)
		for _, i := range sel[h] {
			pairs = append(pairs, [2]string{prem[i], x})
		}
	}
	var res nliResponse
	if err := g.Server.postJSON(ctx, "/nli", nliPairsRequest{Pairs: pairs}, &res); err != nil {
		return nil, fmt.Errorf("nlig: %w", err)
	}
	if res.Error != "" {
		return nil, fmt.Errorf("nlig: %s", res.Error)
	}
	if len(res.Flat) != len(pairs) {
		return nil, fmt.Errorf("nlig: expected %d pairs, got %d", len(pairs), len(res.Flat))
	}
	out := make([]NLIResult, nCand)
	j := 0
	for h := range hyp {
		row := res.Flat[j : j+len(sel[h])]
		j += len(sel[h])
		out[owner[h]].Sents = append(out[owner[h]].Sents, bestEvidence(row, sel[h]))
	}
	return out, nil
}
