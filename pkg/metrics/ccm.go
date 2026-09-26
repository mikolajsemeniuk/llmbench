package metrics

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"regexp"
	"slices"
	"strings"
	"unicode"
)

// CCM — Copy-invariant Contrastive Margin. Reference-free summary-quality
// metric. For source x, candidate y and K automatically generated
// candidate-side perturbations y'_1..y'_K:
//
//	score = log P(y | x) − mean_k log P(y'_k | x)
//
// under a small causal LM (Qwen2.5-1.5B by default, served by
// cmd/modelsrv /ccm). The perturbations are local, meaning-changing
// edits of y itself — swap a named entity for another one from the
// source, change a number, flip a negation, reorder two sentences — so
// y and every y'_k share almost the same extractiveness (copy rate).
// Whatever makes log P(y|x) high merely because y copies the source
// also makes log P(y'_k|x) high, and cancels in the difference; what
// is left is how much more plausible the source makes the candidate
// than its minimally wrong neighbours.
//
// Closest prior art: DeltaScore (Xie et al., Findings EMNLP 2023) uses
// log p(s|c) − log p(s'|c) with a single perturbation for story
// evaluation; DetectGPT / Fast-DetectGPT use the mean over K
// perturbations (unconditional) for machine-text detection. CCM applies
// the K-perturbation, source-conditioned form to summarisation with
// factual perturbation families aimed at copy-invariance.
//
// The canonical score is the conditional margin above. The per-call
// output also carries everything needed for the ablations: the plain
// likelihood (no contrast), the unconditional likelihoods (to form a
// source-attributable margin), and per-family margins.
type CCM struct {
	Server *ModelServer

	// K is the number of perturbations per candidate. Families are
	// used round-robin over those applicable to the candidate.
	K int

	// Seed makes perturbation sampling deterministic; the per-candidate
	// stream is additionally keyed by the candidate's sample ID.
	Seed uint64
}

// CCM perturbation families.
const (
	CCMEntity   = "entity"
	CCMNumber   = "number"
	CCMNegation = "negation"
	CCMOrder    = "order"
)

// CCMFamilies lists the perturbation families in round-robin order.
var CCMFamilies = []string{CCMEntity, CCMNumber, CCMNegation, CCMOrder}

type CCMPerturbation struct {
	Family string
	Text   string // detokenised, ready for the LM
}

// CCMCandidate is one candidate plus its perturbations, as sent to the
// model server.
type CCMCandidate struct {
	ID            string
	Text          string // detokenised candidate
	Perturbations []CCMPerturbation
}

// CCMResult holds the LM log-likelihoods for one candidate.
type CCMResult struct {
	Cond, Uncond         float64 // log P(y|x), log P(y)
	Tokens               int
	PertCond, PertUncond []float64
	Families             []string
}

func NewCCM(host string) *CCM {
	return &CCM{Server: NewModelServer(host), K: 8, Seed: 42}
}

// RenderCandidate turns a SummEval candidate (PTB-tokenised lowercase)
// into natural text: proper-noun casing restored from the source, then
// detokenised. It is exactly how Prepare renders the unperturbed
// candidate, exported for the other source-conditioned metrics.
func RenderCandidate(source, candidate string) string {
	_, casing := sourceEntities(source)
	toks := strings.Fields(candidate)
	for i, t := range toks {
		if c, ok := casing[t]; ok {
			toks[i] = c
		}
	}
	return DetokenizePTB(strings.Join(toks, " "))
}

// Prepare detokenises the candidate and generates its perturbations.
// source is the original (cased) article; candidate is the SummEval
// machine summary in PTB-tokenised lowercase form.
func (m *CCM) Prepare(id, source, candidate string) CCMCandidate {
	h := fnv.New64a()
	h.Write([]byte(id))
	rng := rand.New(rand.NewPCG(m.Seed, h.Sum64()))

	toks := strings.Fields(candidate)
	ents, casing := sourceEntities(source)
	nums := sourceNumbers(source)

	// SummEval candidates are lowercased; restore proper-noun casing
	// from the source so the LM scores natural text. Applied
	// identically to the candidate and every perturbation.
	render := func(ts []string) string {
		out := make([]string, len(ts))
		for i, t := range ts {
			if c, ok := casing[t]; ok {
				t = c
			}
			out[i] = t
		}
		return DetokenizePTB(strings.Join(out, " "))
	}

	orig := render(toks)
	seen := map[string]bool{orig: true}
	var out []CCMPerturbation

	fams := slices.Clone(CCMFamilies)
	for attempt := 0; len(out) < m.K && attempt < 8*m.K && len(fams) > 0; attempt++ {
		f := fams[attempt%len(fams)]
		var p []string
		switch f {
		case CCMEntity:
			p = perturbEntity(toks, ents, rng)
		case CCMNumber:
			p = perturbNumber(toks, nums, rng)
		case CCMNegation:
			p = perturbNegation(toks, rng)
		case CCMOrder:
			p = perturbOrder(toks, rng)
		}
		if p == nil {
			// Family not applicable to this candidate: drop it.
			fams = slices.DeleteFunc(fams, func(x string) bool { return x == f })
			attempt--
			continue
		}
		txt := render(p)
		if seen[txt] {
			continue
		}
		seen[txt] = true
		out = append(out, CCMPerturbation{Family: f, Text: txt})
	}
	return CCMCandidate{ID: id, Text: orig, Perturbations: out}
}

type ccmRequest struct {
	Source     string   `json:"source"`
	Candidates []string `json:"candidates"`
}

type ccmResponse struct {
	Cond   []float64 `json:"cond"`
	Uncond []float64 `json:"uncond"`
	Tokens []int     `json:"tokens"`
	Error  string    `json:"error,omitempty"`
}

// ScoreArticle scores every candidate of one article in a single
// request, so the model server encodes the source once.
func (m *CCM) ScoreArticle(ctx context.Context, source string, cands []CCMCandidate) ([]CCMResult, error) {
	var flat []string
	for _, c := range cands {
		flat = append(flat, c.Text)
		for _, p := range c.Perturbations {
			flat = append(flat, p.Text)
		}
	}

	var res ccmResponse
	if err := m.Server.postJSON(ctx, "/ccm", ccmRequest{Source: source, Candidates: flat}, &res); err != nil {
		return nil, fmt.Errorf("ccm: %w", err)
	}
	if res.Error != "" {
		return nil, fmt.Errorf("ccm: %s", res.Error)
	}
	if len(res.Cond) != len(flat) || len(res.Uncond) != len(flat) {
		return nil, fmt.Errorf("ccm: expected %d scores, got %d/%d", len(flat), len(res.Cond), len(res.Uncond))
	}

	out := make([]CCMResult, len(cands))
	j := 0
	for i, c := range cands {
		r := CCMResult{Cond: res.Cond[j], Uncond: res.Uncond[j], Tokens: res.Tokens[j]}
		j++
		for _, p := range c.Perturbations {
			r.PertCond = append(r.PertCond, res.Cond[j])
			r.PertUncond = append(r.PertUncond, res.Uncond[j])
			r.Families = append(r.Families, p.Family)
			j++
		}
		out[i] = r
	}
	return out, nil
}

// Margin is the canonical CCM score: log P(y|x) − mean_k log P(y'_k|x).
// Zero when no perturbation could be generated.
func (r CCMResult) Margin() float64 {
	if len(r.PertCond) == 0 {
		return 0
	}
	return r.Cond - mean(r.PertCond)
}

// ZMargin divides the margin by the spread of the perturbed
// likelihoods (Fast-DetectGPT-style normalisation). Ablation.
func (r CCMResult) ZMargin() float64 {
	if len(r.PertCond) < 2 {
		return r.Margin()
	}
	return r.Margin() / (stddev(r.PertCond) + 1e-3)
}

// PMIMargin is the source-attributable margin: the conditional margin
// minus the same margin computed without the source, removing the part
// of the contrast that the LM's prior alone explains (e.g. a negation
// that is simply ungrammatical). Ablation.
func (r CCMResult) PMIMargin() float64 {
	if len(r.PertCond) == 0 {
		return 0
	}
	d := make([]float64, len(r.PertCond))
	for i := range d {
		d[i] = r.PertCond[i] - r.PertUncond[i]
	}
	return (r.Cond - r.Uncond) - mean(d)
}

// FamilyMargin is the margin restricted to one perturbation family;
// ok is false when the candidate has no perturbation of that family.
func (r CCMResult) FamilyMargin(family string) (float64, bool) {
	var xs []float64
	for i, f := range r.Families {
		if f == family {
			xs = append(xs, r.PertCond[i])
		}
	}
	if len(xs) == 0 {
		return 0, false
	}
	return r.Cond - mean(xs), true
}

// LogPPerToken is the uncontrasted, length-normalised likelihood
// log P(y|x)/|y| — the BARTScore(s→h)-style baseline CCM is ablated
// against.
func (r CCMResult) LogPPerToken() float64 {
	if r.Tokens == 0 {
		return 0
	}
	return r.Cond / float64(r.Tokens)
}

// ── Per-dimension scores (CCM-D) ───────────────────────────────────────
//
// The single margin above turned out NOT to be copy-invariant (an LM
// that has the source in context predicts copied spans near-certainly,
// so perturbing a copied span costs more than perturbing a paraphrase).
// CCM-D instead assigns each SummEval dimension one signal from the
// SAME forward passes. The assignment was selected on the development
// half (first 50 articles) from a fixed pool of seven signals by the
// copy-partialled Spearman rho of that dimension, and verified on the
// held-out last 50 (see README, "CCM-D"):
//
//	coherence   z-normalised margin over all perturbation families
//	consistency log P(y|x) / |y|
//	fluency     log P(y|x)
//	relevance   log P(y|x)/|y| / σ₁ + margin(order) / σ₂
//
// σ₁, σ₂ are the development-split standard deviations of the two
// relevance signals; they only fix the relative weight of the sum.
const (
	ccmdRelSigmaLogP  = 0.4684
	ccmdRelSigmaOrder = 6.5140
)

// OrderMargin is the margin against sentence-swap perturbations only;
// 0 (no evidence) for single-sentence candidates.
func (r CCMResult) OrderMargin() float64 {
	m, _ := r.FamilyMargin(CCMOrder)
	return m
}

// CoherenceScore is the z-normalised margin; 0 (no evidence) when fewer
// than two perturbations exist (15 of 1600 SummEval candidates), which
// is the definition the dev selection was run with.
func (r CCMResult) CoherenceScore() float64 {
	if len(r.PertCond) < 2 {
		return 0
	}
	return r.ZMargin()
}

func (r CCMResult) ConsistencyScore() float64 { return r.LogPPerToken() }
func (r CCMResult) FluencyScore() float64     { return r.Cond }
func (r CCMResult) RelevanceScore() float64 {
	return r.LogPPerToken()/ccmdRelSigmaLogP + r.OrderMargin()/ccmdRelSigmaOrder
}

func mean(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func stddev(xs []float64) float64 {
	m := mean(xs)
	var s float64
	for _, x := range xs {
		s += (x - m) * (x - m)
	}
	return math.Sqrt(s / float64(len(xs)))
}

// ── Perturbations ──────────────────────────────────────────────────────
//
// All perturbations operate on the PTB-tokenised lowercase candidate and
// change one local span, so the perturbed text keeps the candidate's
// length and nearly all of its source n-grams.

var ccmStopwords = map[string]bool{
	"the": true, "a": true, "an": true, "he": true, "she": true, "it": true,
	"they": true, "we": true, "i": true, "you": true, "his": true, "her": true,
	"their": true, "this": true, "that": true, "these": true, "those": true,
	"in": true, "on": true, "at": true, "but": true, "and": true, "or": true,
	"if": true, "when": true, "after": true, "before": true, "as": true,
	"there": true, "what": true, "who": true, "mr": true, "mrs": true, "ms": true,
	"monday": true, "tuesday": true, "wednesday": true, "thursday": true,
	"friday": true, "saturday": true, "sunday": true, "cnn": true,
}

var capWord = regexp.MustCompile(`[A-Za-z][A-Za-z\-]*`)

// sourceEntities returns lowercase tokens that appear capitalised in
// the source away from a sentence start — a cheap proper-noun proxy
// that needs no tagger and works on the lowercased candidate — plus a
// map from each such token to its cased source form.
func sourceEntities(source string) ([]string, map[string]string) {
	set := map[string]bool{}
	casing := map[string]string{}
	for _, sent := range SplitSentences(source) {
		words := capWord.FindAllString(sent, -1)
		for i, w := range words {
			if i == 0 || len(w) < 3 || !unicode.IsUpper(rune(w[0])) {
				continue
			}
			lw := strings.ToLower(w)
			if !ccmStopwords[lw] {
				set[lw] = true
				casing[lw] = w
			}
		}
	}
	out := make([]string, 0, len(set))
	for w := range set {
		out = append(out, w)
	}
	slices.Sort(out)
	return out, casing
}

func hasDigit(s string) bool {
	return strings.ContainsFunc(s, unicode.IsDigit)
}

func sourceNumbers(source string) []string {
	set := map[string]bool{}
	for _, t := range strings.Fields(strings.ToLower(source)) {
		t = strings.Trim(t, ".,;:!?()\"'$%")
		if hasDigit(t) {
			set[t] = true
		}
	}
	out := make([]string, 0, len(set))
	for w := range set {
		out = append(out, w)
	}
	slices.Sort(out)
	return out
}

// perturbEntity replaces one entity mention with a different source
// entity that the candidate does not already mention.
func perturbEntity(toks, ents []string, rng *rand.Rand) []string {
	entSet := map[string]bool{}
	for _, e := range ents {
		entSet[e] = true
	}
	var pos []int
	for i, t := range toks {
		if entSet[t] {
			pos = append(pos, i)
		}
	}
	if len(pos) == 0 {
		return nil
	}
	inCand := map[string]bool{}
	for _, t := range toks {
		inCand[t] = true
	}
	var repl []string
	for _, e := range ents {
		if !inCand[e] {
			repl = append(repl, e)
		}
	}
	if len(repl) == 0 {
		return nil
	}
	out := slices.Clone(toks)
	out[pos[rng.IntN(len(pos))]] = repl[rng.IntN(len(repl))]
	return out
}

// perturbNumber replaces one numeric token with a different number,
// preferring numbers that occur in the source.
func perturbNumber(toks, nums []string, rng *rand.Rand) []string {
	var pos []int
	for i, t := range toks {
		if hasDigit(t) {
			pos = append(pos, i)
		}
	}
	if len(pos) == 0 {
		return nil
	}
	out := slices.Clone(toks)
	i := pos[rng.IntN(len(pos))]
	var repl []string
	for _, n := range nums {
		if n != toks[i] {
			repl = append(repl, n)
		}
	}
	if len(repl) > 0 && rng.IntN(2) == 0 {
		out[i] = repl[rng.IntN(len(repl))]
		return out
	}
	// Change one digit.
	r := []rune(toks[i])
	var dpos []int
	for j, c := range r {
		if unicode.IsDigit(c) {
			dpos = append(dpos, j)
		}
	}
	j := dpos[rng.IntN(len(dpos))]
	d := int(r[j]-'0') + 1 + rng.IntN(8)
	r[j] = rune('0' + d%10)
	if string(r) == toks[i] {
		return nil
	}
	out[i] = string(r)
	return out
}

var ccmAux = map[string]bool{
	"is": true, "was": true, "are": true, "were": true, "has": true, "have": true,
	"had": true, "will": true, "would": true, "can": true, "could": true,
	"did": true, "does": true, "do": true, "should": true, "must": true,
	"may": true, "might": true, "been": false,
}

// perturbNegation removes an existing negation after an auxiliary, or
// inserts one.
func perturbNegation(toks []string, rng *rand.Rand) []string {
	var pos []int
	for i, t := range toks {
		if ccmAux[t] {
			pos = append(pos, i)
		}
	}
	if len(pos) == 0 {
		return nil
	}
	i := pos[rng.IntN(len(pos))]
	if i+1 < len(toks) && (toks[i+1] == "not" || toks[i+1] == "n't") {
		return slices.Delete(slices.Clone(toks), i+1, i+2)
	}
	return slices.Insert(slices.Clone(toks), i+1, "not")
}

// perturbOrder swaps two sentences of the candidate.
func perturbOrder(toks []string, rng *rand.Rand) []string {
	var sents [][]string
	var cur []string
	for _, t := range toks {
		cur = append(cur, t)
		if t == "." || t == "!" || t == "?" {
			sents = append(sents, cur)
			cur = nil
		}
	}
	if len(cur) > 0 {
		sents = append(sents, cur)
	}
	if len(sents) < 2 {
		return nil
	}
	i := rng.IntN(len(sents))
	j := rng.IntN(len(sents) - 1)
	if j >= i {
		j++
	}
	sents[i], sents[j] = sents[j], sents[i]
	var out []string
	for _, s := range sents {
		out = append(out, s...)
	}
	return out
}

// ── Detokenisation ─────────────────────────────────────────────────────

var ptbReplacer = strings.NewReplacer(
	"-lrb-", "(", "-rrb-", ")", "-lsb-", "[", "-rsb-", "]",
	"``", "\"", "''", "\"",
)

var (
	ptbNoSpaceBefore = map[string]bool{
		".": true, ",": true, "!": true, "?": true, ";": true, ":": true,
		")": true, "]": true, "%": true, "'s": true, "n't": true, "'re": true,
		"'ve": true, "'ll": true, "'d": true, "'m": true, "'": true,
	}
	ptbNoSpaceAfter = map[string]bool{"(": true, "[": true, "$": true}
)

// DetokenizePTB turns SummEval's PTB-tokenised lowercase text back into
// ordinary prose (joined punctuation and clitics, sentence-initial
// capitals) so the LM scores text in the form it was trained on. The
// same function is applied to the candidate and to its perturbations.
func DetokenizePTB(s string) string {
	toks := strings.Fields(ptbReplacer.Replace(s))
	var b strings.Builder
	capNext := true
	for i, t := range toks {
		if i > 0 && !ptbNoSpaceBefore[t] && !ptbNoSpaceAfter[toks[i-1]] {
			b.WriteByte(' ')
		}
		if capNext && t != "" && unicode.IsLetter(rune(t[0])) {
			r := []rune(t)
			r[0] = unicode.ToUpper(r[0])
			t = string(r)
			capNext = false
		}
		b.WriteString(t)
		if t == "." || t == "!" || t == "?" {
			capNext = true
		}
	}
	return b.String()
}
