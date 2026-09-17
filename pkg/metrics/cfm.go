package metrics

import (
	"math/rand/v2"
	"regexp"
	"strings"
	"unicode"
)

// Counterfactual perturbation families used by the counterfactual-margin
// metric (cmd/cfm).
//
// The metric scores a candidate summary not by how similar it is to the
// source but by how much a frozen language model prefers the candidate
// over minimally corrupted versions of itself. The point of that design
// is decorrelation: every perturbation below either permutes the
// candidate's own tokens or substitutes a token drawn out of the source,
// so the corrupted candidate has (almost exactly) the same verbatim
// overlap with the source as the original. A metric that only counts
// copied n-grams therefore assigns the same value to both and its margin
// is identically zero. Extractiveness cannot leak into a margin the way
// it leaks into a similarity (see cmd/confound).
//
// Each family targets one SummEval dimension:
//
//	Permute       sentence order          -> coherence
//	Scramble      local word order        -> fluency
//	SwapEntities  named entities/numbers  -> consistency
//
// Relevance needs no candidate-side corruption: the counterfactual is on
// the source side (score the candidate against a different article), so
// it lives in cmd/cfm rather than here.

// Permute returns up to m distinct non-identity permutations of the
// candidate's sentences, rejoined into text. Fewer than m are returned
// when the sentence count admits fewer permutations.
func Permute(sents []string, m int, rng *rand.Rand) []string {
	if len(sents) < 2 {
		return nil
	}
	seen := map[string]bool{strings.Join(sents, " "): true}
	out := make([]string, 0, m)
	for attempt := 0; attempt < 50*m && len(out) < m; attempt++ {
		p := append([]string(nil), sents...)
		rng.Shuffle(len(p), func(i, j int) { p[i], p[j] = p[j], p[i] })
		joined := strings.Join(p, " ")
		if seen[joined] {
			continue
		}
		seen[joined] = true
		out = append(out, joined)
	}
	return out
}

// Scramble returns m local word-order corruptions: within each sentence,
// ceil(len/8) adjacent word pairs are transposed. Sentence order and the
// token multiset are preserved, so the corruption is local and lexically
// invisible to any bag-of-n-grams source-overlap counter beyond the
// bigrams it breaks -- which is why cmd/cfm reports the copy rate of the
// perturbed candidates as a diagnostic.
func Scramble(sents []string, m int, rng *rand.Rand) []string {
	out := make([]string, 0, m)
	for k := 0; k < m; k++ {
		perturbed := make([]string, len(sents))
		changed := false
		for i, s := range sents {
			w := strings.Fields(s)
			if len(w) < 4 {
				perturbed[i] = s
				continue
			}
			swaps := (len(w) + 7) / 8
			for range swaps {
				j := rng.IntN(len(w) - 1)
				w[j], w[j+1] = w[j+1], w[j]
				changed = true
			}
			perturbed[i] = strings.Join(w, " ")
		}
		if !changed {
			return out
		}
		out = append(out, strings.Join(perturbed, " "))
	}
	return out
}

var numberRe = regexp.MustCompile(`^[0-9][0-9,.%$]*$`)

// EntityPool collects substitution candidates out of the source: proper
// nouns (capitalised tokens not sentence-initial) and numerals.
type EntityPool struct {
	Names   []string
	Numbers []string
}

// NewEntityPool extracts the substitution pool from a source document.
func NewEntityPool(doc string) EntityPool {
	var p EntityPool
	nameSeen, numSeen := map[string]bool{}, map[string]bool{}
	for _, sent := range SplitSentences(doc) {
		for i, tok := range strings.Fields(sent) {
			t := strings.Trim(tok, ".,;:!?\"'()")
			if t == "" {
				continue
			}
			switch {
			case numberRe.MatchString(t):
				if !numSeen[t] {
					numSeen[t] = true
					p.Numbers = append(p.Numbers, t)
				}
			case i > 0 && unicode.IsUpper([]rune(t)[0]) && len([]rune(t)) > 2:
				if !nameSeen[t] {
					nameSeen[t] = true
					p.Names = append(p.Names, t)
				}
			}
		}
	}
	return p
}

// SwapEntities returns m corruptions in which one to three entity
// mentions in the candidate are each replaced by a different entity of
// the same class (name for name, numeral for numeral) drawn from the
// source. The substitute comes from the source, so the corruption stays
// on-topic and preserves lexical overlap with the source: what it
// destroys is only the factual alignment between candidate and source.
// Returns nil when the candidate has no substitutable mention, which is
// reported by cmd/cfm as a coverage statistic rather than silently
// scored as zero.
func SwapEntities(candidate string, pool EntityPool, m int, rng *rand.Rand) []string {
	words := strings.Fields(candidate)
	type slot struct {
		idx     int
		core    string
		isNum   bool
		prefix  string
		postfix string
	}
	var slots []slot
	for i, w := range words {
		core := strings.Trim(w, ".,;:!?\"'()")
		if core == "" {
			continue
		}
		cut := strings.Index(w, core)
		pre, post := w[:cut], w[cut+len(core):]
		switch {
		case numberRe.MatchString(core):
			if len(pool.Numbers) > 1 {
				slots = append(slots, slot{i, core, true, pre, post})
			}
		case i > 0 && unicode.IsUpper([]rune(core)[0]) && len([]rune(core)) > 2:
			if len(pool.Names) > 1 {
				slots = append(slots, slot{i, core, false, pre, post})
			}
		}
	}
	if len(slots) == 0 {
		return nil
	}

	pick := func(class []string, avoid string) string {
		for range 20 {
			c := class[rng.IntN(len(class))]
			if c != avoid {
				return c
			}
		}
		return ""
	}

	out := make([]string, 0, m)
	for range m {
		w := append([]string(nil), words...)
		k := 1 + rng.IntN(min(3, len(slots)))
		rng.Shuffle(len(slots), func(i, j int) { slots[i], slots[j] = slots[j], slots[i] })
		changed := false
		for _, s := range slots[:k] {
			class := pool.Names
			if s.isNum {
				class = pool.Numbers
			}
			sub := pick(class, s.core)
			if sub == "" {
				continue
			}
			w[s.idx] = s.prefix + sub + s.postfix
			changed = true
		}
		if changed {
			out = append(out, strings.Join(w, " "))
		}
	}
	return out
}
