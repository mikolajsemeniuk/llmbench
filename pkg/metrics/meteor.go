package metrics

import (
	"math"
	"strings"
)

// METEOR computes the METEOR score between a reference and candidate.
// It uses exact matching and Porter stemming with a chunk penalty.
//
// Parameters follow the original paper (Banerjee & Lavie, 2005):
//
//	α = 0.9  (harmonic mean weight, favoring recall)
//	β = 3.0  (chunk penalty exponent)
//	γ = 0.5  (chunk penalty weight)
func METEOR(reference, candidate string) float64 {
	refTokens := tokenize(reference)
	candTokens := tokenize(candidate)

	if len(refTokens) == 0 || len(candTokens) == 0 {
		return 0
	}

	// Stage 1: exact match, Stage 2: stem match.
	_, candMatched := align(refTokens, candTokens)

	matches := 0
	for _, v := range candMatched {
		if v {
			matches++
		}
	}
	if matches == 0 {
		return 0
	}

	precision := float64(matches) / float64(len(candTokens))
	recall := float64(matches) / float64(len(refTokens))

	// Harmonic mean weighted toward recall (α = 0.9).
	const alpha = 0.9
	fmean := (precision * recall) / (alpha*recall + (1-alpha)*precision)

	// Chunk penalty.
	chunks := countChunks(candMatched)
	const beta = 3.0
	const gamma = 0.5
	penalty := gamma * math.Pow(float64(chunks)/float64(matches), beta)

	return fmean * (1 - penalty)
}

// align finds the best unigram alignment between reference and candidate
// using two stages: exact match, then stem match.
func align(ref, cand []string) (refMatched, candMatched []bool) {
	refMatched = make([]bool, len(ref))
	candMatched = make([]bool, len(cand))

	// Stage 1: exact match.
	for ci, ct := range cand {
		if candMatched[ci] {
			continue
		}
		for ri, rt := range ref {
			if refMatched[ri] {
				continue
			}
			if ct == rt {
				candMatched[ci] = true
				refMatched[ri] = true
				break
			}
		}
	}

	// Stage 2: stem match.
	for ci, ct := range cand {
		if candMatched[ci] {
			continue
		}
		cs := stem(ct)
		for ri, rt := range ref {
			if refMatched[ri] {
				continue
			}
			if cs == stem(rt) {
				candMatched[ci] = true
				refMatched[ri] = true
				break
			}
		}
	}

	return
}

func countChunks(matched []bool) int {
	chunks := 0
	inChunk := false
	for _, m := range matched {
		if m && !inChunk {
			chunks++
			inChunk = true
		} else if !m {
			inChunk = false
		}
	}
	return chunks
}

func stem(word string) string {
	w := strings.ToLower(word)
	if len(w) <= 3 {
		return w
	}

	switch {
	case strings.HasSuffix(w, "sses"):
		w = w[:len(w)-2]
	case strings.HasSuffix(w, "ies"):
		w = w[:len(w)-2]
	case strings.HasSuffix(w, "ss"):
	case strings.HasSuffix(w, "s"):
		w = w[:len(w)-1]
	}

	switch {
	case strings.HasSuffix(w, "eed"):
		if measure(w[:len(w)-3]) > 0 {
			w = w[:len(w)-1]
		}
	case strings.HasSuffix(w, "ed"):
		stem := w[:len(w)-2]
		if hasVowel(stem) {
			w = fixStem(stem)
		}
	case strings.HasSuffix(w, "ing"):
		stem := w[:len(w)-3]
		if hasVowel(stem) {
			w = fixStem(stem)
		}
	}

	if strings.HasSuffix(w, "y") && len(w) > 3 && hasVowel(w[:len(w)-1]) {
		w = w[:len(w)-1] + "i"
	}

	suffixes := []struct{ from, to string }{
		{"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"},
		{"anci", "ance"}, {"izer", "ize"}, {"isation", "ize"},
		{"ization", "ize"}, {"ation", "ate"}, {"ator", "ate"},
		{"alism", "al"}, {"iveness", "ive"}, {"fulness", "ful"},
		{"ousness", "ous"}, {"aliti", "al"}, {"iviti", "ive"},
		{"biliti", "ble"},
	}
	for _, sf := range suffixes {
		if strings.HasSuffix(w, sf.from) {
			base := w[:len(w)-len(sf.from)]
			if measure(base) > 0 {
				w = base + sf.to
			}
			break
		}
	}

	// Step 4: remove common endings.
	endings := []string{
		"al", "ance", "ence", "er", "ic", "able", "ible",
		"ant", "ement", "ment", "ent", "ism", "ate", "iti",
		"ous", "ive", "ize",
	}
	for _, e := range endings {
		if strings.HasSuffix(w, e) {
			base := w[:len(w)-len(e)]
			if measure(base) > 1 {
				w = base
			}
			break
		}
	}

	return w
}

func isVowel(b byte) bool {
	return b == 'a' || b == 'e' || b == 'i' || b == 'o' || b == 'u'
}

func hasVowel(s string) bool {
	for i := 0; i < len(s); i++ {
		if isVowel(s[i]) {
			return true
		}
	}
	return false
}

func measure(s string) int {
	if len(s) == 0 {
		return 0
	}
	m := 0
	inVowel := false
	for i := 0; i < len(s); i++ {
		v := isVowel(s[i])
		if inVowel && !v {
			m++
		}
		inVowel = v
	}
	return m
}

func fixStem(s string) string {
	switch {
	case strings.HasSuffix(s, "at"), strings.HasSuffix(s, "bl"), strings.HasSuffix(s, "iz"):
		return s + "e"
	}
	if len(s) >= 2 && s[len(s)-1] == s[len(s)-2] {
		ch := s[len(s)-1]
		if ch != 'l' && ch != 's' && ch != 'z' {
			return s[:len(s)-1]
		}
	}
	return s
}
