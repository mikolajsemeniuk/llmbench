package llmbench

import (
	"math"
	"strings"
)

// BLEU computes a BLEU-4 score between a reference and candidate text.
// It uses uniform weights across 1- through 4-grams with a brevity penalty.
// Returns a value in [0, 1].
func BLEU(reference, candidate string) float64 {
	refTokens := tokenize(reference)
	candTokens := tokenize(candidate)

	// Brevity penalty.
	bp := 1.0
	if len(candTokens) < len(refTokens) {
		bp = math.Exp(1.0 - float64(len(refTokens))/float64(len(candTokens)))
	}

	// Modified n-gram precision for n=1..4 with smoothing (method 1:
	// add 1 to numerator and denominator for n>1 when count is zero).
	maxN := 4
	if len(candTokens) < maxN {
		maxN = len(candTokens)
	}
	if maxN == 0 {
		return 0
	}

	logAvg := 0.0
	weight := 1.0 / float64(maxN)

	for n := 1; n <= maxN; n++ {
		refNgrams := ngrams(refTokens, n)
		candNgrams := ngrams(candTokens, n)

		clipped := 0
		for ng, cnt := range candNgrams {
			refCnt := refNgrams[ng]
			if refCnt < cnt {
				clipped += refCnt
			} else {
				clipped += cnt
			}
		}

		total := len(candTokens) - n + 1
		if total <= 0 {
			continue
		}

		precision := float64(clipped) / float64(total)

		// Smoothing: if precision is 0 for any n, use epsilon to avoid
		// log(0). This is a simplified variant of Chen & Cherry smoothing.
		if precision == 0 {
			precision = 1e-10
		}

		logAvg += weight * math.Log(precision)
	}

	return bp * math.Exp(logAvg)
}

// tokenize lowercases and splits on whitespace and common punctuation.
func tokenize(s string) []string {
	s = strings.ToLower(s)
	// Insert space before/after punctuation so they become separate tokens.
	for _, p := range []string{".", ",", "!", "?", ";", ":", "(", ")", "\"", "'"} {
		s = strings.ReplaceAll(s, p, " "+p+" ")
	}
	fields := strings.Fields(s)
	return fields
}

// ngrams returns a frequency map of n-grams for the given tokens.
func ngrams(tokens []string, n int) map[string]int {
	m := make(map[string]int)
	for i := 0; i <= len(tokens)-n; i++ {
		key := strings.Join(tokens[i:i+n], " ")
		m[key]++
	}
	return m
}
