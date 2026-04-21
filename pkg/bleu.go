package llmbench

import (
	"math"
	"strings"
)

func BLEU(reference, candidate string) float64 {
	refTokens := tokenize(reference)
	candTokens := tokenize(candidate)

	if len(candTokens) == 0 {
		return 0
	}

	bp := 1.0
	if len(candTokens) < len(refTokens) {
		bp = math.Exp(1.0 - float64(len(refTokens))/float64(len(candTokens)))
	}

	maxN := 4
	maxN = min(maxN, len(candTokens))

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
				continue
			}

			clipped += cnt
		}

		total := len(candTokens) - n + 1
		if total <= 0 {
			continue
		}

		precision := float64(clipped) / float64(total)
		if precision == 0 {
			precision = 1e-10
		}

		logAvg += weight * math.Log(precision)
	}

	return bp * math.Exp(logAvg)
}

func tokenize(s string) []string {
	s = strings.ToLower(s)
	for _, p := range []string{".", ",", "!", "?", ";", ":", "(", ")", "\"", "'"} {
		s = strings.ReplaceAll(s, p, " "+p+" ")
	}
	return strings.Fields(s)
}

func ngrams(tokens []string, n int) map[string]int {
	m := make(map[string]int)
	for i := 0; i <= len(tokens)-n; i++ {
		key := strings.Join(tokens[i:i+n], " ")
		m[key]++
	}
	return m
}
