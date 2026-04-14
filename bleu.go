package llmbench

import (
	"math"
	"strings"
)

func BLEU(reference, candidate string) float64 {
	refTokens := tokenize(reference)
	candTokens := tokenize(candidate)

	bp := 1.0
	if len(candTokens) < len(refTokens) {
		bp = math.Exp(1.0 - float64(len(refTokens))/float64(len(candTokens)))
	}

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

		if precision == 0 {
			precision = 1e-10
		}

		logAvg += weight * math.Log(precision)
	}

	return bp * math.Exp(logAvg)
}

func MaxBLEU(references []string, candidate string) float64 {
	best := 0.0
	for _, ref := range references {
		if s := BLEU(ref, candidate); s > best {
			best = s
		}
	}
	return best
}

func tokenize(s string) []string {
	s = strings.ToLower(s)
	for _, p := range []string{".", ",", "!", "?", ";", ":", "(", ")", "\"", "'"} {
		s = strings.ReplaceAll(s, p, " "+p+" ")
	}

	fields := strings.Fields(s)
	return fields
}

func ngrams(tokens []string, n int) map[string]int {
	m := make(map[string]int)
	for i := 0; i <= len(tokens)-n; i++ {
		key := strings.Join(tokens[i:i+n], " ")
		m[key]++
	}
	return m
}

// BLEUScorer implements Scorer using BLEU-4.
type BLEUScorer struct{}

func NewBLEUScorer() *BLEUScorer { return &BLEUScorer{} }

func (s *BLEUScorer) Score(entries []Entry) (ScoreOutput, error) {
	var out ScoreOutput
	for _, e := range entries {
		for mi, mach := range e.MachineSummaries {
			out.Scores = append(out.Scores, MaxBLEU(e.HumanSummaries, mach))
			out.Relevance = append(out.Relevance, e.Relevance[mi])
			out.Coherence = append(out.Coherence, e.Coherence[mi])
			out.Fluency = append(out.Fluency, e.Fluency[mi])
			out.Consistency = append(out.Consistency, e.Consistency[mi])
		}
	}

	return out, nil
}
