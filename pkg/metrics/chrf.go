package metrics

import "strings"

func ChrF(reference, candidate string) float64 {
	ref := strings.ToLower(reference)
	cand := strings.ToLower(candidate)

	if len(ref) == 0 || len(cand) == 0 {
		return 0
	}

	const maxN = 6
	const beta = 2.0

	var avgPrecision, avgRecall float64
	count := 0

	for n := 1; n <= maxN; n++ {
		refNgrams := charNgrams(ref, n)
		candNgrams := charNgrams(cand, n)

		if len(candNgrams) == 0 || len(refNgrams) == 0 {
			continue
		}

		clipped := 0
		total := 0
		for ng, cnt := range candNgrams {
			total += cnt
			if refCnt, ok := refNgrams[ng]; ok {
				if refCnt < cnt {
					clipped += refCnt
				} else {
					clipped += cnt
				}
			}
		}

		refTotal := 0
		for _, cnt := range refNgrams {
			refTotal += cnt
		}

		avgPrecision += float64(clipped) / float64(total)
		avgRecall += float64(clipped) / float64(refTotal)
		count++
	}

	if count == 0 {
		return 0
	}

	avgPrecision /= float64(count)
	avgRecall /= float64(count)

	if avgPrecision+avgRecall == 0 {
		return 0
	}

	beta2 := beta * beta
	return (1 + beta2) * avgPrecision * avgRecall / (beta2*avgPrecision + avgRecall)
}

func charNgrams(s string, n int) map[string]int {
	m := make(map[string]int)
	runes := []rune(s)
	for i := 0; i <= len(runes)-n; i++ {
		key := string(runes[i : i+n])
		m[key]++
	}

	return m
}
