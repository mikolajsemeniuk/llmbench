package llmbench

func ROUGEL(reference, candidate string) float64 {
	ref := tokenize(reference)
	cand := tokenize(candidate)

	if len(ref) == 0 || len(cand) == 0 {
		return 0
	}

	lcs := lcs(ref, cand)
	precision := float64(lcs) / float64(len(cand))
	recall := float64(lcs) / float64(len(ref))

	if precision+recall == 0 {
		return 0
	}

	f1 := 2 * precision * recall / (precision + recall)
	return f1
}

func MaxROUGEL(references []string, candidate string) float64 {
	best := 0.0
	for _, ref := range references {
		if s := ROUGEL(ref, candidate); s > best {
			best = s
		}
	}
	return best
}

func lcs(a, b []string) int {
	m, n := len(a), len(b)
	prev := make([]int, n+1)
	curr := make([]int, n+1)

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if a[i-1] == b[j-1] {
				curr[j] = prev[j-1] + 1
				continue
			}

			curr[j] = prev[j]
			if curr[j-1] > curr[j] {
				curr[j] = curr[j-1]
			}
		}

		prev, curr = curr, prev
		for k := range curr {
			curr[k] = 0
		}
	}

	return prev[n]
}
