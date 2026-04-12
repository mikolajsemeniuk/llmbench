package llmbench

import "math"

// PearsonCorrelation computes Pearson's r between two score vectors.
func PearsonCorrelation(x, y []float64) float64 {
	n := len(x)
	if n != len(y) || n == 0 {
		return 0
	}
	var sx, sy, sxy, sx2, sy2 float64
	for i := range x {
		sx += x[i]
		sy += y[i]
		sxy += x[i] * y[i]
		sx2 += x[i] * x[i]
		sy2 += y[i] * y[i]
	}
	nf := float64(n)
	num := nf*sxy - sx*sy
	den := math.Sqrt((nf*sx2 - sx*sx) * (nf*sy2 - sy*sy))
	if den == 0 {
		return 0
	}
	return num / den
}

// SpearmanCorrelation computes Spearman's ρ (rank correlation).
func SpearmanCorrelation(x, y []float64) float64 {
	return PearsonCorrelation(toRanks(x), toRanks(y))
}

func toRanks(vals []float64) []float64 {
	n := len(vals)
	type iv struct {
		v float64
		i int
	}

	s := make([]iv, n)
	for i, v := range vals {
		s[i] = iv{v, i}
	}

	for i := 1; i < n; i++ {
		for j := i; j > 0 && s[j].v < s[j-1].v; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}

	ranks := make([]float64, n)
	for i := 0; i < n; {
		j := i + 1
		for j < n && s[j].v == s[i].v {
			j++
		}
		avg := float64(i+j+1) / 2.0
		for k := i; k < j; k++ {
			ranks[s[k].i] = avg
		}
		i = j
	}

	return ranks
}
