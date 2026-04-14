package llmbench

import (
	"fmt"
	"math"
	"os"
)

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
	return PearsonCorrelation(ranks(x), ranks(y))
}

// ScoreOutput holds parallel slices of metric scores and human annotation
// dimensions collected over all (entry, machine-summary) pairs in a dataset.
type ScoreOutput struct {
	Scores      []float64
	Relevance   []float64
	Coherence   []float64
	Fluency     []float64
	Consistency []float64
}

// DimCorrelation is the Spearman and Pearson correlation of metric scores
// against a single human-annotation dimension.
type DimCorrelation struct {
	Dimension string
	Spearman  float64
	Pearson   float64
}

// CorrelationOutput groups correlations for all four SummEval dimensions.
type CorrelationOutput struct {
	Dimensions []DimCorrelation
}

// Correlation computes Spearman ρ and Pearson r between out.Scores and each
// of the four annotation dimensions.
func Correlation(out ScoreOutput) CorrelationOutput {
	dims := []struct {
		name string
		vals []float64
	}{
		{"coherence", out.Coherence},
		{"consistency", out.Consistency},
		{"fluency", out.Fluency},
		{"relevance", out.Relevance},
	}
	result := CorrelationOutput{Dimensions: make([]DimCorrelation, 0, len(dims))}
	for _, d := range dims {
		result.Dimensions = append(result.Dimensions, DimCorrelation{
			Dimension: d.name,
			Spearman:  SpearmanCorrelation(out.Scores, d.vals),
			Pearson:   PearsonCorrelation(out.Scores, d.vals),
		})
	}
	return result
}

// PrintResult prints a correlation table for the named metric to stderr.
// The leading newline terminates any in-progress \r progress line from
// async scorers.
func PrintResult(name string, out ScoreOutput, corr CorrelationOutput) {
	fmt.Fprintf(os.Stderr, "\nsamples: %d\n\n", len(out.Scores))
	fmt.Fprintf(os.Stderr, "%s summary-level correlations:\n", name)
	fmt.Fprintf(os.Stderr, "%-15s %10s %10s\n", "dimension", "spearman", "pearson")
	fmt.Fprintf(os.Stderr, "%-15s %10s %10s\n", "---------", "--------", "-------")
	for _, d := range corr.Dimensions {
		fmt.Fprintf(os.Stderr, "%-15s %10.4f %10.4f\n", d.Dimension, d.Spearman, d.Pearson)
	}
}

func ranks(vals []float64) []float64 {
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
