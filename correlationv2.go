package llmbench

import (
	"fmt"
	"math"
	"os"
)

// correlation.go
func KendallTauCorrelation(x, y []float64) float64 {
	n := len(x)
	if n != len(y) || n < 2 {
		return 0
	}
	var concordant, discordant, tiesX, tiesY int
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			dx := x[i] - x[j]
			dy := y[i] - y[j]
			switch {
			case dx*dy > 0:
				concordant++
			case dx*dy < 0:
				discordant++
			default:
				if dx == 0 {
					tiesX++
				}
				if dy == 0 {
					tiesY++
				}
			}
		}
	}
	// Kendall-Tau-B (handles ties, standard for Likert scales)
	n0 := n * (n - 1) / 2
	den := math.Sqrt(float64(n0-tiesX) * float64(n0-tiesY))
	if den == 0 {
		return 0
	}
	return float64(concordant-discordant) / den
}

// Correlation computes Spearman ρ and Pearson r between out.Scores and each
// of the four annotation dimensions.
func CorrelationV2(samples []Sample, scores []float64) CorrelationOutput {
	dims := []struct {
		name string
		vals func(Sample) float64
	}{
		{"coherence", func(s Sample) float64 { return s.Coherence }},
		{"consistency", func(s Sample) float64 { return s.Consistency }},
		{"fluency", func(s Sample) float64 { return s.Fluency }},
		{"relevance", func(s Sample) float64 { return s.Relevance }},
	}

	human := make([]float64, len(samples))
	out := CorrelationOutput{Dimensions: make([]DimCorrelation, len(dims))}
	for i, d := range dims {
		for j, s := range samples {
			human[j] = d.vals(s)
		}
		out.Dimensions[i] = DimCorrelation{
			Dimension:  d.name,
			Spearman:   SpearmanCorrelation(scores, human),
			Pearson:    PearsonCorrelation(scores, human),
			KendallTau: KendallTauCorrelation(scores, human),
		}
	}
	return out
}

func PrintResultV2(name string, n int, corr CorrelationOutput) {
	fmt.Fprintf(os.Stderr, "\n%-16s  samples=%d\n", name, n)
	fmt.Fprintf(os.Stderr, "  %-13s %8s %8s %8s\n", "dimension", "ρ", "r", "τ")
	fmt.Fprintf(os.Stderr, "  %-13s %8s %8s %8s\n", "---------", "---", "---", "---")
	for _, d := range corr.Dimensions {
		fmt.Fprintf(os.Stderr, "  %-13s %8.4f %8.4f %8.4f\n",
			d.Dimension, d.Spearman, d.Pearson, d.KendallTau)
	}
}
