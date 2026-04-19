package llmbench

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
	Dimension  string
	Spearman   float64
	Pearson    float64
	KendallTau float64
}
