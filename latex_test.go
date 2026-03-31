package llmbench

import (
	"bytes"
	"strings"
	"testing"
)

func TestTexEscape(t *testing.T) {
	t.Parallel()

	got := TexEscape(`foo_bar 100% & # $ { } ~`)
	if !strings.Contains(got, `\_`) || !strings.Contains(got, `\%`) {
		t.Fatalf("TexEscape: %q", got)
	}
}

func TestShortModelTexName(t *testing.T) {
	t.Parallel()

	t.Run("qwen_path", func(t *testing.T) {
		t.Parallel()

		if got := ShortModelTexName("ollama/qwen2.5:3b-instruct"); got != `qwen2.5:3b-instruct` {
			t.Errorf("got %q", got)
		}
	})

	t.Run("strips_latest", func(t *testing.T) {
		t.Parallel()

		if got := ShortModelTexName("ollama/llama:latest"); !strings.Contains(got, "llama") {
			t.Errorf("got %q", got)
		}
	})
}

func TestLevelTagShort(t *testing.T) {
	t.Parallel()

	t.Run("L1_diagnostic", func(t *testing.T) {
		t.Parallel()

		if got := LevelTagShort("L1-diagnostic"); got != "L1" {
			t.Errorf("got %q, want L1", got)
		}
	})

	t.Run("L2_repair", func(t *testing.T) {
		t.Parallel()

		if got := LevelTagShort("L2-repair"); got != "L2" {
			t.Errorf("got %q, want L2", got)
		}
	})

	t.Run("L3_multi_step", func(t *testing.T) {
		t.Parallel()

		if got := LevelTagShort("L3-multi-step"); got != "L3" {
			t.Errorf("got %q, want L3", got)
		}
	})

	t.Run("passthrough", func(t *testing.T) {
		t.Parallel()

		if got := LevelTagShort("other"); got != "other" {
			t.Errorf("got %q, want other", got)
		}
	})
}

func TestFmtDelta(t *testing.T) {
	t.Parallel()

	t.Run("positive", func(t *testing.T) {
		t.Parallel()

		if got := FmtDelta(0.01); got != "+0.0100" {
			t.Errorf("got %q", got)
		}
	})

	t.Run("negative", func(t *testing.T) {
		t.Parallel()

		if got := FmtDelta(-0.02); got != "-0.0200" {
			t.Errorf("got %q", got)
		}
	})

	t.Run("zero", func(t *testing.T) {
		t.Parallel()

		if got := FmtDelta(0); got != "0" {
			t.Errorf("got %q", got)
		}
	})
}

func TestMetricIsHolmTested(t *testing.T) {
	t.Parallel()

	t.Run("holm_bonferroni_true", func(t *testing.T) {
		t.Parallel()

		if !MetricIsHolmTested(MetricComparison{CorrectionMethod: "holm-bonferroni"}) {
			t.Error("want true")
		}
	})

	t.Run("na_false", func(t *testing.T) {
		t.Parallel()

		if MetricIsHolmTested(MetricComparison{CorrectionMethod: "n/a"}) {
			t.Error("want false")
		}
	})
	t.Run("empty_false", func(t *testing.T) {
		t.Parallel()

		if MetricIsHolmTested(MetricComparison{CorrectionMethod: ""}) {
			t.Error("want false")
		}
	})
}

func TestFilterHolmTestedMetrics(t *testing.T) {
	t.Parallel()

	ms := []MetricComparison{
		{Name: "A", CorrectionMethod: "holm-bonferroni"},
		{Name: "B", CorrectionMethod: "n/a"},
	}

	out := FilterHolmTestedMetrics(ms)
	if len(out) != 1 || out[0].Name != "A" {
		t.Errorf("got %+v", out)
	}
}

func TestCountTaskComparisonDelta(t *testing.T) {
	t.Parallel()

	tc := []TaskComparison{
		{Delta: 0.1},
		{Delta: 0},
		{Delta: -0.05},
	}

	t.Run("wins", func(t *testing.T) {
		t.Parallel()

		if got := CountTaskComparisonDelta(tc, 1); got != 1 {
			t.Errorf("got %d, want 1", got)
		}
	})

	t.Run("ties", func(t *testing.T) {
		t.Parallel()

		if got := CountTaskComparisonDelta(tc, 0); got != 1 {
			t.Errorf("got %d, want 1", got)
		}
	})

	t.Run("losses", func(t *testing.T) {
		t.Parallel()

		if got := CountTaskComparisonDelta(tc, -1); got != 1 {
			t.Errorf("got %d, want 1", got)
		}
	})
}

func TestRenderCompareLatex_smoke(t *testing.T) {
	t.Parallel()

	cr := CompareReport{
		GeneratedAt: "2020-01-01T00:00:00Z",
		ModelA:      "ollama/model_a",
		ModelB:      "ollama/model_b",
		Aggregate: []MetricComparison{
			{
				Name: "ESR", FullName: "Execution Success Rate", HigherIsBetter: true,
				ValueA: 0.5, ValueB: 0.6, Delta: -0.1,
				PValueCorrected: 0.04, CorrectionMethod: "holm-bonferroni",
				Significance: "*", EffectSize: 0.2, EffectLabel: "small",
			},
		},
		PerLevel: []LevelComparison{
			{
				Level: "L1-diagnostic", ESRA: 0.5, ESRB: 0.5, TSAA: 0.5, TSAB: 0.5,
				CHRA: 0.1, CHRB: 0.1, RunsA: 10, RunsB: 10,
			},
		},
		PerTask: []TaskComparison{
			{TaskID: "L1-diag-001", Level: "L1-diagnostic", ESRA: 0.5, ESRB: 0.4, Delta: 0.1},
			{TaskID: "L1-diag-002", Level: "L1-diagnostic", ESRA: 0.6, ESRB: 0.5, Delta: 0.1},
		},
	}
	cr.Raw.A.Metadata.TotalTasks = 60
	cr.Raw.A.Metadata.RunsPerTask = 1
	cr.Raw.A.Metadata.TotalRuns = 60
	cr.Raw.A.Metadata.Seed = 42
	cr.Raw.A.RAG = RAGQualityMetrics{
		MeanPrecisionAtK: 0.5, MeanRecallAtK: 1, MeanFScoreAtK: 0.6,
		MeanMRR: 0.5, MeanNDCGAtK: 0.5,
	}

	var buf bytes.Buffer
	if err := RenderCompareLatex(&buf, cr); err != nil {
		t.Fatalf("RenderCompareLatex: %v", err)
	}
	s := buf.String()
	if !strings.Contains(s, `\begin{table`) {
		t.Fatal("output missing LaTeX table")
	}
	// Model names appear escaped (underscore → \_) in body tables.
	if !strings.Contains(s, "ollama") || !strings.Contains(s, `model\_a`) {
		t.Fatal("output missing model A")
	}
}
