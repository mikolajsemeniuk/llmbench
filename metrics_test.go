package llmbench

import (
	"math"
	"math/rand"
	"testing"
)

const floatTolerance = 1e-9

func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

// ---------------------------------------------------------------------------
// Ratio-based metrics (ESR, SVR, TSA, CHR, SCR, CDS, FCSR, MFS, ERR, CTR, DAAR)
// ---------------------------------------------------------------------------

func TestExecutionSuccessRate(t *testing.T) {
	t.Parallel()

	t.Run("all_pass", func(t *testing.T) {
		t.Parallel()
		if got := ExecutionSuccessRate(10, 10); got != 1.0 {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("half", func(t *testing.T) {
		t.Parallel()
		if got := ExecutionSuccessRate(5, 10); got != 0.5 {
			t.Errorf("got %f, want 0.5", got)
		}
	})

	t.Run("none", func(t *testing.T) {
		t.Parallel()
		if got := ExecutionSuccessRate(0, 10); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("zero_total", func(t *testing.T) {
		t.Parallel()
		if got := ExecutionSuccessRate(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestSyntaxValidationRate(t *testing.T) {
	t.Parallel()

	t.Run("all_valid", func(t *testing.T) {
		t.Parallel()
		if got := SyntaxValidationRate(20, 20); got != 1.0 {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("zero_total", func(t *testing.T) {
		t.Parallel()
		if got := SyntaxValidationRate(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestToolSelectionAccuracy(t *testing.T) {
	t.Parallel()

	t.Run("perfect", func(t *testing.T) {
		t.Parallel()
		if got := ToolSelectionAccuracy(8, 8); got != 1.0 {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("partial", func(t *testing.T) {
		t.Parallel()
		if got := ToolSelectionAccuracy(3, 4); got != 0.75 {
			t.Errorf("got %f, want 0.75", got)
		}
	})

	t.Run("zero_total", func(t *testing.T) {
		t.Parallel()
		if got := ToolSelectionAccuracy(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestContextHallucinationRate(t *testing.T) {
	t.Parallel()

	t.Run("no_hallucination", func(t *testing.T) {
		t.Parallel()
		if got := ContextHallucinationRate(0, 5); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("all_hallucinated", func(t *testing.T) {
		t.Parallel()
		if got := ContextHallucinationRate(5, 5); got != 1.0 {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("zero_args", func(t *testing.T) {
		t.Parallel()
		if got := ContextHallucinationRate(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestSchemaComplianceRate(t *testing.T) {
	t.Parallel()

	t.Run("all_compliant", func(t *testing.T) {
		t.Parallel()
		if got := SchemaComplianceRate(10, 10); got != 1.0 {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("zero_total", func(t *testing.T) {
		t.Parallel()
		if got := SchemaComplianceRate(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestContextDensityScore(t *testing.T) {
	t.Parallel()

	t.Run("half_relevant", func(t *testing.T) {
		t.Parallel()
		if got := ContextDensityScore(50, 100); got != 0.5 {
			t.Errorf("got %f, want 0.5", got)
		}
	})

	t.Run("zero_context", func(t *testing.T) {
		t.Parallel()
		if got := ContextDensityScore(10, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestFirstCallSuccessRate(t *testing.T) {
	t.Parallel()

	t.Run("all_first_call", func(t *testing.T) {
		t.Parallel()
		if got := FirstCallSuccessRate(8, 8); got != 1.0 {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("partial", func(t *testing.T) {
		t.Parallel()
		if got := FirstCallSuccessRate(6, 8); got != 0.75 {
			t.Errorf("got %f, want 0.75", got)
		}
	})

	t.Run("zero_tasks", func(t *testing.T) {
		t.Parallel()
		if got := FirstCallSuccessRate(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestMultiStepFaithfulnessScore(t *testing.T) {
	t.Parallel()

	t.Run("all_grounded", func(t *testing.T) {
		t.Parallel()
		if got := MultiStepFaithfulnessScore(5, 5); got != 1.0 {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("zero_steps", func(t *testing.T) {
		t.Parallel()
		if got := MultiStepFaithfulnessScore(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestErrorRecoveryRate(t *testing.T) {
	t.Parallel()

	t.Run("all_recovered", func(t *testing.T) {
		t.Parallel()
		if got := ErrorRecoveryRate(3, 3); got != 1.0 {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("none_recovered", func(t *testing.T) {
		t.Parallel()
		if got := ErrorRecoveryRate(0, 5); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("no_initial_errors", func(t *testing.T) {
		t.Parallel()
		if got := ErrorRecoveryRate(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestContextTruncationRate(t *testing.T) {
	t.Parallel()

	t.Run("some_truncated", func(t *testing.T) {
		t.Parallel()
		if got := ContextTruncationRate(2, 10); got != 0.2 {
			t.Errorf("got %f, want 0.2", got)
		}
	})

	t.Run("zero_tasks", func(t *testing.T) {
		t.Parallel()
		if got := ContextTruncationRate(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestDestructiveActionAttemptRate(t *testing.T) {
	t.Parallel()

	t.Run("no_destructive", func(t *testing.T) {
		t.Parallel()
		if got := DestructiveActionAttemptRate(0, 20); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("some_destructive", func(t *testing.T) {
		t.Parallel()
		if got := DestructiveActionAttemptRate(2, 10); got != 0.2 {
			t.Errorf("got %f, want 0.2", got)
		}
	})

	t.Run("zero_total", func(t *testing.T) {
		t.Parallel()
		if got := DestructiveActionAttemptRate(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

// ---------------------------------------------------------------------------
// LatencyToActionEfficiency
// ---------------------------------------------------------------------------

func TestLatencyToActionEfficiency(t *testing.T) {
	t.Parallel()

	t.Run("normal", func(t *testing.T) {
		t.Parallel()
		got := LatencyToActionEfficiency(0.8, 2.0)
		if !approxEqual(got, 0.4, floatTolerance) {
			t.Errorf("got %f, want 0.4", got)
		}
	})

	t.Run("zero_latency", func(t *testing.T) {
		t.Parallel()
		if got := LatencyToActionEfficiency(1.0, 0.0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("negative_latency", func(t *testing.T) {
		t.Parallel()
		if got := LatencyToActionEfficiency(1.0, -1.0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

// ---------------------------------------------------------------------------
// MeanTimeToRecovery
// ---------------------------------------------------------------------------

func TestMeanTimeToRecovery(t *testing.T) {
	t.Parallel()

	t.Run("single_value", func(t *testing.T) {
		t.Parallel()
		if got := MeanTimeToRecovery([]float64{5.0}); got != 5.0 {
			t.Errorf("got %f, want 5.0", got)
		}
	})

	t.Run("multiple_values", func(t *testing.T) {
		t.Parallel()
		got := MeanTimeToRecovery([]float64{2.0, 4.0, 6.0})
		if !approxEqual(got, 4.0, floatTolerance) {
			t.Errorf("got %f, want 4.0", got)
		}
	})

	t.Run("empty_slice", func(t *testing.T) {
		t.Parallel()
		if got := MeanTimeToRecovery(nil); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

// ---------------------------------------------------------------------------
// LatencyPercentile — interpolation logic
// ---------------------------------------------------------------------------

func TestLatencyPercentile(t *testing.T) {
	t.Parallel()

	t.Run("p50_odd_count", func(t *testing.T) {
		t.Parallel()
		got := LatencyPercentile([]float64{1, 2, 3, 4, 5}, 50)
		if !approxEqual(got, 3.0, floatTolerance) {
			t.Errorf("got %f, want 3.0", got)
		}
	})

	t.Run("p50_even_count", func(t *testing.T) {
		t.Parallel()
		got := LatencyPercentile([]float64{1, 2, 3, 4}, 50)
		if !approxEqual(got, 2.5, floatTolerance) {
			t.Errorf("got %f, want 2.5", got)
		}
	})

	t.Run("p0_returns_min", func(t *testing.T) {
		t.Parallel()
		got := LatencyPercentile([]float64{5, 1, 9, 3}, 0)
		if !approxEqual(got, 1.0, floatTolerance) {
			t.Errorf("got %f, want 1.0 (min)", got)
		}
	})

	t.Run("p100_returns_max", func(t *testing.T) {
		t.Parallel()
		got := LatencyPercentile([]float64{5, 1, 9, 3}, 100)
		if !approxEqual(got, 9.0, floatTolerance) {
			t.Errorf("got %f, want 9.0 (max)", got)
		}
	})

	t.Run("p90_interpolation", func(t *testing.T) {
		t.Parallel()
		// [1,2,3,4,5], idx = 0.9*4 = 3.6 → 4*(1-0.6) + 5*0.6 = 1.6+3.0 = 4.6
		got := LatencyPercentile([]float64{1, 2, 3, 4, 5}, 90)
		if !approxEqual(got, 4.6, floatTolerance) {
			t.Errorf("got %f, want 4.6", got)
		}
	})

	t.Run("single_element", func(t *testing.T) {
		t.Parallel()
		got := LatencyPercentile([]float64{42.0}, 50)
		if !approxEqual(got, 42.0, floatTolerance) {
			t.Errorf("got %f, want 42.0", got)
		}
	})

	t.Run("empty_slice", func(t *testing.T) {
		t.Parallel()
		if got := LatencyPercentile(nil, 50); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("negative_percentile", func(t *testing.T) {
		t.Parallel()
		if got := LatencyPercentile([]float64{1, 2, 3}, -5); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("percentile_above_100", func(t *testing.T) {
		t.Parallel()
		if got := LatencyPercentile([]float64{1, 2, 3}, 101); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("unsorted_input", func(t *testing.T) {
		t.Parallel()
		got := LatencyPercentile([]float64{5, 3, 1, 4, 2}, 50)
		if !approxEqual(got, 3.0, floatTolerance) {
			t.Errorf("got %f, want 3.0 (should sort internally)", got)
		}
	})
}

// ---------------------------------------------------------------------------
// RAG Precision / Recall / F-Score
// ---------------------------------------------------------------------------

func TestRAGPrecisionAtK(t *testing.T) {
	t.Parallel()

	t.Run("two_of_four_relevant", func(t *testing.T) {
		t.Parallel()
		if got := RAGPrecisionAtK(2, 4); got != 0.5 {
			t.Errorf("got %f, want 0.5", got)
		}
	})

	t.Run("zero_k", func(t *testing.T) {
		t.Parallel()
		if got := RAGPrecisionAtK(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestRAGRecallAtK(t *testing.T) {
	t.Parallel()

	t.Run("perfect_recall", func(t *testing.T) {
		t.Parallel()
		if got := RAGRecallAtK(3, 3); got != 1.0 {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("partial_recall", func(t *testing.T) {
		t.Parallel()
		if got := RAGRecallAtK(2, 4); got != 0.5 {
			t.Errorf("got %f, want 0.5", got)
		}
	})

	t.Run("zero_relevant", func(t *testing.T) {
		t.Parallel()
		if got := RAGRecallAtK(0, 0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestRAGFScoreAtK(t *testing.T) {
	t.Parallel()

	t.Run("perfect_f1", func(t *testing.T) {
		t.Parallel()
		got := RAGFScoreAtK(1.0, 1.0, 1.0)
		if !approxEqual(got, 1.0, floatTolerance) {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("precision_biased_f1", func(t *testing.T) {
		t.Parallel()
		// P=0.5, R=1.0, beta=1.0 → F1 = 2*0.5*1/(0.5+1) = 2/3
		got := RAGFScoreAtK(0.5, 1.0, 1.0)
		if !approxEqual(got, 2.0/3.0, floatTolerance) {
			t.Errorf("got %f, want %f", got, 2.0/3.0)
		}
	})

	t.Run("both_zero", func(t *testing.T) {
		t.Parallel()
		if got := RAGFScoreAtK(0.0, 0.0, 1.0); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("beta_2_favors_recall", func(t *testing.T) {
		t.Parallel()
		// P=0.5, R=1.0, beta=2.0 → F = (1+4)*0.5*1/(4*0.5+1) = 2.5/3 ≈ 0.8333
		got := RAGFScoreAtK(0.5, 1.0, 2.0)
		if !approxEqual(got, 2.5/3.0, floatTolerance) {
			t.Errorf("got %f, want %f", got, 2.5/3.0)
		}
	})
}

// ---------------------------------------------------------------------------
// MeanReciprocalRank
// ---------------------------------------------------------------------------

func TestMeanReciprocalRank(t *testing.T) {
	t.Parallel()

	t.Run("all_rank_1", func(t *testing.T) {
		t.Parallel()
		got := MeanReciprocalRank([]int{1, 1, 1})
		if !approxEqual(got, 1.0, floatTolerance) {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("varied_ranks", func(t *testing.T) {
		t.Parallel()
		// (1/1 + 1/2 + 1/3) / 3 = (1 + 0.5 + 0.333) / 3 ≈ 0.6111
		got := MeanReciprocalRank([]int{1, 2, 3})
		want := (1.0 + 0.5 + 1.0/3.0) / 3.0
		if !approxEqual(got, want, floatTolerance) {
			t.Errorf("got %f, want %f", got, want)
		}
	})

	t.Run("no_relevant_found", func(t *testing.T) {
		t.Parallel()
		got := MeanReciprocalRank([]int{0, 0})
		if !approxEqual(got, 0.0, floatTolerance) {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("empty_input", func(t *testing.T) {
		t.Parallel()
		if got := MeanReciprocalRank(nil); got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

// ---------------------------------------------------------------------------
// NDCGAtK — logarithmic discounting + normalization
// ---------------------------------------------------------------------------

func TestNDCGAtK(t *testing.T) {
	t.Parallel()

	t.Run("perfect_ranking", func(t *testing.T) {
		t.Parallel()
		retrieved := []float64{3, 1, 0}
		ideal := []float64{3, 1, 0}
		got := NDCGAtK(retrieved, ideal, 3)
		if !approxEqual(got, 1.0, floatTolerance) {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("reversed_ranking", func(t *testing.T) {
		t.Parallel()
		retrieved := []float64{0, 1, 3}
		ideal := []float64{3, 1, 0}
		got := NDCGAtK(retrieved, ideal, 3)
		// DCG = 0/log2(2) + 1/log2(3) + 7/log2(4) = 0 + 0.63093 + 3.5 = 4.13093
		// IDCG = 7/log2(2) + 1/log2(3) + 0/log2(4) = 7 + 0.63093 + 0 = 7.63093
		want := 4.13093 / 7.63093
		if !approxEqual(got, want, 1e-4) {
			t.Errorf("got %f, want %f", got, want)
		}
	})

	t.Run("all_zeros", func(t *testing.T) {
		t.Parallel()
		got := NDCGAtK([]float64{0, 0, 0}, []float64{0, 0, 0}, 3)
		if got != 0.0 {
			t.Errorf("got %f, want 0.0 (IDCG=0)", got)
		}
	})

	t.Run("k_larger_than_slice", func(t *testing.T) {
		t.Parallel()
		got := NDCGAtK([]float64{3}, []float64{3}, 5)
		if !approxEqual(got, 1.0, floatTolerance) {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("zero_k", func(t *testing.T) {
		t.Parallel()
		got := NDCGAtK([]float64{3, 1}, []float64{3, 1}, 0)
		if got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("empty_input", func(t *testing.T) {
		t.Parallel()
		got := NDCGAtK(nil, nil, 3)
		if got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

// ---------------------------------------------------------------------------
// RecoveryPlanRationality — Levenshtein on string sequences
// ---------------------------------------------------------------------------

func TestRecoveryPlanRationality(t *testing.T) {
	t.Parallel()

	t.Run("identical_sequences", func(t *testing.T) {
		t.Parallel()
		got := RecoveryPlanRationality(
			[]string{"get_pod", "describe_pod", "patch_pod"},
			[]string{"get_pod", "describe_pod", "patch_pod"},
		)
		if !approxEqual(got, 1.0, floatTolerance) {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("completely_different", func(t *testing.T) {
		t.Parallel()
		got := RecoveryPlanRationality(
			[]string{"delete_ns", "delete_all"},
			[]string{"get_pod", "patch_pod"},
		)
		if !approxEqual(got, 0.0, floatTolerance) {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("one_substitution", func(t *testing.T) {
		t.Parallel()
		got := RecoveryPlanRationality(
			[]string{"get_pod", "delete_pod", "patch_pod"},
			[]string{"get_pod", "describe_pod", "patch_pod"},
		)
		// dist=1, maxLen=3, RPR = 1 - 1/3 ≈ 0.6667
		want := 1.0 - 1.0/3.0
		if !approxEqual(got, want, floatTolerance) {
			t.Errorf("got %f, want %f", got, want)
		}
	})

	t.Run("model_empty", func(t *testing.T) {
		t.Parallel()
		got := RecoveryPlanRationality(nil, []string{"get_pod", "patch_pod"})
		if !approxEqual(got, 0.0, floatTolerance) {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("both_empty", func(t *testing.T) {
		t.Parallel()
		got := RecoveryPlanRationality(nil, nil)
		if !approxEqual(got, 1.0, floatTolerance) {
			t.Errorf("got %f, want 1.0 (trivial plan)", got)
		}
	})

	t.Run("model_longer", func(t *testing.T) {
		t.Parallel()
		got := RecoveryPlanRationality(
			[]string{"get_pod", "describe_pod", "patch_pod", "verify_pod"},
			[]string{"get_pod", "patch_pod"},
		)
		// dist=2 (delete "describe_pod" + delete "verify_pod"), maxLen=4, RPR = 1 - 2/4 = 0.5
		want := 0.5
		if !approxEqual(got, want, floatTolerance) {
			t.Errorf("got %f, want %f", got, want)
		}
	})
}

// ---------------------------------------------------------------------------
// BootstrapConfidenceInterval — Monte Carlo simulation
// ---------------------------------------------------------------------------

func TestBootstrapConfidenceInterval(t *testing.T) {
	t.Parallel()

	t.Run("all_successes", func(t *testing.T) {
		t.Parallel()
		rng := rand.New(rand.NewSource(42))
		ci := BootstrapConfidenceInterval(10, 10, 10000, 0.05, rng.Float64)
		if ci[0] != 1.0 || ci[1] != 1.0 {
			t.Errorf("got [%f, %f], want [1.0, 1.0]", ci[0], ci[1])
		}
	})

	t.Run("zero_successes", func(t *testing.T) {
		t.Parallel()
		rng := rand.New(rand.NewSource(42))
		ci := BootstrapConfidenceInterval(0, 10, 10000, 0.05, rng.Float64)
		if ci[0] != 0.0 || ci[1] != 0.0 {
			t.Errorf("got [%f, %f], want [0.0, 0.0]", ci[0], ci[1])
		}
	})

	t.Run("zero_total", func(t *testing.T) {
		t.Parallel()
		rng := rand.New(rand.NewSource(42))
		ci := BootstrapConfidenceInterval(0, 0, 10000, 0.05, rng.Float64)
		if ci[0] != 0.0 || ci[1] != 0.0 {
			t.Errorf("got [%f, %f], want [0.0, 0.0]", ci[0], ci[1])
		}
	})

	t.Run("mid_range_contains_observed", func(t *testing.T) {
		t.Parallel()
		rng := rand.New(rand.NewSource(42))
		ci := BootstrapConfidenceInterval(8, 10, 10000, 0.05, rng.Float64)
		pObs := 0.8
		if ci[0] > pObs || ci[1] < pObs {
			t.Errorf("CI [%f, %f] does not contain observed rate %f", ci[0], ci[1], pObs)
		}
	})

	t.Run("mid_range_lower_lt_upper", func(t *testing.T) {
		t.Parallel()
		rng := rand.New(rand.NewSource(42))
		ci := BootstrapConfidenceInterval(8, 10, 10000, 0.05, rng.Float64)
		if ci[0] >= ci[1] {
			t.Errorf("lower bound %f should be < upper bound %f", ci[0], ci[1])
		}
	})

	t.Run("mid_range_bounds_are_valid", func(t *testing.T) {
		t.Parallel()
		rng := rand.New(rand.NewSource(42))
		ci := BootstrapConfidenceInterval(8, 10, 10000, 0.05, rng.Float64)
		if ci[0] < 0.0 || ci[1] > 1.0 {
			t.Errorf("CI [%f, %f] out of [0, 1] range", ci[0], ci[1])
		}
	})

	t.Run("zero_bootstrap_samples", func(t *testing.T) {
		t.Parallel()
		rng := rand.New(rand.NewSource(42))
		ci := BootstrapConfidenceInterval(5, 10, 0, 0.05, rng.Float64)
		if ci[0] != 0.0 || ci[1] != 0.0 {
			t.Errorf("got [%f, %f], want [0.0, 0.0]", ci[0], ci[1])
		}
	})
}

// ---------------------------------------------------------------------------
// CliffsData — dominance effect size
// ---------------------------------------------------------------------------

func TestCliffsData(t *testing.T) {
	t.Parallel()

	t.Run("a_fully_dominates", func(t *testing.T) {
		t.Parallel()
		got := CliffsData([]float64{5, 6, 7}, []float64{1, 2, 3})
		if !approxEqual(got, 1.0, floatTolerance) {
			t.Errorf("got %f, want 1.0", got)
		}
	})

	t.Run("b_fully_dominates", func(t *testing.T) {
		t.Parallel()
		got := CliffsData([]float64{1, 2, 3}, []float64{5, 6, 7})
		if !approxEqual(got, -1.0, floatTolerance) {
			t.Errorf("got %f, want -1.0", got)
		}
	})

	t.Run("identical_groups", func(t *testing.T) {
		t.Parallel()
		got := CliffsData([]float64{1, 2, 3}, []float64{1, 2, 3})
		if !approxEqual(got, 0.0, floatTolerance) {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("empty_group_a", func(t *testing.T) {
		t.Parallel()
		got := CliffsData(nil, []float64{1, 2, 3})
		if got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("empty_group_b", func(t *testing.T) {
		t.Parallel()
		got := CliffsData([]float64{1, 2, 3}, nil)
		if got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("partial_overlap", func(t *testing.T) {
		t.Parallel()
		// A=[3,4], B=[1,5] → pairs: (3>1,3<5,4>1,4<5) → dom=2, dominated=2, δ=0
		got := CliffsData([]float64{3, 4}, []float64{1, 5})
		if !approxEqual(got, 0.0, floatTolerance) {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

func TestCliffsEffectSizeLabel(t *testing.T) {
	t.Parallel()

	t.Run("negligible", func(t *testing.T) {
		t.Parallel()
		if got := CliffsEffectSizeLabel(0.1); got != "negligible" {
			t.Errorf("got %q, want negligible", got)
		}
	})

	t.Run("small", func(t *testing.T) {
		t.Parallel()
		if got := CliffsEffectSizeLabel(0.2); got != "small" {
			t.Errorf("got %q, want small", got)
		}
	})

	t.Run("medium", func(t *testing.T) {
		t.Parallel()
		if got := CliffsEffectSizeLabel(0.4); got != "medium" {
			t.Errorf("got %q, want medium", got)
		}
	})

	t.Run("large", func(t *testing.T) {
		t.Parallel()
		if got := CliffsEffectSizeLabel(0.5); got != "large" {
			t.Errorf("got %q, want large", got)
		}
	})

	t.Run("negative_large", func(t *testing.T) {
		t.Parallel()
		if got := CliffsEffectSizeLabel(-0.8); got != "large" {
			t.Errorf("got %q, want large (uses abs)", got)
		}
	})

	t.Run("zero", func(t *testing.T) {
		t.Parallel()
		if got := CliffsEffectSizeLabel(0.0); got != "negligible" {
			t.Errorf("got %q, want negligible", got)
		}
	})
}

// ---------------------------------------------------------------------------
// WilcoxonRankSum — Mann-Whitney U test
// ---------------------------------------------------------------------------

func TestWilcoxonRankSum(t *testing.T) {
	t.Parallel()

	t.Run("clearly_different_groups", func(t *testing.T) {
		t.Parallel()
		a := []float64{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
		b := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
		_, p := WilcoxonRankSum(a, b)
		if p >= 0.05 {
			t.Errorf("p=%f, expected p < 0.05 for non-overlapping groups", p)
		}
	})

	t.Run("identical_groups_not_significant", func(t *testing.T) {
		t.Parallel()
		a := []float64{1, 2, 3, 4, 5}
		b := []float64{1, 2, 3, 4, 5}
		_, p := WilcoxonRankSum(a, b)
		if p < 0.05 {
			t.Errorf("p=%f, expected p >= 0.05 for identical groups", p)
		}
	})

	t.Run("u_stat_is_zero_for_full_separation", func(t *testing.T) {
		t.Parallel()
		a := []float64{1, 2, 3}
		b := []float64{4, 5, 6}
		u, _ := WilcoxonRankSum(a, b)
		if !approxEqual(u, 0.0, floatTolerance) {
			t.Errorf("U=%f, want 0.0 (group A fully below group B)", u)
		}
	})

	t.Run("symmetric_u", func(t *testing.T) {
		t.Parallel()
		a := []float64{4, 5, 6}
		b := []float64{1, 2, 3}
		u, _ := WilcoxonRankSum(a, b)
		if !approxEqual(u, 0.0, floatTolerance) {
			t.Errorf("U=%f, want 0.0 (min of U_A, U_B)", u)
		}
	})

	t.Run("overlapping_groups", func(t *testing.T) {
		t.Parallel()
		a := []float64{1, 3, 5, 7, 9}
		b := []float64{2, 4, 6, 8, 10}
		_, p := WilcoxonRankSum(a, b)
		if p < 0.05 {
			t.Errorf("p=%f, expected p >= 0.05 for interleaved groups", p)
		}
	})

	t.Run("empty_group_a", func(t *testing.T) {
		t.Parallel()
		_, p := WilcoxonRankSum(nil, []float64{1, 2, 3})
		if p != 1.0 {
			t.Errorf("p=%f, want 1.0 for empty group", p)
		}
	})

	t.Run("empty_group_b", func(t *testing.T) {
		t.Parallel()
		_, p := WilcoxonRankSum([]float64{1, 2, 3}, nil)
		if p != 1.0 {
			t.Errorf("p=%f, want 1.0 for empty group", p)
		}
	})

	t.Run("tied_values_handled", func(t *testing.T) {
		t.Parallel()
		a := []float64{1, 1, 1, 1, 1}
		b := []float64{1, 1, 1, 1, 1}
		_, p := WilcoxonRankSum(a, b)
		if p < 0.05 {
			t.Errorf("p=%f, expected p >= 0.05 for all tied values", p)
		}
	})
}

func TestWilcoxonSignificanceLabel(t *testing.T) {
	t.Parallel()

	t.Run("highly_significant", func(t *testing.T) {
		t.Parallel()
		if got := WilcoxonSignificanceLabel(0.0005); got != "***" {
			t.Errorf("got %q, want ***", got)
		}
	})

	t.Run("very_significant", func(t *testing.T) {
		t.Parallel()
		if got := WilcoxonSignificanceLabel(0.005); got != "**" {
			t.Errorf("got %q, want **", got)
		}
	})

	t.Run("significant", func(t *testing.T) {
		t.Parallel()
		if got := WilcoxonSignificanceLabel(0.03); got != "*" {
			t.Errorf("got %q, want *", got)
		}
	})

	t.Run("not_significant", func(t *testing.T) {
		t.Parallel()
		if got := WilcoxonSignificanceLabel(0.1); got != "n.s." {
			t.Errorf("got %q, want n.s.", got)
		}
	})
}

// ---------------------------------------------------------------------------
// Token Efficiency + Payload extraction
// ---------------------------------------------------------------------------

func TestExtractMachineActionablePayload(t *testing.T) {
	t.Parallel()

	t.Run("valid_json_minification", func(t *testing.T) {
		t.Parallel()
		input := `{  "name":  "nginx" ,  "namespace": "default"  }`
		got, err := ExtractMachineActionablePayload(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := `{"name":"nginx","namespace":"default"}`
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	t.Run("already_minified", func(t *testing.T) {
		t.Parallel()
		input := `{"a":1}`
		got, err := ExtractMachineActionablePayload(input)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != input {
			t.Errorf("got %q, want %q", got, input)
		}
	})

	t.Run("invalid_json", func(t *testing.T) {
		t.Parallel()
		_, err := ExtractMachineActionablePayload(`{not json}`)
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})

	t.Run("empty_string", func(t *testing.T) {
		t.Parallel()
		_, err := ExtractMachineActionablePayload("")
		if err == nil {
			t.Error("expected error for empty string")
		}
	})
}

func TestCalculateTokenEfficiency(t *testing.T) {
	t.Parallel()

	t.Run("valid_json_with_simple_tokenizer", func(t *testing.T) {
		t.Parallel()
		tokenizer := func(s string) int { return len(s) }
		got := CalculateTokenEfficiency(`{"name":"nginx"}`, 100, tokenizer)
		// minified = `{"name":"nginx"}` (16 chars), TE = 16/100 = 0.16
		if !approxEqual(got, 0.16, floatTolerance) {
			t.Errorf("got %f, want 0.16", got)
		}
	})

	t.Run("invalid_json_returns_zero", func(t *testing.T) {
		t.Parallel()
		tokenizer := func(s string) int { return len(s) }
		got := CalculateTokenEfficiency(`not json`, 100, tokenizer)
		if got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("zero_completion_tokens", func(t *testing.T) {
		t.Parallel()
		tokenizer := func(s string) int { return len(s) }
		got := CalculateTokenEfficiency(`{"a":1}`, 0, tokenizer)
		if got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

// ---------------------------------------------------------------------------
// LostInTheMiddleVulnerability
// ---------------------------------------------------------------------------

func TestLostInTheMiddleVulnerability(t *testing.T) {
	t.Parallel()

	t.Run("edges_better_than_middle", func(t *testing.T) {
		t.Parallel()
		got := LostInTheMiddleVulnerability(0.9, 0.6)
		if !approxEqual(got, 0.3, floatTolerance) {
			t.Errorf("got %f, want 0.3", got)
		}
	})

	t.Run("no_vulnerability", func(t *testing.T) {
		t.Parallel()
		got := LostInTheMiddleVulnerability(0.8, 0.8)
		if !approxEqual(got, 0.0, floatTolerance) {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("middle_better_than_edges", func(t *testing.T) {
		t.Parallel()
		got := LostInTheMiddleVulnerability(0.5, 0.7)
		if !approxEqual(got, -0.2, floatTolerance) {
			t.Errorf("got %f, want -0.2", got)
		}
	})
}

// ---------------------------------------------------------------------------
// CostEfficiencyScore
// ---------------------------------------------------------------------------

func TestCostEfficiencyScore(t *testing.T) {
	t.Parallel()

	t.Run("positive_cost", func(t *testing.T) {
		t.Parallel()
		got := CostEfficiencyScore(10, 2.0)
		if !approxEqual(got, 5.0, floatTolerance) {
			t.Errorf("got %f, want 5.0", got)
		}
	})

	t.Run("zero_cost_returns_inf", func(t *testing.T) {
		t.Parallel()
		got := CostEfficiencyScore(10, 0.0)
		if !math.IsInf(got, 1) {
			t.Errorf("got %f, want +Inf", got)
		}
	})

	t.Run("negative_cost_returns_inf", func(t *testing.T) {
		t.Parallel()
		got := CostEfficiencyScore(10, -1.0)
		if !math.IsInf(got, 1) {
			t.Errorf("got %f, want +Inf", got)
		}
	})
}

// ---------------------------------------------------------------------------
// ContextCompressionRatio
// ---------------------------------------------------------------------------

func TestContextCompressionRatio(t *testing.T) {
	t.Parallel()

	t.Run("fifty_percent_compression", func(t *testing.T) {
		t.Parallel()
		got := ContextCompressionRatio(1000, 500)
		if !approxEqual(got, 0.5, floatTolerance) {
			t.Errorf("got %f, want 0.5", got)
		}
	})

	t.Run("no_compression", func(t *testing.T) {
		t.Parallel()
		got := ContextCompressionRatio(1000, 1000)
		if got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("compressed_bigger_than_original", func(t *testing.T) {
		t.Parallel()
		got := ContextCompressionRatio(1000, 1200)
		if got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})

	t.Run("zero_original", func(t *testing.T) {
		t.Parallel()
		got := ContextCompressionRatio(0, 0)
		if got != 0.0 {
			t.Errorf("got %f, want 0.0", got)
		}
	})
}

// ---------------------------------------------------------------------------
// CountRelevantTokensFromContext
// ---------------------------------------------------------------------------

func TestCountRelevantTokensFromContext(t *testing.T) {
	t.Parallel()

	t.Run("overlapping_tokens", func(t *testing.T) {
		t.Parallel()
		payload := `{"name": "nginx", "namespace": "default"}`
		context := `name: nginx namespace: default status: running`
		tokenizer := func(s string) int { return len(s) }
		got := CountRelevantTokensFromContext(payload, context, tokenizer)
		// payload words (>1 char): "name", "nginx", "namespace", "default"
		// context words: "name", "nginx", "namespace", "default", "status", "running"
		// overlap: all 4 payload words exist in context
		if got != 4 {
			t.Errorf("got %d, want 4", got)
		}
	})

	t.Run("no_overlap", func(t *testing.T) {
		t.Parallel()
		payload := `{"foo": "bar"}`
		context := `completely unrelated text here`
		tokenizer := func(s string) int { return len(s) }
		got := CountRelevantTokensFromContext(payload, context, tokenizer)
		if got != 0 {
			t.Errorf("got %d, want 0", got)
		}
	})
}

// ---------------------------------------------------------------------------
// Multiple comparisons correction — Bonferroni and Holm-Bonferroni
// ---------------------------------------------------------------------------

func TestBonferroniCorrection(t *testing.T) {
	t.Parallel()

	t.Run("basic_scaling", func(t *testing.T) {
		t.Parallel()
		// m=3: adjusted[i] = raw[i] * 3; no cap required.
		in := []float64{0.01, 0.04, 0.03}
		got := BonferroniCorrection(in)
		want := []float64{0.03, 0.12, 0.09}
		for i := range want {
			if !approxEqual(got[i], want[i], floatTolerance) {
				t.Errorf("index %d: got %.10f, want %.10f", i, got[i], want[i])
			}
		}
	})

	t.Run("cap_at_one", func(t *testing.T) {
		t.Parallel()
		// m=3, raw=0.8 → 0.8*3=2.4 → capped at 1.0.
		in := []float64{0.8, 0.01, 0.02}
		got := BonferroniCorrection(in)
		if got[0] != 1.0 {
			t.Errorf("got %f, want 1.0 (cap)", got[0])
		}
	})

	t.Run("empty_input", func(t *testing.T) {
		t.Parallel()
		got := BonferroniCorrection([]float64{})
		if len(got) != 0 {
			t.Errorf("got len %d, want 0", len(got))
		}
	})

	t.Run("single_hypothesis", func(t *testing.T) {
		t.Parallel()
		// m=1: correction factor is 1 — adjusted p equals raw p.
		in := []float64{0.03}
		got := BonferroniCorrection(in)
		if !approxEqual(got[0], 0.03, floatTolerance) {
			t.Errorf("got %.10f, want 0.03", got[0])
		}
	})

	t.Run("preserves_input_order", func(t *testing.T) {
		t.Parallel()
		// Largest raw p is at index 0; output must keep that order.
		in := []float64{0.04, 0.01, 0.02}
		got := BonferroniCorrection(in)
		// m=3 → want = [0.12, 0.03, 0.06]
		want := []float64{0.12, 0.03, 0.06}
		for i := range want {
			if !approxEqual(got[i], want[i], floatTolerance) {
				t.Errorf("index %d: got %.10f, want %.10f", i, got[i], want[i])
			}
		}
	})
}

func TestHolmBonferroniCorrection(t *testing.T) {
	t.Parallel()

	// Reference values for three_metrics_hand_computed, derived by hand
	// following Holm (1979):
	//   input order:     [0.04, 0.01, 0.03]   (idx 0, 1, 2)
	//   sorted asc:      p_(1)=0.01(idx1), p_(2)=0.03(idx2), p_(3)=0.04(idx0)
	//   step-down mult:  (3)*0.01=0.03,  (2)*0.03=0.06,  (1)*0.04=0.04
	//   monotone max:    0.03,  max(0.03,0.06)=0.06,  max(0.06,0.04)=0.06
	//   restore order:   idx0→0.06, idx1→0.03, idx2→0.06
	t.Run("three_metrics_hand_computed", func(t *testing.T) {
		t.Parallel()
		in := []float64{0.04, 0.01, 0.03}
		got := HolmBonferroniCorrection(in)
		want := []float64{0.06, 0.03, 0.06}
		if len(got) != len(want) {
			t.Fatalf("len=%d, want %d", len(got), len(want))
		}
		for i := range want {
			if !approxEqual(got[i], want[i], floatTolerance) {
				t.Errorf("index %d: got %.10f, want %.10f", i, got[i], want[i])
			}
		}
	})

	t.Run("holm_never_exceeds_bonferroni", func(t *testing.T) {
		t.Parallel()
		// Holm is uniformly more powerful: its adjusted p-values are always
		// ≤ the corresponding Bonferroni adjusted p-values.
		in := []float64{0.04, 0.01, 0.03}
		holm := HolmBonferroniCorrection(in)
		bonf := BonferroniCorrection(in)
		for i := range in {
			if holm[i] > bonf[i]+floatTolerance {
				t.Errorf("Holm[%d]=%.6f > Bonferroni[%d]=%.6f: Holm must never exceed Bonferroni",
					i, holm[i], i, bonf[i])
			}
		}
	})

	t.Run("all_identical_pvalues", func(t *testing.T) {
		t.Parallel()
		// When all raw p-values are equal, Holm and Bonferroni coincide.
		in := []float64{0.02, 0.02, 0.02}
		holm := HolmBonferroniCorrection(in)
		bonf := BonferroniCorrection(in)
		for i := range in {
			if !approxEqual(holm[i], bonf[i], floatTolerance) {
				t.Errorf("equal inputs: Holm[%d]=%.6f != Bonferroni[%d]=%.6f",
					i, holm[i], i, bonf[i])
			}
		}
	})

	t.Run("cap_at_one", func(t *testing.T) {
		t.Parallel()
		// All adjusted p-values must be capped at 1.0.
		in := []float64{0.5, 0.6, 0.7}
		got := HolmBonferroniCorrection(in)
		for i, v := range got {
			if v > 1.0+floatTolerance {
				t.Errorf("index %d: adjusted p=%.6f exceeds 1.0", i, v)
			}
		}
	})

	t.Run("empty_input", func(t *testing.T) {
		t.Parallel()
		got := HolmBonferroniCorrection([]float64{})
		if got != nil {
			t.Errorf("got %v, want nil for empty input", got)
		}
	})

	t.Run("single_hypothesis", func(t *testing.T) {
		t.Parallel()
		// m=1: Holm correction factor is 1 — adjusted p equals raw p.
		in := []float64{0.04}
		got := HolmBonferroniCorrection(in)
		if !approxEqual(got[0], 0.04, floatTolerance) {
			t.Errorf("got %.10f, want 0.04", got[0])
		}
	})

	t.Run("preserves_input_order", func(t *testing.T) {
		t.Parallel()
		// The smallest raw p is at index 1; after correction the output at
		// index 1 must still be the smallest adjusted p (no reordering).
		in := []float64{0.04, 0.01, 0.03}
		got := HolmBonferroniCorrection(in)
		minRawIdx := 1 // 0.01 is the smallest raw p
		for i := range in {
			if i != minRawIdx && got[i] < got[minRawIdx]-floatTolerance {
				t.Errorf("index %d (p̃=%.4f) has lower adjusted p than minRaw index %d (p̃=%.4f)",
					i, got[i], minRawIdx, got[minRawIdx])
			}
		}
	})

	t.Run("four_metrics_llmbench_family", func(t *testing.T) {
		t.Parallel()
		// Reproduces the actual benchmark family: ESR, TSA, CHR, LatP50.
		// Raw p-values are chosen so the step-down correction changes at
		// least one significance decision vs. the uncorrected α=0.05 —
		// this is the statistical argument reported in the paper.
		//
		// Sorted ascending:
		//   p_(1)=0.008 (TSA, idx1), p_(2)=0.012 (ESR, idx0),
		//   p_(3)=0.040 (LatP50, idx3), p_(4)=0.090 (CHR, idx2)
		//
		// Step-down × monotone max:
		//   rank1: 4*0.008=0.032
		//   rank2: max(0.032, 3*0.012)=max(0.032,0.036)=0.036
		//   rank3: max(0.036, 2*0.040)=max(0.036,0.080)=0.080
		//   rank4: max(0.080, 1*0.090)=max(0.080,0.090)=0.090
		//
		// Restored to input order:
		//   ESR(idx0)=0.036, TSA(idx1)=0.032, CHR(idx2)=0.090, LatP50(idx3)=0.080
		in := []float64{0.012, 0.008, 0.090, 0.040}
		got := HolmBonferroniCorrection(in)
		want := []float64{0.036, 0.032, 0.090, 0.080}
		for i := range want {
			if !approxEqual(got[i], want[i], floatTolerance) {
				t.Errorf("metric[%d]: got %.10f, want %.10f", i, got[i], want[i])
			}
		}

		// ESR (idx 0) and TSA (idx 1) must remain significant at α=0.05.
		if got[0] >= 0.05 {
			t.Errorf("ESR adjusted p=%.4f should be <0.05 after Holm", got[0])
		}
		if got[1] >= 0.05 {
			t.Errorf("TSA adjusted p=%.4f should be <0.05 after Holm", got[1])
		}
		// CHR raw was 0.09 > α; must remain non-significant after correction.
		if got[2] < 0.05 {
			t.Errorf("CHR adjusted p=%.4f should be >=0.05 (n.s.) after Holm", got[2])
		}
		// LatP50 raw was exactly α=0.040; after step-up it must be non-significant.
		if got[3] < 0.05 {
			t.Errorf("LatP50 adjusted p=%.4f should be >=0.05 after Holm", got[3])
		}
	})
}
