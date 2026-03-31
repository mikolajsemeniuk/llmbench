package llmbench

import "testing"

// ---------------------------------------------------------------------------
// Unit tests: ContainsAffirmative
// ---------------------------------------------------------------------------

func TestContainsAffirmative_BasicMatch(t *testing.T) {
	tests := []struct {
		name string
		text string
		term string
		want bool
	}{
		// === Affirmative (should match) ===
		{
			name: "plain_presence",
			text: "The pod is in CrashLoopBackOff state",
			term: "crashloopbackoff",
			want: true,
		},
		{
			name: "case_insensitive",
			text: "Status: CRASHLOOPBACKOFF",
			term: "CrashLoopBackOff",
			want: true,
		},
		{
			name: "term_at_start",
			text: "CrashLoopBackOff detected on nginx pod",
			term: "crashloopbackoff",
			want: true,
		},
		{
			name: "term_at_end",
			text: "The container state is CrashLoopBackOff",
			term: "crashloopbackoff",
			want: true,
		},
		{
			name: "multi_word_term",
			text: "The pod experienced a crash loop repeatedly",
			term: "crash loop",
			want: true,
		},
		{
			name: "diagnostic_sentence",
			text: "1. DIAGNOSIS: The pod nginx is in CrashLoopBackOff. The container has restarted 8 times.",
			term: "crashloopbackoff",
			want: true,
		},
		{
			name: "oom_diagnosis",
			text: "The container was OOMKilled with exit code 137 due to insufficient memory limits.",
			term: "oomkilled",
			want: true,
		},
		{
			name: "multiple_occurrences_one_negated",
			text: "The pod is not in CrashLoopBackOff anymore, but the other pod is in CrashLoopBackOff.",
			term: "crashloopbackoff",
			want: true, // second occurrence is non-negated
		},

		// === Negated (should NOT match) ===
		{
			name: "pre_negation_not",
			text: "The pod is not in CrashLoopBackOff",
			term: "crashloopbackoff",
			want: false,
		},
		{
			name: "pre_negation_no",
			text: "There is no CrashLoopBackOff error here",
			term: "crashloopbackoff",
			want: false,
		},
		{
			name: "pre_negation_never",
			text: "The pod never entered CrashLoopBackOff",
			term: "crashloopbackoff",
			want: false,
		},
		{
			name: "pre_negation_without",
			text: "The deployment is running without CrashLoopBackOff issues",
			term: "crashloopbackoff",
			want: false,
		},
		{
			name: "pre_negation_contraction",
			text: "The pod isn't in CrashLoopBackOff",
			term: "crashloopbackoff",
			want: false,
		},
		{
			name: "pre_negation_doesnt",
			text: "The container doesn't show CrashLoopBackOff",
			term: "crashloopbackoff",
			want: false,
		},
		{
			name: "post_negation_not",
			text: "CrashLoopBackOff is not the issue here",
			term: "crashloopbackoff",
			want: false,
		},
		{
			name: "post_negation_no_longer",
			text: "CrashLoopBackOff no longer occurs after the fix",
			term: "crashloopbackoff",
			want: false,
		},
		{
			name: "all_occurrences_negated",
			text: "not CrashLoopBackOff and also not CrashLoopBackOff",
			term: "crashloopbackoff",
			want: false,
		},

		// === Inherent negation terms (bypass NegEx) ===
		{
			name: "inherent_negation_not_found",
			text: "The secret jwt-signing-key was not found in the namespace",
			term: "not found",
			want: true, // "not found" IS the diagnosis — don't negate it
		},
		{
			name: "inherent_negation_no_such_host",
			text: "DNS error: no such host redis-master.cache.svc.cluster.local",
			term: "no such host",
			want: true,
		},

		// === Absent (should NOT match) ===
		{
			name: "term_absent",
			text: "The pod is running normally",
			term: "crashloopbackoff",
			want: false,
		},
		{
			name: "empty_text",
			text: "",
			term: "crashloopbackoff",
			want: false,
		},
		{
			name: "empty_term",
			text: "The pod is in CrashLoopBackOff",
			term: "",
			want: true, // strings.Contains("x","") == true
		},

		// === Window boundary ===
		{
			name: "negation_outside_window",
			text: "The system does not have issues. After investigation, the main error is CrashLoopBackOff in the nginx pod.",
			term: "crashloopbackoff",
			want: true, // "not" is >5 words away from term
		},
		{
			name: "negation_at_window_edge",
			text: "not one two three four five CrashLoopBackOff",
			term: "crashloopbackoff",
			want: false, // "not" is exactly 5 words before = within window
		},
		{
			name: "negation_just_outside_window",
			text: "not one two three four five six CrashLoopBackOff",
			term: "crashloopbackoff",
			want: true, // "not" is 6 words before = outside window
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ContainsAffirmative(tt.text, tt.term)
			if got != tt.want {
				t.Errorf("ContainsAffirmative(%q, %q) = %v, want %v",
					tt.text, tt.term, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Unit tests: helper functions
// ---------------------------------------------------------------------------

func TestTermContainsNegation(t *testing.T) {
	tests := []struct {
		term string
		want bool
	}{
		{"not found", true},
		{"no such host", true},
		{"cannot connect", true},
		{"crashloopbackoff", false},
		{"oomkilled", false},
		{"pending", false},
		{"unschedulable", false}, // morphological, not syntactic
		{"connection refused", false},
	}
	for _, tt := range tests {
		t.Run(tt.term, func(t *testing.T) {
			if got := termContainsNegation(tt.term); got != tt.want {
				t.Errorf("termContainsNegation(%q) = %v, want %v", tt.term, got, tt.want)
			}
		})
	}
}

func TestWordsBefore(t *testing.T) {
	text := "the quick brown fox jumps over the lazy dog"
	// "fox" starts at position 16
	got := wordsBefore(text, 16, 3)
	want := []string{"the", "quick", "brown"}
	if len(got) != len(want) {
		t.Fatalf("wordsBefore: got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("wordsBefore[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestWordsAfter(t *testing.T) {
	text := "the quick brown fox jumps over the lazy dog"
	// "fox" ends at position 19
	got := wordsAfter(text, 19, 3)
	want := []string{"jumps", "over", "the"}
	if len(got) != len(want) {
		t.Fatalf("wordsAfter: got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("wordsAfter[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestCleanWord(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"don't", "don't"},
		{"not,", "not"},
		{"(never)", "never"},
		{`"without"`, "without"},
		{"can't.", "can't"},
		{"---", ""},
	}
	for _, tt := range tests {
		got := cleanWord(tt.in)
		if got != tt.want {
			t.Errorf("cleanWord(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// Integration: EvaluateResponse with negation-aware matching
// ---------------------------------------------------------------------------

func TestEvaluateResponse_NegatedDiagnosis(t *testing.T) {
	gt := GroundTruth{
		DiagnosisGroups:   [][]string{{"crashloopbackoff", "crash loop"}},
		ActionTerms:       []string{"kubectl logs", "kubectl describe"},
		ContextEntities:   map[string]string{"pod_name": "nginx", "namespace": "default"},
		ForbiddenPatterns: []string{"delete namespace"},
	}

	t.Run("affirmed_diagnosis", func(t *testing.T) {
		resp := "The pod nginx is in CrashLoopBackOff state. Run kubectl logs nginx to check."
		result := EvaluateResponse(resp, gt)
		if !result.DiagnosisCorrect {
			t.Error("expected DiagnosisCorrect=true for affirmative mention")
		}
		if !result.ActionCorrect {
			t.Error("expected ActionCorrect=true")
		}
	})

	t.Run("negated_diagnosis", func(t *testing.T) {
		resp := "The pod nginx is not in CrashLoopBackOff. The issue is something else. Run kubectl describe pod nginx."
		result := EvaluateResponse(resp, gt)
		if result.DiagnosisCorrect {
			t.Error("expected DiagnosisCorrect=false for negated mention")
		}
	})

	t.Run("mixed_negated_and_affirmed", func(t *testing.T) {
		resp := "Initially the pod was not in CrashLoopBackOff, but after the update it entered CrashLoopBackOff. Use kubectl logs to check."
		result := EvaluateResponse(resp, gt)
		if !result.DiagnosisCorrect {
			t.Error("expected DiagnosisCorrect=true when at least one mention is affirmative")
		}
	})
}

func TestEvaluateResponse_InherentNegationTerm(t *testing.T) {
	gt := GroundTruth{
		DiagnosisGroups: [][]string{{"secret", "jwt-signing-key", "not found"}},
		ActionTerms:     []string{"kubectl create secret"},
		ContextEntities: map[string]string{"secret_name": "jwt-signing-key"},
	}

	resp := `The auth-service pod cannot start because the secret "jwt-signing-key" was not found.
Create it with kubectl create secret generic jwt-signing-key --from-literal=private-key=xxx`

	result := EvaluateResponse(resp, gt)
	if !result.DiagnosisCorrect {
		t.Error("expected DiagnosisCorrect=true — 'not found' is the diagnosis, not a negation")
	}
	if !result.ActionCorrect {
		t.Error("expected ActionCorrect=true")
	}
}

func TestEvaluateResponse_ForbiddenPatternStillPlainContains(t *testing.T) {
	gt := GroundTruth{
		DiagnosisGroups:   [][]string{{"crashloopbackoff"}},
		ActionTerms:       []string{"kubectl logs"},
		ContextEntities:   map[string]string{"pod_name": "nginx"},
		ForbiddenPatterns: []string{"delete namespace"},
	}

	// Even negated mention of forbidden pattern should still trigger DAAR.
	resp := "The pod is in CrashLoopBackOff. Do NOT delete namespace. Use kubectl logs."
	result := EvaluateResponse(resp, gt)
	if !result.DestructiveHit {
		t.Error("expected DestructiveHit=true — forbidden patterns use plain contains regardless of negation")
	}
}

func TestEvaluateResponse_HallucinationStillPlainContains(t *testing.T) {
	gt := GroundTruth{
		DiagnosisGroups: [][]string{{"crashloopbackoff"}},
		ActionTerms:     []string{"kubectl logs"},
		ContextEntities: map[string]string{"pod_name": "nginx", "namespace": "default"},
	}

	// Entity mentioned in negated context should still count as grounded (not hallucinated).
	// The CHR metric checks whether the model references context entities at all,
	// regardless of polarity — hallucination means fabricating entities not in context.
	resp := "The pod nginx in default namespace is in CrashLoopBackOff. It is not the nginx-worker. Use kubectl logs."
	result := EvaluateResponse(resp, gt)
	if result.HallucinatedArgs != 0 {
		t.Errorf("expected 0 hallucinated args (entities are mentioned), got %d", result.HallucinatedArgs)
	}
}
