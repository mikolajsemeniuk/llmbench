package llmbench

import (
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// EvaluateResponse
// ---------------------------------------------------------------------------

func TestEvaluateResponse(t *testing.T) {
	t.Parallel()

	gt := GroundTruth{
		DiagnosisGroups: [][]string{
			{"crashloopbackoff", "crash loop"},
			{"restart", "exit", "error"},
		},
		ActionTerms: []string{"kubectl logs", "kubectl describe"},
		ContextEntities: map[string]string{
			"pod_name":  "nginx",
			"namespace": "default",
			"state":     "CrashLoopBackOff",
		},
		ForbiddenPatterns: []string{"delete namespace", "delete --all"},
	}

	t.Run("full_pass", func(t *testing.T) {
		t.Parallel()
		response := "The Pod nginx in namespace default is in CrashLoopBackOff state. " +
			"It has restarted 8 times with exit code 1. " +
			"Run kubectl logs nginx to inspect the error."
		r := EvaluateResponse(response, gt)
		if !r.DiagnosisCorrect {
			t.Error("expected DiagnosisCorrect=true")
		}
		if !r.ActionCorrect {
			t.Error("expected ActionCorrect=true")
		}
		if r.HallucinatedArgs != 0 {
			t.Errorf("expected 0 hallucinated args, got %d", r.HallucinatedArgs)
		}
		if r.DestructiveHit {
			t.Error("expected DestructiveHit=false")
		}
	})

	t.Run("diagnosis_missing_one_group", func(t *testing.T) {
		t.Parallel()
		response := "The Pod nginx in default namespace is in CrashLoopBackOff. " +
			"Run kubectl logs nginx."
		r := EvaluateResponse(response, gt)
		if r.DiagnosisCorrect {
			t.Error("expected DiagnosisCorrect=false (second group not matched)")
		}
		if !r.ActionCorrect {
			t.Error("expected ActionCorrect=true")
		}
	})

	t.Run("action_missing", func(t *testing.T) {
		t.Parallel()
		response := "The Pod nginx in default is in CrashLoopBackOff with restart errors."
		r := EvaluateResponse(response, gt)
		if !r.DiagnosisCorrect {
			t.Error("expected DiagnosisCorrect=true")
		}
		if r.ActionCorrect {
			t.Error("expected ActionCorrect=false")
		}
	})

	t.Run("hallucinated_entities", func(t *testing.T) {
		t.Parallel()
		response := "The Pod my-app in staging namespace has errors and keeps restarting. " +
			"It shows CrashLoopBackOff. Run kubectl logs my-app."
		r := EvaluateResponse(response, gt)
		// "nginx" missing, "default" missing, "CrashLoopBackOff" present → 2 hallucinated out of 3
		if r.HallucinatedArgs != 2 {
			t.Errorf("expected 2 hallucinated args, got %d", r.HallucinatedArgs)
		}
		if r.TotalArgs != 3 {
			t.Errorf("expected 3 total args, got %d", r.TotalArgs)
		}
	})

	t.Run("destructive_action_detected", func(t *testing.T) {
		t.Parallel()
		response := "The Pod nginx in default is in CrashLoopBackOff with restart errors. " +
			"Fix it by running: kubectl delete namespace default."
		r := EvaluateResponse(response, gt)
		if !r.DestructiveHit {
			t.Error("expected DestructiveHit=true")
		}
	})

	t.Run("case_insensitive_matching", func(t *testing.T) {
		t.Parallel()
		response := "CRASHLOOPBACKOFF detected on pod NGINX in DEFAULT namespace. " +
			"The container keeps RESTARTING. Use KUBECTL LOGS to debug."
		r := EvaluateResponse(response, gt)
		if !r.DiagnosisCorrect {
			t.Error("expected DiagnosisCorrect=true (case insensitive)")
		}
		if !r.ActionCorrect {
			t.Error("expected ActionCorrect=true (case insensitive)")
		}
		if r.HallucinatedArgs != 0 {
			t.Errorf("expected 0 hallucinated, got %d (case insensitive)", r.HallucinatedArgs)
		}
	})

	t.Run("empty_response", func(t *testing.T) {
		t.Parallel()
		r := EvaluateResponse("", gt)
		if r.DiagnosisCorrect {
			t.Error("expected DiagnosisCorrect=false for empty response")
		}
		if r.ActionCorrect {
			t.Error("expected ActionCorrect=false for empty response")
		}
		if r.HallucinatedArgs != 3 {
			t.Errorf("expected 3 hallucinated, got %d", r.HallucinatedArgs)
		}
	})

	t.Run("empty_ground_truth", func(t *testing.T) {
		t.Parallel()
		r := EvaluateResponse("some response", GroundTruth{})
		if !r.DiagnosisCorrect {
			t.Error("expected DiagnosisCorrect=true (no groups to match)")
		}
		if r.ActionCorrect {
			t.Error("expected ActionCorrect=false (no action terms)")
		}
		if r.HallucinatedArgs != 0 {
			t.Errorf("expected 0 hallucinated (no entities), got %d", r.HallucinatedArgs)
		}
	})

	t.Run("response_length_recorded", func(t *testing.T) {
		t.Parallel()
		response := "short"
		r := EvaluateResponse(response, GroundTruth{})
		if r.ResponseLen != 5 {
			t.Errorf("expected ResponseLen=5, got %d", r.ResponseLen)
		}
	})
}

// ---------------------------------------------------------------------------
// BuildPrompt
// ---------------------------------------------------------------------------

func TestBuildPrompt(t *testing.T) {
	t.Parallel()

	task := Task{
		ID:          "test-001",
		Level:       LevelDiagnostic,
		Description: "Find the problem.",
		Documents: []RAGDocument{
			{ID: "doc-1", Content: "apiVersion: v1\nkind: Pod", Relevance: 3},
			{ID: "doc-2", Content: "apiVersion: v1\nkind: ConfigMap", Relevance: 0},
		},
	}

	t.Run("contains_all_documents", func(t *testing.T) {
		t.Parallel()
		prompt := BuildPrompt(task)
		if !strings.Contains(prompt, "kind: Pod") {
			t.Error("prompt missing first document content")
		}
		if !strings.Contains(prompt, "kind: ConfigMap") {
			t.Error("prompt missing second document content")
		}
	})

	t.Run("contains_task_description", func(t *testing.T) {
		t.Parallel()
		prompt := BuildPrompt(task)
		if !strings.Contains(prompt, "Find the problem.") {
			t.Error("prompt missing task description")
		}
	})

	t.Run("contains_structure_markers", func(t *testing.T) {
		t.Parallel()
		prompt := BuildPrompt(task)
		if !strings.Contains(prompt, "=== CLUSTER STATE ===") {
			t.Error("prompt missing CLUSTER STATE marker")
		}
		if !strings.Contains(prompt, "=== END STATE ===") {
			t.Error("prompt missing END STATE marker")
		}
		if !strings.Contains(prompt, "TASK:") {
			t.Error("prompt missing TASK: prefix")
		}
	})

	t.Run("contains_response_format", func(t *testing.T) {
		t.Parallel()
		prompt := BuildPrompt(task)
		if !strings.Contains(prompt, "1. DIAGNOSIS:") {
			t.Error("prompt missing DIAGNOSIS instruction")
		}
		if !strings.Contains(prompt, "2. ROOT CAUSE:") {
			t.Error("prompt missing ROOT CAUSE instruction")
		}
		if !strings.Contains(prompt, "3. FIX:") {
			t.Error("prompt missing FIX instruction")
		}
	})

	t.Run("documents_separated_by_delimiter", func(t *testing.T) {
		t.Parallel()
		prompt := BuildPrompt(task)
		if !strings.Contains(prompt, "---") {
			t.Error("documents not separated by --- delimiter")
		}
	})
}

// ---------------------------------------------------------------------------
// ComputeTaskRAGMetrics
// ---------------------------------------------------------------------------

func TestComputeTaskRAGMetrics(t *testing.T) {
	t.Parallel()

	t.Run("primary_first_perfect_ranking", func(t *testing.T) {
		t.Parallel()
		task := Task{
			Documents: []RAGDocument{
				{Relevance: 3},
				{Relevance: 0},
			},
		}
		p, r, mrr, ndcg := ComputeTaskRAGMetrics(task)
		if !approxEqual(p, 0.5, floatTolerance) {
			t.Errorf("P@K: got %f, want 0.5", p)
		}
		if !approxEqual(r, 1.0, floatTolerance) {
			t.Errorf("R@K: got %f, want 1.0", r)
		}
		if !approxEqual(mrr, 1.0, floatTolerance) {
			t.Errorf("MRR: got %f, want 1.0", mrr)
		}
		if !approxEqual(ndcg, 1.0, floatTolerance) {
			t.Errorf("NDCG@K: got %f, want 1.0", ndcg)
		}
	})

	t.Run("noise_first_penalizes_ndcg", func(t *testing.T) {
		t.Parallel()
		task := Task{
			Documents: []RAGDocument{
				{Relevance: 0},
				{Relevance: 3},
			},
		}
		p, _, mrr, ndcg := ComputeTaskRAGMetrics(task)
		if !approxEqual(p, 0.5, floatTolerance) {
			t.Errorf("P@K: got %f, want 0.5", p)
		}
		if !approxEqual(mrr, 0.5, floatTolerance) {
			t.Errorf("MRR: got %f, want 0.5 (relevant at rank 2)", mrr)
		}
		if ndcg >= 1.0 {
			t.Errorf("NDCG@K: got %f, expected < 1.0 (non-ideal ranking)", ndcg)
		}
	})

	t.Run("all_relevant", func(t *testing.T) {
		t.Parallel()
		task := Task{
			Documents: []RAGDocument{
				{Relevance: 3},
				{Relevance: 1},
			},
		}
		p, _, _, _ := ComputeTaskRAGMetrics(task)
		if !approxEqual(p, 1.0, floatTolerance) {
			t.Errorf("P@K: got %f, want 1.0 (all relevant)", p)
		}
	})

	t.Run("empty_documents", func(t *testing.T) {
		t.Parallel()
		task := Task{Documents: nil}
		p, r, mrr, ndcg := ComputeTaskRAGMetrics(task)
		if p != 0 || r != 0 || mrr != 0 || ndcg != 0 {
			t.Errorf("expected all zeros for empty docs, got P=%f R=%f MRR=%f NDCG=%f", p, r, mrr, ndcg)
		}
	})

	t.Run("three_docs_with_noise_in_middle", func(t *testing.T) {
		t.Parallel()
		task := Task{
			Documents: []RAGDocument{
				{Relevance: 3},
				{Relevance: 0},
				{Relevance: 3},
			},
		}
		p, _, mrr, _ := ComputeTaskRAGMetrics(task)
		// 2 relevant out of 3
		if !approxEqual(p, 2.0/3.0, floatTolerance) {
			t.Errorf("P@K: got %f, want %f", p, 2.0/3.0)
		}
		if !approxEqual(mrr, 1.0, floatTolerance) {
			t.Errorf("MRR: got %f, want 1.0 (first doc is relevant)", mrr)
		}
	})
}

// ---------------------------------------------------------------------------
// BenchmarkTasks — structural validation
// ---------------------------------------------------------------------------

func TestBenchmarkTasks(t *testing.T) {
	t.Parallel()

	tasks := BenchmarkTasks()

	t.Run("returns_eight_tasks", func(t *testing.T) {
		t.Parallel()
		if len(tasks) != 8 {
			t.Errorf("got %d tasks, want 8", len(tasks))
		}
	})

	t.Run("three_l1_tasks", func(t *testing.T) {
		t.Parallel()
		count := 0
		for _, task := range tasks {
			if task.Level == LevelDiagnostic {
				count++
			}
		}
		if count != 3 {
			t.Errorf("got %d L1 tasks, want 3", count)
		}
	})

	t.Run("three_l2_tasks", func(t *testing.T) {
		t.Parallel()
		count := 0
		for _, task := range tasks {
			if task.Level == LevelRepair {
				count++
			}
		}
		if count != 3 {
			t.Errorf("got %d L2 tasks, want 3", count)
		}
	})

	t.Run("two_l3_tasks", func(t *testing.T) {
		t.Parallel()
		count := 0
		for _, task := range tasks {
			if task.Level == LevelMultiStep {
				count++
			}
		}
		if count != 2 {
			t.Errorf("got %d L3 tasks, want 2", count)
		}
	})

	t.Run("unique_task_ids", func(t *testing.T) {
		t.Parallel()
		seen := make(map[string]bool)
		for _, task := range tasks {
			if seen[task.ID] {
				t.Errorf("duplicate task ID: %s", task.ID)
			}
			seen[task.ID] = true
		}
	})

	t.Run("all_tasks_have_documents", func(t *testing.T) {
		t.Parallel()
		for _, task := range tasks {
			if len(task.Documents) == 0 {
				t.Errorf("task %s has no documents", task.ID)
			}
		}
	})

	t.Run("all_tasks_have_ground_truth", func(t *testing.T) {
		t.Parallel()
		for _, task := range tasks {
			if len(task.GroundTruth.DiagnosisGroups) == 0 {
				t.Errorf("task %s has no DiagnosisGroups", task.ID)
			}
			if len(task.GroundTruth.ActionTerms) == 0 {
				t.Errorf("task %s has no ActionTerms", task.ID)
			}
			if len(task.GroundTruth.ContextEntities) == 0 {
				t.Errorf("task %s has no ContextEntities", task.ID)
			}
		}
	})

	t.Run("every_task_has_noise_document", func(t *testing.T) {
		t.Parallel()
		for _, task := range tasks {
			hasNoise := false
			for _, doc := range task.Documents {
				if doc.Relevance == 0 {
					hasNoise = true
					break
				}
			}
			if !hasNoise {
				t.Errorf("task %s has no noise document (relevance=0)", task.ID)
			}
		}
	})

	t.Run("all_tasks_have_description", func(t *testing.T) {
		t.Parallel()
		for _, task := range tasks {
			if task.Description == "" {
				t.Errorf("task %s has empty description", task.ID)
			}
		}
	})
}
