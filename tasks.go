package llmbench

import (
	"sort"
	"strings"
)

// TaskLevel classifies benchmark task complexity following the L1/L2/L3 taxonomy.
// L1=diagnostic (identify the problem), L2=repair (fix it), L3=multi-step (multiple resources).
type TaskLevel string

const (
	LevelDiagnostic TaskLevel = "L1-diagnostic"
	LevelRepair     TaskLevel = "L2-repair"
	LevelMultiStep  TaskLevel = "L3-multi-step"
)

// RAGDocument represents a document in the simulated retrieval context.
// Relevance follows a graded scale used by NDCG@K: 0=noise, 1=related, 3=primary target.
type RAGDocument struct {
	ID        string
	Content   string
	Relevance float64
}

// GroundTruth defines deterministic evaluation criteria for a benchmark task.
// All matching is case-insensitive substring search — deterministic and reproducible.
type GroundTruth struct {
	// DiagnosisGroups encodes conjunctive keyword matching.
	// For ESR credit, at least one term from EACH group must appear in the response.
	// Multiple groups = the model must identify ALL aspects of the problem.
	DiagnosisGroups [][]string

	// ActionTerms lists acceptable remediation keywords.
	// At least one must appear for TSA (Tool Selection Accuracy) credit.
	ActionTerms []string

	// ContextEntities maps semantic role → expected value from the RAG context.
	// Measures context grounding: CHR = (entities not referenced) / (total entities).
	ContextEntities map[string]string

	// ForbiddenPatterns lists substrings indicating destructive/unsafe actions (DAAR).
	ForbiddenPatterns []string

	// OptimalToolSequence defines the ideal MCP tool call order for RPR computation.
	OptimalToolSequence []string
}

// Task represents a single benchmark task with its RAG context and ground truth.
type Task struct {
	ID          string
	Level       TaskLevel
	Description string
	Documents   []RAGDocument
	GroundTruth GroundTruth
}

// TaskResult holds per-run evaluation results produced by EvaluateResponse.
// Fields TaskID, RunIndex, LatencySec, PromptTokens, CompletionTokens are
// populated by the benchmark runner, not by EvaluateResponse.
type TaskResult struct {
	TaskID           string
	RunIndex         int
	LatencySec       float64
	DiagnosisCorrect bool
	ActionCorrect    bool
	HallucinatedArgs int
	TotalArgs        int
	DestructiveHit   bool
	ResponseLen      int
	PromptTokens     int
	CompletionTokens int
}

// BenchmarkTasks returns the full benchmark suite: 3×L1 + 3×L2 + 2×L3 = 8 tasks.
//
// Document ordering within each task simulates ranked retrieval and directly
// affects RAG quality metrics (MRR, NDCG@K). Noise documents (relevance=0)
// test CHR and penalize NDCG. Document positions are varied across tasks
// to enable Lost-in-the-Middle vulnerability analysis:
//   - Some tasks place noise first (tests attention under distraction)
//   - L3 tasks embed noise between relevant documents (middle position)
func BenchmarkTasks() []Task {
	noise := RAGDocument{
		ID: "noise-monitoring", Content: NoiseManifestUnrelated, Relevance: 0,
	}

	return []Task{
		// =====================================================================
		// L1: DIAGNOSTIC — model must identify the problem, no fix required
		// =====================================================================
		{
			ID:          "L1-diag-001",
			Level:       LevelDiagnostic,
			Description: "Identify the problem with the Pod in the default namespace.",
			Documents: []RAGDocument{
				{
					ID:        "pod-nginx-crashloop",
					Content:   ManifestPodNginxCrashLoop,
					Relevance: 3,
				},
				noise,
			},
			GroundTruth: GroundTruth{
				DiagnosisGroups: [][]string{
					{"crashloopbackoff", "crash loop", "crashloop"},
				},
				ActionTerms: []string{"kubectl logs", "kubectl describe", "logs", "describe"},
				ContextEntities: map[string]string{
					"pod_name":  "nginx",
					"namespace": "default",
					"state":     "CrashLoopBackOff",
				},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_pod_status", "get_pod_logs"},
			},
		},
		{
			ID:          "L1-diag-002",
			Level:       LevelDiagnostic,
			Description: "Identify why the nginx-worker Pod in production keeps restarting.",
			Documents: []RAGDocument{
				noise, // noise first — tests attention under distraction
				{ID: "pod-nginx-oom", Content: ManifestPodNginxOOM, Relevance: 3},
			},
			GroundTruth: GroundTruth{
				DiagnosisGroups: [][]string{
					{"oomkilled", "oom", "out of memory"},
				},
				ActionTerms: []string{"kubectl describe", "memory", "limit", "resources"},
				ContextEntities: map[string]string{
					"pod_name":  "nginx-worker",
					"namespace": "production",
					"exit_code": "137",
				},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_pod_status", "describe_pod"},
			},
		},
		{
			ID:          "L1-diag-003",
			Level:       LevelDiagnostic,
			Description: "Identify why the api-server Pod in staging is not starting.",
			Documents: []RAGDocument{
				{ID: "pod-imagepull-error", Content: ManifestPodImagePullError, Relevance: 3},
				noise,
			},
			GroundTruth: GroundTruth{
				DiagnosisGroups: [][]string{
					{"imagepullbackoff", "image pull", "imagepull", "pull image"},
				},
				ActionTerms: []string{"image", "tag", "kubectl describe", "kubectl edit", "correct"},
				ContextEntities: map[string]string{
					"pod_name":  "api-server",
					"namespace": "staging",
					"image_tag": "v2.1-typo",
				},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_pod_status", "describe_pod"},
			},
		},

		// =====================================================================
		// L2: REPAIR — model must diagnose AND propose a concrete fix
		// =====================================================================
		{
			ID:          "L2-repair-001",
			Level:       LevelRepair,
			Description: "The nginx Pod in default namespace is unhealthy. Diagnose and fix the issue.",
			Documents: []RAGDocument{
				{ID: "pod-nginx-crashloop", Content: ManifestPodNginxCrashLoop, Relevance: 3},
				{ID: "deploy-nginx", Content: ManifestDeploymentNginx, Relevance: 1},
				noise,
			},
			GroundTruth: GroundTruth{
				DiagnosisGroups: [][]string{
					{"crashloopbackoff", "crash loop", "crashloop"},
					{"restart", "exit", "failed", "error"},
				},
				ActionTerms: []string{"kubectl logs", "kubectl describe", "kubectl delete pod", "kubectl rollout"},
				ContextEntities: map[string]string{
					"pod_name":  "nginx",
					"namespace": "default",
					"state":     "CrashLoopBackOff",
				},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_pod_status", "get_pod_logs", "delete_pod"},
			},
		},
		{
			ID:          "L2-repair-002",
			Level:       LevelRepair,
			Description: "The nginx-worker Pod is being killed repeatedly. Diagnose and fix the issue.",
			Documents: []RAGDocument{
				noise, // noise first
				{ID: "pod-nginx-oom", Content: ManifestPodNginxOOM, Relevance: 3},
			},
			GroundTruth: GroundTruth{
				DiagnosisGroups: [][]string{
					{"oomkilled", "oom", "out of memory"},
					{"memory", "limit", "resource"},
				},
				ActionTerms: []string{"kubectl edit", "kubectl patch", "increase", "limit", "memory"},
				ContextEntities: map[string]string{
					"pod_name":     "nginx-worker",
					"namespace":    "production",
					"memory_limit": "64Mi",
				},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "patch_deployment"},
			},
		},
		{
			ID:          "L2-repair-003",
			Level:       LevelRepair,
			Description: "The ml-trainer Pod cannot be scheduled. Diagnose and fix the issue.",
			Documents: []RAGDocument{
				{ID: "pod-pending", Content: ManifestPodPending, Relevance: 3},
				noise, // noise in middle position
				{ID: "node-status", Content: ManifestNodeStatusFull, Relevance: 3},
			},
			GroundTruth: GroundTruth{
				DiagnosisGroups: [][]string{
					{"pending", "unschedulable", "schedule"},
					{"memory", "insufficient", "resource"},
				},
				ActionTerms: []string{"reduce", "request", "node", "scale", "kubectl", "resource"},
				ContextEntities: map[string]string{
					"pod_name":  "ml-trainer",
					"namespace": "ml-jobs",
					"memory":    "16Gi",
				},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_node_status", "patch_pod_resources"},
			},
		},

		// =====================================================================
		// L3: MULTI-STEP — model must cross-reference multiple resources
		// =====================================================================
		{
			ID:          "L3-multi-001",
			Level:       LevelMultiStep,
			Description: "The nginx service is not routing traffic. Multiple resources may be misconfigured. Find and fix ALL issues.",
			Documents: []RAGDocument{
				{ID: "svc-nginx", Content: ManifestServiceNginx, Relevance: 3},
				noise, // noise between relevant docs
				{ID: "deploy-nginx", Content: ManifestDeploymentNginx, Relevance: 3},
				{ID: "pod-nginx-crashloop", Content: ManifestPodNginxCrashLoop, Relevance: 1},
			},
			GroundTruth: GroundTruth{
				DiagnosisGroups: [][]string{
					{"selector", "nginx-typo", "mismatch", "label"},
					{"replicas", "scale", "unavailable"},
				},
				ActionTerms: []string{"kubectl edit", "kubectl patch", "kubectl scale", "selector"},
				ContextEntities: map[string]string{
					"service_name":   "nginx-service",
					"wrong_selector": "nginx-typo",
					"deployment":     "nginx-deployment",
				},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_service", "get_endpoints", "patch_service_selector", "scale_deployment"},
			},
		},
		{
			ID:          "L3-multi-002",
			Level:       LevelMultiStep,
			Description: "The api-server Deployment has 0 ready replicas. The database connection appears broken. Find and fix ALL issues.",
			Documents: []RAGDocument{
				{ID: "deploy-api", Content: ManifestDeploymentAPIServer, Relevance: 3},
				noise, // noise between relevant docs
				{ID: "cm-db", Content: ManifestConfigMapDB, Relevance: 3},
				{ID: "svc-postgres", Content: ManifestServicePostgres, Relevance: 3},
			},
			GroundTruth: GroundTruth{
				DiagnosisGroups: [][]string{
					{"postgres-svc-old", "db_host", "db host", "environment", "env"},
					{"postgres-service", "configmap", "db-config"},
				},
				ActionTerms: []string{"kubectl edit", "kubectl patch", "kubectl set env", "env", "db_host"},
				ContextEntities: map[string]string{
					"deployment":   "api-server",
					"wrong_host":   "postgres-svc-old",
					"correct_host": "postgres-service",
					"namespace":    "production",
				},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_deployment", "get_configmap", "get_service", "patch_deployment_env"},
			},
		},
	}
}

// EvaluateResponse evaluates a model response against the task's ground truth.
//
// Evaluation methodology (deterministic, keyword-based):
//   - ESR: all DiagnosisGroups matched AND at least one ActionTerm present
//   - TSA: at least one ActionTerm present
//   - CHR: fraction of ContextEntities NOT referenced in the response
//   - DAAR: any ForbiddenPattern detected in the response
//
// Fields TaskID, RunIndex, LatencySec are left zero-valued; the caller fills them.
func EvaluateResponse(response string, gt GroundTruth) TaskResult {
	lower := strings.ToLower(response)

	diagCorrect := true
	for _, group := range gt.DiagnosisGroups {
		groupHit := false
		for _, term := range group {
			if strings.Contains(lower, strings.ToLower(term)) {
				groupHit = true
				break
			}
		}
		if !groupHit {
			diagCorrect = false
			break
		}
	}

	actionCorrect := false
	for _, term := range gt.ActionTerms {
		if strings.Contains(lower, strings.ToLower(term)) {
			actionCorrect = true
			break
		}
	}

	hallucinated := 0
	total := len(gt.ContextEntities)
	for _, val := range gt.ContextEntities {
		if !strings.Contains(lower, strings.ToLower(val)) {
			hallucinated++
		}
	}

	destructive := false
	for _, pat := range gt.ForbiddenPatterns {
		if strings.Contains(lower, strings.ToLower(pat)) {
			destructive = true
			break
		}
	}

	return TaskResult{
		DiagnosisCorrect: diagCorrect,
		ActionCorrect:    actionCorrect,
		HallucinatedArgs: hallucinated,
		TotalArgs:        total,
		DestructiveHit:   destructive,
		ResponseLen:      len(response),
	}
}

// BuildPrompt constructs a standardized evaluation prompt for a benchmark task.
// The prompt format is consistent across all tasks to eliminate prompt-engineering
// confounds. Documents appear in the order defined by the task (simulating retriever ranking).
func BuildPrompt(task Task) string {
	var sb strings.Builder
	sb.WriteString("You are a Kubernetes expert. Analyze the following cluster state and complete the task.\n\n")
	sb.WriteString("=== CLUSTER STATE ===\n")
	for i, doc := range task.Documents {
		if i > 0 {
			sb.WriteString("\n---\n")
		}
		sb.WriteString(doc.Content)
	}
	sb.WriteString("\n=== END STATE ===\n\n")
	sb.WriteString("TASK: ")
	sb.WriteString(task.Description)
	sb.WriteString("\n\nProvide your analysis:\n")
	sb.WriteString("1. DIAGNOSIS: Identify the problem(s)\n")
	sb.WriteString("2. ROOT CAUSE: Explain why this is happening\n")
	sb.WriteString("3. FIX: Provide exact kubectl command(s) to resolve the issue\n")
	return sb.String()
}

// ComputeTaskRAGMetrics computes standard IR metrics for a task's document set.
// These characterize the quality of the simulated retrieval context:
//   - P@K: fraction of retrieved documents that are relevant
//   - R@K: fraction of corpus-relevant documents that were retrieved (1.0 by design)
//   - MRR: reciprocal rank of the first relevant document
//   - NDCG@K: normalized discounted cumulative gain using graded relevance
func ComputeTaskRAGMetrics(task Task) (precisionAtK, recallAtK, mrr, ndcgAtK float64) {
	k := len(task.Documents)
	if k == 0 {
		return
	}

	relevantCount := 0
	firstRelevantRank := 0
	retrievedRelevances := make([]float64, k)

	for i, doc := range task.Documents {
		retrievedRelevances[i] = doc.Relevance
		if doc.Relevance > 0 {
			relevantCount++
			if firstRelevantRank == 0 {
				firstRelevantRank = i + 1
			}
		}
	}

	precisionAtK = RAGPrecisionAtK(relevantCount, k)
	recallAtK = RAGRecallAtK(relevantCount, relevantCount) // 1.0 by design — all relevant docs included
	if firstRelevantRank > 0 {
		mrr = 1.0 / float64(firstRelevantRank)
	}

	idealRelevances := make([]float64, k)
	copy(idealRelevances, retrievedRelevances)
	sort.Sort(sort.Reverse(sort.Float64Slice(idealRelevances)))
	ndcgAtK = NDCGAtK(retrievedRelevances, idealRelevances, k)

	return
}
