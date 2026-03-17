package llmbench

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
type Result struct {
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
