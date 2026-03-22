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

// Response holds the raw output from any model provider after a single
// completion call. All fields required for metric computation are present
// so the benchmark runner never needs to know which provider produced them.
type Response struct {
	// Text is the raw completion string returned by the model.
	Text string

	// PromptTokens is the number of tokens in the input prompt as reported
	// by the provider. Used for context-compression ratio (CCR) analysis.
	PromptTokens int

	// CompletionTokens is the number of tokens generated. Used for token
	// efficiency (TE) and cost-efficiency score (CES) computation.
	CompletionTokens int

	// LatencySec is the wall-clock time from request dispatch to full
	// response receipt, measured by the caller. Populated by the runner,
	// not by the provider implementation.
	LatencySec float64
}

type Report struct {
	Metadata  Metadata          `json:"metadata"`
	Metrics   Metrics           `json:"aggregate"`
	PerLevel  []LevelMetrics    `json:"per_level"`
	RAG       RAGQualityMetrics `json:"rag_quality"`
	Summaries []Summary         `json:"per_task"`
	Records   []Record          `json:"runs"`
}

type Metadata struct {
	Provider    string `json:"provider"`
	Model       string `json:"model"`
	ModelDigest string `json:"model_digest,omitempty"`
	ModelFamily string `json:"model_family,omitempty"`
	ModelQuant  string `json:"model_quantization,omitempty"`
	Timestamp   string `json:"timestamp"`
	TotalTasks  int    `json:"total_tasks"`
	RunsPerTask int    `json:"runs_per_task"`
	TotalRuns   int    `json:"total_runs"`
	Seed        int64  `json:"random_seed"`
}

type Metrics struct {
	ESR        float64    `json:"esr"`
	ESRCI      [2]float64 `json:"esr_ci_95"`
	TSA        float64    `json:"tsa"`
	CHR        float64    `json:"chr"`
	DAAR       float64    `json:"daar"`
	FCSR       float64    `json:"fcsr"`
	LAE        float64    `json:"lae"`
	MTTR       float64    `json:"mttr_sec"`
	LatencyP50 float64    `json:"latency_p50_sec"`
	LatencyP95 float64    `json:"latency_p95_sec"`
	LatencyP99 float64    `json:"latency_p99_sec"`
}

type LevelMetrics struct {
	Name string  `json:"name"`
	ESR  float64 `json:"esr"`
	TSA  float64 `json:"tsa"`
	CHR  float64 `json:"chr"`
	Runs int     `json:"runs"`
}

type RAGQualityMetrics struct {
	MeanPrecisionAtK float64 `json:"mean_precision_at_k"`
	MeanRecallAtK    float64 `json:"mean_recall_at_k"`
	MeanMRR          float64 `json:"mean_mrr"`
	MeanNDCGAtK      float64 `json:"mean_ndcg_at_k"`
	MeanFScoreAtK    float64 `json:"mean_f1_at_k"`
}

type Summary struct {
	TaskID     string  `json:"task_id"`
	Level      string  `json:"level"`
	ESR        float64 `json:"esr"`
	TSA        float64 `json:"tsa"`
	CHR        float64 `json:"chr"`
	MeanLatSec float64 `json:"mean_latency_sec"`
}

type Record struct {
	TaskID           string  `json:"task_id"`
	RunIndex         int     `json:"run_index"`
	LatencySec       float64 `json:"latency_sec"`
	DiagCorrect      bool    `json:"diagnosis_correct"`
	ActionCorrect    bool    `json:"action_correct"`
	Hallucinations   int     `json:"hallucinations"`
	TotalEntities    int     `json:"total_entities"`
	Destructive      bool    `json:"destructive_action"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TokensPerSec     float64 `json:"tokens_per_sec"`
	Error            string  `json:"error,omitempty"`
}

// CompareReport is written to compare.json and read by cmd/report.
type CompareReport struct {
	// GeneratedAt is the UTC timestamp of this comparison run.
	GeneratedAt string `json:"generated_at"`

	// ModelA / ModelB are the full provider/model strings, e.g.
	// "ollama/qwen2.5:3b-instruct".
	ModelA string `json:"model_a"`
	ModelB string `json:"model_b"`

	// Aggregate holds the metric values for both models side-by-side
	// with the statistical test result for each metric.
	Aggregate []MetricComparison `json:"aggregate"`

	// PerLevel holds level-by-level breakdowns for both models.
	PerLevel []LevelComparison `json:"per_level"`

	// PerTask allows per-task delta visualisation in the report.
	PerTask []TaskComparison `json:"per_task"`

	// Raw holds the original reports for display/download.
	Raw struct {
		A Report `json:"a"`
		B Report `json:"b"`
	} `json:"raw"`
}

// MetricComparison holds a head-to-head comparison for a single scalar metric.
type MetricComparison struct {
	// Name is the metric abbreviation, e.g. "ESR", "TSA".
	Name string `json:"name"`

	// FullName is the human-readable label for the report UI.
	FullName string `json:"full_name"`

	// HigherIsBetter controls the colour coding in the report template:
	// true → higher value is green; false (CHR, DAAR, latency) → lower is green.
	HigherIsBetter bool `json:"higher_is_better"`

	ValueA float64 `json:"value_a"`
	ValueB float64 `json:"value_b"`

	// Delta = ValueA - ValueB (positive means A is better for HigherIsBetter metrics).
	Delta float64 `json:"delta"`

	// WilcoxonU is the U statistic from the rank-sum test on per-run values.
	// Present only for metrics that have a per-run sample (ESR, TSA, CHR).
	// For scalar-only metrics (LAE, MTTR) this is 0.
	WilcoxonU float64 `json:"wilcoxon_u"`

	// PValue is the two-sided p-value before correction.
	PValue float64 `json:"p_value"`

	// PValueCorrected is Bonferroni-corrected over the number of metrics tested.
	PValueCorrected float64 `json:"p_value_corrected"`

	// Significance is the conventional label: "***", "**", "*", or "n.s."
	Significance string `json:"significance"`

	// EffectSize is the rank-biserial correlation r (range −1 to +1).
	// Magnitude: |r| < 0.1 = negligible, 0.1–0.3 = small, 0.3–0.5 = medium, >0.5 = large.
	EffectSize float64 `json:"effect_size_r"`

	// EffectLabel is "negligible", "small", "medium", or "large".
	EffectLabel string `json:"effect_label"`
}

// LevelComparison holds per-level ESR/TSA/CHR for both models.
type LevelComparison struct {
	Level string  `json:"level"`
	ESRA  float64 `json:"esr_a"`
	ESRB  float64 `json:"esr_b"`
	TSAA  float64 `json:"tsa_a"`
	TSAB  float64 `json:"tsa_b"`
	CHRA  float64 `json:"chr_a"`
	CHRB  float64 `json:"chr_b"`
	RunsA int     `json:"runs_a"`
	RunsB int     `json:"runs_b"`
}

// TaskComparison holds per-task ESR delta (A − B) for the scatter view.
type TaskComparison struct {
	TaskID string  `json:"task_id"`
	Level  string  `json:"level"`
	ESRA   float64 `json:"esr_a"`
	ESRB   float64 `json:"esr_b"`
	Delta  float64 `json:"delta"` // ESRA − ESRB
}
