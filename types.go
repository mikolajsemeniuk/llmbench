package llmbench

type TaskLevel string

const (
	LevelDiagnostic TaskLevel = "L1-diagnostic"
	LevelRepair     TaskLevel = "L2-repair"
	LevelMultiStep  TaskLevel = "L3-multi-step"
)

type RAGDocument struct {
	ID        string
	Content   string
	Relevance float64
}

type GroundTruth struct {
	DiagnosisGroups     [][]string
	ActionTerms         []string
	ContextEntities     map[string]string
	ForbiddenPatterns   []string
	OptimalToolSequence []string
}

type Task struct {
	ID          string
	Level       TaskLevel
	Description string
	Documents   []RAGDocument
	GroundTruth GroundTruth
}

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

	JSONValid            bool
	SchemaCompliant      bool
	ExtractedTools       []string
	RPRScore             float64
	GroundedSteps        int
	TotalSteps           int
	ContextRelevantWords int
	ContextTotalWords    int
	Truncated            bool
}

type Response struct {
	Text             string
	PromptTokens     int
	CompletionTokens int
	LatencySec       float64
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

type Report struct {
	Metadata  Metadata          `json:"metadata"`
	Metrics   Metrics           `json:"aggregate"`
	PerLevel  []LevelMetrics    `json:"per_level"`
	RAG       RAGQualityMetrics `json:"rag_quality"`
	Summaries []Summary         `json:"per_task"`
	Records   []Record          `json:"runs"`
}

type Metadata struct {
	Provider      string `json:"provider"`
	Model         string `json:"model"`
	ModelDigest   string `json:"model_digest,omitempty"`
	ModelFamily   string `json:"model_family,omitempty"`
	ModelQuant    string `json:"model_quantization,omitempty"`
	Timestamp     string `json:"timestamp"`
	TotalTasks    int    `json:"total_tasks"`
	RunsPerTask   int    `json:"runs_per_task"`
	TotalRuns     int    `json:"total_runs"`
	Seed          int64  `json:"random_seed"`
	ContextWindow int    `json:"context_window,omitempty"`
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

	// Extended.
	SVR float64 `json:"svr"`
	SCR float64 `json:"scr"`
	TE  float64 `json:"te"`
	CDS float64 `json:"cds"`
	RPR float64 `json:"rpr"`
	MFS float64 `json:"mfs"`
	CES float64 `json:"ces"`
	CTR float64 `json:"ctr"`
	CCR float64 `json:"ccr"`
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

	// Extended.
	JSONValid       bool    `json:"json_valid"`
	SchemaCompliant bool    `json:"schema_compliant"`
	RPR             float64 `json:"rpr"`
	MFS             float64 `json:"mfs,omitempty"`
	CDS             float64 `json:"cds"`
	TE              float64 `json:"te"`
	Truncated       bool    `json:"truncated"`
}

// ---------------------------------------------------------------------------
// CompareReport
// ---------------------------------------------------------------------------

type CompareReport struct {
	GeneratedAt string             `json:"generated_at"`
	ModelA      string             `json:"model_a"`
	ModelB      string             `json:"model_b"`
	Aggregate   []MetricComparison `json:"aggregate"`
	PerLevel    []LevelComparison  `json:"per_level"`
	PerTask     []TaskComparison   `json:"per_task"`
	Raw         struct {
		A Report `json:"a"`
		B Report `json:"b"`
	} `json:"raw"`
}

type MetricComparison struct {
	Name             string  `json:"name"`
	FullName         string  `json:"full_name"`
	HigherIsBetter   bool    `json:"higher_is_better"`
	ValueA           float64 `json:"value_a"`
	ValueB           float64 `json:"value_b"`
	Delta            float64 `json:"delta"`
	WilcoxonU        float64 `json:"wilcoxon_u"`
	PValue           float64 `json:"p_value"`
	PValueCorrected  float64 `json:"p_value_corrected"`
	CorrectionMethod string  `json:"correction_method"`
	Significance     string  `json:"significance"`
	EffectSize       float64 `json:"effect_size_r"`
	EffectLabel      string  `json:"effect_label"`
}

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

type TaskComparison struct {
	TaskID string  `json:"task_id"`
	Level  string  `json:"level"`
	ESRA   float64 `json:"esr_a"`
	ESRB   float64 `json:"esr_b"`
	Delta  float64 `json:"delta"`
}

// ProviderPricing holds per-token costs in USD. Local providers use zero.
type ProviderPricing struct {
	PromptPricePerToken     float64
	CompletionPricePerToken float64
	ContextWindow           int
}

// RunCostUSD computes the API cost for a single run.
func (p ProviderPricing) RunCostUSD(promptTokens, completionTokens int) float64 {
	return float64(promptTokens)*p.PromptPricePerToken +
		float64(completionTokens)*p.CompletionPricePerToken
}
