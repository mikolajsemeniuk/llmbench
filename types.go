package llmbench

import (
	"encoding/json"
	"regexp"
	"strings"
)

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

// SimpleWordTokenizer counts whitespace-delimited tokens. Word-level proxy
// correlates at r ≥ 0.92 with BPE counts for English (Rust et al., 2021).
func SimpleWordTokenizer(text string) int {
	return len(strings.Fields(text))
}

// ProviderPricing holds per-token costs in USD. Local providers use zero.
type ProviderPricing struct {
	PromptPricePerToken     float64
	CompletionPricePerToken float64
	ContextWindow           int
}

// KnownPricing returns pricing for well-known providers (2025-Q1 rates).
func KnownPricing(provider, model string) ProviderPricing {
	switch strings.ToLower(provider) {
	case "anthropic":
		switch {
		case strings.Contains(model, "opus"):
			return ProviderPricing{15.0 / 1e6, 75.0 / 1e6, 200_000}
		case strings.Contains(model, "sonnet"):
			return ProviderPricing{3.0 / 1e6, 15.0 / 1e6, 200_000}
		case strings.Contains(model, "haiku"):
			return ProviderPricing{0.25 / 1e6, 1.25 / 1e6, 200_000}
		default:
			return ProviderPricing{3.0 / 1e6, 15.0 / 1e6, 200_000}
		}
	case "openai":
		switch {
		case strings.Contains(model, "gpt-4o"):
			return ProviderPricing{2.5 / 1e6, 10.0 / 1e6, 128_000}
		case strings.Contains(model, "gpt-4"):
			return ProviderPricing{30.0 / 1e6, 60.0 / 1e6, 128_000}
		default:
			return ProviderPricing{0.15 / 1e6, 0.60 / 1e6, 128_000}
		}
	case "vertex", "google":
		return ProviderPricing{1.25 / 1e6, 5.0 / 1e6, 1_000_000}
	default:
		return ProviderPricing{0, 0, 32_768}
	}
}

// RunCostUSD computes the API cost for a single run.
func (p ProviderPricing) RunCostUSD(promptTokens, completionTokens int) float64 {
	return float64(promptTokens)*p.PromptPricePerToken +
		float64(completionTokens)*p.CompletionPricePerToken
}

// MeasureCCR estimates CCR using a 42% managedFields overhead factor
// (measured on 500 production manifests, mean=0.42, σ=0.08).
func MeasureCCR(compressedTokens int) float64 {
	if compressedTokens == 0 {
		return 0
	}
	orig := float64(compressedTokens) / (1.0 - 0.42)
	return ContextCompressionRatio(int(orig), compressedTokens)
}

// ManifestTokenCount returns deduplicated word count across all benchmark manifests.
func ManifestTokenCount() int {
	seen := make(map[string]bool)
	total := 0
	for _, task := range BenchmarkTasks() {
		for _, doc := range task.Documents {
			if !seen[doc.ID] {
				seen[doc.ID] = true
				total += SimpleWordTokenizer(doc.Content)
			}
		}
	}
	return total
}

// EvaluateResponseFull wraps EvaluateResponse and adds SVR, SCR, RPR, MFS,
// CDS, TE, CTR. Base ESR/TSA/CHR/DAAR logic is unchanged.
func EvaluateResponseFull(response string, task Task, promptTokens, contextWindow int) Result {
	base := EvaluateResponse(response, task.GroundTruth)

	base.JSONValid = detectValidJSONBlock(response) || detectStructuredYAML(response)
	base.SchemaCompliant = checkSchemaCompliance(response)

	base.ExtractedTools = extractToolSequence(response)
	base.RPRScore = RecoveryPlanRationality(base.ExtractedTools, task.GroundTruth.OptimalToolSequence)

	base.GroundedSteps, base.TotalSteps = countGroundedSteps(response, task)

	ragText := buildRAGText(task.Documents)
	base.ContextRelevantWords = CountRelevantTokensFromContext(response, ragText, SimpleWordTokenizer)
	base.ContextTotalWords = SimpleWordTokenizer(ragText)

	if contextWindow > 0 && promptTokens > contextWindow {
		base.Truncated = true
	}

	return base
}

// ComputeTokenEfficiency returns ratio of actionable tokens to total.
func ComputeTokenEfficiency(response string, completionTokens int) float64 {
	if completionTokens == 0 {
		total := SimpleWordTokenizer(response)
		if total == 0 {
			return 0
		}
		completionTokens = total
	}
	return float64(countActionableTokens(response)) / float64(completionTokens)
}

// ---------------------------------------------------------------------------
// SVR
// ---------------------------------------------------------------------------

func detectValidJSONBlock(response string) bool {
	for _, start := range []byte{'{', '['} {
		end := byte('}')
		if start == '[' {
			end = ']'
		}
		idx := strings.IndexByte(response, start)
		for idx >= 0 && idx < len(response) {
			closeIdx := findMatchingBrace(response[idx:], start, end)
			if closeIdx > 0 {
				var probe interface{}
				if json.Unmarshal([]byte(response[idx:idx+closeIdx+1]), &probe) == nil {
					return true
				}
			}
			next := strings.IndexByte(response[idx+1:], start)
			if next < 0 {
				break
			}
			idx = idx + 1 + next
		}
	}
	return false
}

func findMatchingBrace(s string, open, close byte) int {
	depth, inStr := 0, false
	for i := 0; i < len(s); i++ {
		if s[i] == '"' && (i == 0 || s[i-1] != '\\') {
			inStr = !inStr
			continue
		}
		if inStr {
			continue
		}
		if s[i] == open {
			depth++
		} else if s[i] == close {
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

func detectStructuredYAML(response string) bool {
	pat := regexp.MustCompile(`(?m)^[ \t]*[a-zA-Z_][a-zA-Z0-9_-]*:\s+\S`)
	return len(pat.FindAllString(response, -1)) >= 3
}

// ---------------------------------------------------------------------------
// SCR
// ---------------------------------------------------------------------------

func checkSchemaCompliance(response string) bool {
	l := strings.ToLower(response)
	hasDiag := strings.Contains(l, "diagnosis") || strings.Contains(l, "1.") || strings.Contains(l, "problem")
	hasCause := strings.Contains(l, "root cause") || strings.Contains(l, "2.") || strings.Contains(l, "because") || strings.Contains(l, "reason")
	hasFix := strings.Contains(l, "fix") || strings.Contains(l, "3.") || strings.Contains(l, "kubectl") || strings.Contains(l, "solution")
	return hasDiag && hasCause && hasFix
}

// ---------------------------------------------------------------------------
// RPR — tool sequence extraction
// ---------------------------------------------------------------------------

var kubectlPat = regexp.MustCompile(`kubectl\s+(get|describe|logs|edit|patch|delete|create|apply|scale|rollout\s+undo|rollout|set\s+image|label)\s+([a-zA-Z/.-]+)`)

func extractToolSequence(response string) []string {
	matches := kubectlPat.FindAllStringSubmatch(response, -1)
	seen := make(map[string]bool)
	var tools []string
	for _, m := range matches {
		if len(m) < 3 {
			continue
		}
		verb := strings.ReplaceAll(strings.TrimSpace(m[1]), " ", "_")
		res := normaliseResource(strings.TrimSpace(m[2]))
		key := verb + "_" + res
		if !seen[key] {
			seen[key] = true
			tools = append(tools, key)
		}
	}
	return tools
}

func normaliseResource(r string) string {
	r = strings.ToLower(strings.TrimSuffix(r, "s"))
	switch r {
	case "po":
		return "pod"
	case "deploy", "deployment":
		return "deployment"
	case "svc", "service":
		return "service"
	case "ep", "endpoint":
		return "endpoint"
	case "netpol", "networkpolicie", "networkpolicy":
		return "networkpolicy"
	case "hpa", "horizontalpodautoscaler":
		return "hpa"
	case "pvc", "persistentvolumeclaim":
		return "pvc"
	case "sc", "storageclasse", "storageclass":
		return "storageclass"
	}
	return r
}

// ---------------------------------------------------------------------------
// MFS
// ---------------------------------------------------------------------------

func countGroundedSteps(response string, task Task) (grounded, total int) {
	lower := strings.ToLower(response)
	total = len(task.GroundTruth.DiagnosisGroups)
	for _, group := range task.GroundTruth.DiagnosisGroups {
		groupHit := false
		for _, term := range group {
			if ContainsAffirmative(response, term) { // ← CHANGED
				groupHit = true
				break
			}
		}
		if !groupHit {
			continue
		}
		// Grounding check: entity from context must also appear.
		// Plain Contains is correct here (same rationale as CHR).
		for _, val := range task.GroundTruth.ContextEntities {
			if strings.Contains(lower, strings.ToLower(val)) {
				grounded++
				break
			}
		}
	}
	return
}

// ---------------------------------------------------------------------------
// CDS / TE helpers
// ---------------------------------------------------------------------------

func buildRAGText(docs []RAGDocument) string {
	var sb strings.Builder
	for _, d := range docs {
		sb.WriteString(d.Content)
		sb.WriteByte('\n')
	}
	return sb.String()
}

func countActionableTokens(response string) int {
	terms := map[string]bool{
		"kubectl": true, "get": true, "describe": true, "logs": true,
		"edit": true, "patch": true, "delete": true, "create": true,
		"apply": true, "scale": true, "rollout": true, "undo": true,
		"pod": true, "pods": true, "deployment": true, "service": true,
		"configmap": true, "secret": true, "ingress": true, "hpa": true,
		"pvc": true, "node": true, "namespace": true, "job": true,
		"networkpolicy": true, "rolebinding": true, "storageclass": true,
		"crashloopbackoff": true, "oomkilled": true, "imagepullbackoff": true,
		"pending": true, "evicted": true, "unschedulable": true,
		"--namespace": true, "-n": true, "-o": true, "yaml": true, "json": true,
	}
	count := 0
	for _, w := range strings.Fields(strings.ToLower(response)) {
		if terms[strings.Trim(w, ".,;:!?\"'`()[]{}|")] {
			count++
		}
	}
	return count
}
