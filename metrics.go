package llmbench

import (
	"encoding/json"
	"math"
	"sort"
	"strings"

	"gonum.org/v1/gonum/stat/distuv"
)

// ExecutionSuccessRate (ESR) measures the fraction of LLM-generated MCP tool
// calls that achieved the expected end-state in the Kubernetes cluster.
//
// $$\text{ESR} = \frac{E_{success}}{E_{total}}$$
func ExecutionSuccessRate(successfulExecutions, totalExecutions int) float64 {
	if totalExecutions == 0 {
		return 0.0
	}
	return float64(successfulExecutions) / float64(totalExecutions)
}

// SyntaxValidationRate (SVR) measures the fraction of model responses that
// parse without error as valid JSON or YAML, independent of semantic correctness.
//
// $$\text{SVR} = \frac{V_{valid}}{V_{total}}$$
func SyntaxValidationRate(validResponses, totalResponses int) float64 {
	if totalResponses == 0 {
		return 0.0
	}
	return float64(validResponses) / float64(totalResponses)
}

// ToolSelectionAccuracy (TSA) measures the model's ability to choose the
// correct MCP tool from the available set, regardless of argument correctness.
//
// $$\text{TSA} = \frac{T_{correct}}{T_{total}}$$
func ToolSelectionAccuracy(correctSelections, totalSelections int) float64 {
	if totalSelections == 0 {
		return 0.0
	}
	return float64(correctSelections) / float64(totalSelections)
}

// Tokenizer is a function that counts the number of tokens in a string
// using a model-native tokenizer (e.g., tiktoken, HuggingFace tokenizers).
type Tokenizer func(text string) int

// CalculateTokenEfficiency (TE) quantifies model verbosity by comparing the
// machine-actionable JSON payload to the total completion length. The payload
// is structurally minified to establish the theoretical minimum information
// entropy before re-tokenization. A TE of 0 is assigned when the model fails
// to produce valid JSON, penalizing non-actionable text generation.
//
// $$\text{TE} = \frac{\text{Tokens}_{payload}}{\text{Tokens}_{total}}$$
func CalculateTokenEfficiency(rawJSON string, totalCompletionTokens int, tokenize Tokenizer) float64 {
	if totalCompletionTokens == 0 {
		return 0.0
	}
	minifiedPayload, err := ExtractMachineActionablePayload(rawJSON)
	if err != nil {
		return 0.0
	}
	payloadTokens := tokenize(minifiedPayload)
	return float64(payloadTokens) / float64(totalCompletionTokens)
}

// ExtractMachineActionablePayload parses and re-serializes a JSON string to
// its compact (whitespace-free) canonical form, establishing the theoretical
// minimum information entropy of the payload.
//
// $$P_{minified} = \min_{whitespace}\;\text{JSON}(P_{raw})$$
func ExtractMachineActionablePayload(rawJSON string) (string, error) {
	var parsed interface{}
	if err := json.Unmarshal([]byte(rawJSON), &parsed); err != nil {
		return "", err
	}
	minified, err := json.Marshal(parsed)
	if err != nil {
		return "", err
	}
	return string(minified), nil
}

// ContextHallucinationRate (CHR) measures the fraction of tool-call arguments
// that do not appear in the provided RAG context, indicating fabrication from
// model weights rather than grounding in the retrieved documents.
//
// $$\text{CHR} = \frac{A_{hallucinated}}{A_{total}}$$
func ContextHallucinationRate(hallucinatedArgs, totalArgs int) float64 {
	if totalArgs == 0 {
		return 0.0
	}
	return float64(hallucinatedArgs) / float64(totalArgs)
}

// SchemaComplianceRate (SCR) measures the fraction of responses whose JSON
// payload fully conforms to the expected MCP tool schema, including all
// required fields, correct types, and no extraneous properties. Unlike SVR
// which only checks syntactic validity, SCR enforces full schema adherence.
//
// $$\text{SCR} = \frac{C_{schema\_valid}}{C_{total}}$$
func SchemaComplianceRate(schemaValidResponses, totalResponses int) float64 {
	if totalResponses == 0 {
		return 0.0
	}
	return float64(schemaValidResponses) / float64(totalResponses)
}

// ContextDensityScore (CDS) measures how effectively the model utilizes the
// provided RAG context by computing the ratio of context-derived tokens that
// appear in the generated MCP call to the total context window size.
// A high CDS with low CHR indicates strong analytical extraction capability.
//
// $$\text{CDS} = \frac{T_{relevant}}{T_{context}}$$
func ContextDensityScore(relevantTokens, contextWindowTokens int) float64 {
	if contextWindowTokens == 0 {
		return 0.0
	}
	return float64(relevantTokens) / float64(contextWindowTokens)
}

// LatencyToActionEfficiency (LAE) captures operational efficiency by relating
// execution success to response time. It enables comparison of low-latency
// models (Qwen, DeepSeek) against high-latency models (Anthropic, Vertex)
// in real-time DevOps scenarios where time-to-action is critical.
//
// $$\text{LAE} = \frac{\text{ESR}}{L_{avg}}$$
func LatencyToActionEfficiency(esr float64, avgLatencySeconds float64) float64 {
	if avgLatencySeconds <= 0 {
		return 0.0
	}
	return esr / avgLatencySeconds
}

// FirstCallSuccessRate (FCSR) measures the model's ability to solve a task on
// the first attempt without any retry or follow-up call. In production K8s
// environments, retries incur real operational cost and latency.
//
// $$\text{FCSR} = \frac{T_{solved\_in\_1}}{T_{total}}$$
func FirstCallSuccessRate(solvedInFirstCall, totalTasks int) float64 {
	if totalTasks == 0 {
		return 0.0
	}
	return float64(solvedInFirstCall) / float64(totalTasks)
}

// MeanTimeToRecovery (MTTR) adapts the classic SRE metric to LLM evaluation,
// measuring the average wall-clock time from fault detection to successful
// cluster state restoration across all recovery attempts.
//
// $$\text{MTTR} = \frac{1}{N}\sum_{i=1}^{N}(t_{resolved,i} - t_{detected,i})$$
func MeanTimeToRecovery(recoveryDurationsSeconds []float64) float64 {
	if len(recoveryDurationsSeconds) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, d := range recoveryDurationsSeconds {
		sum += d
	}
	return sum / float64(len(recoveryDurationsSeconds))
}

// LatencyPercentile computes the p-th percentile of a latency sample using
// linear interpolation consistent with NumPy's default method. Percentile
// reporting (p50, p95, p99) is required over mean latency because the mean
// is sensitive to outliers such as cold starts and API throttling.
//
// $$L_{p} = (1 - f)\,x_{\lfloor i \rfloor} + f\,x_{\lceil i \rceil},
// \quad i = \frac{p}{100}(n-1)$$
func LatencyPercentile(latenciesSec []float64, percentile float64) float64 {
	if len(latenciesSec) == 0 || percentile < 0 || percentile > 100 {
		return 0.0
	}
	sorted := make([]float64, len(latenciesSec))
	copy(sorted, latenciesSec)
	sort.Float64s(sorted)

	idx := (percentile / 100.0) * float64(len(sorted)-1)
	lower := int(math.Floor(idx))
	upper := int(math.Ceil(idx))
	if lower == upper {
		return sorted[lower]
	}
	frac := idx - float64(lower)
	return sorted[lower]*(1-frac) + sorted[upper]*frac
}

// RAGPrecisionAtK (P@K) measures the fraction of the top-K retrieved
// documents that are ground-truth relevant. It quantifies retriever quality
// independently of the downstream LLM generation quality.
//
// $$\text{P@K} = \frac{|\text{relevant} \cap \text{retrieved@K}|}{K}$$
func RAGPrecisionAtK(retrievedRelevantCount, k int) float64 {
	if k == 0 {
		return 0.0
	}
	return float64(retrievedRelevantCount) / float64(k)
}

// RAGRecallAtK (R@K) measures the fraction of all corpus-relevant documents
// that appear in the top-K retrieval results. High recall is critical when
// omitting a key manifest (e.g., a ConfigMap with credentials) directly
// causes MCP call failure.
//
// $$\text{R@K} = \frac{|\text{relevant} \cap \text{retrieved@K}|}{|\text{relevant}|}$$
func RAGRecallAtK(retrievedRelevantCount, totalRelevantInCorpus int) float64 {
	if totalRelevantInCorpus == 0 {
		return 0.0
	}
	return float64(retrievedRelevantCount) / float64(totalRelevantInCorpus)
}

// RAGFScoreAtK computes the weighted harmonic mean of P@K and R@K.
// Beta=1.0 yields the standard F1 (equal weight). Beta<1 favors precision
// (less noise in context); beta>1 favors recall (do not miss key manifests).
//
// $$F_{\beta}\text{@K} = (1+\beta^2)\,\frac{P\text{@K}\cdot R\text{@K}}
// {\beta^2\,P\text{@K} + R\text{@K}}$$
func RAGFScoreAtK(precisionAtK, recallAtK, beta float64) float64 {
	if precisionAtK+recallAtK == 0 {
		return 0.0
	}
	betaSq := beta * beta
	return (1 + betaSq) * (precisionAtK * recallAtK) / (betaSq*precisionAtK + recallAtK)
}

// RecoveryPlanRationality (RPR) evaluates the optimality of the model's
// proposed MCP tool-call sequence against the ground-truth optimal sequence
// using normalized Levenshtein edit distance over tool names.
// RPR=1.0 indicates an identical plan; RPR=0.0 indicates complete mismatch.
//
// $$\text{RPR} = 1 - \frac{\text{EditDist}(S_{model},\,S_{optimal})}
// {\max(|S_{model}|,\,|S_{optimal}|)}$$
func RecoveryPlanRationality(modelToolSequence, optimalToolSequence []string) float64 {
	dist := levenshteinDistance(modelToolSequence, optimalToolSequence)
	maxLen := math.Max(float64(len(modelToolSequence)), float64(len(optimalToolSequence)))
	if maxLen == 0 {
		return 1.0
	}
	return 1.0 - float64(dist)/maxLen
}

// levenshteinDistance computes the edit distance between two string sequences.
func levenshteinDistance(a, b []string) int {
	la, lb := len(a), len(b)
	dp := make([][]int, la+1)
	for i := range dp {
		dp[i] = make([]int, lb+1)
	}
	for i := 0; i <= la; i++ {
		dp[i][0] = i
	}
	for j := 0; j <= lb; j++ {
		dp[0][j] = j
	}
	for i := 1; i <= la; i++ {
		for j := 1; j <= lb; j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
		}
	}
	return dp[la][lb]
}

// MultiStepFaithfulnessScore (MFS) measures whether each agent step is
// grounded in the actual output of the preceding step, rather than
// hallucinated. Adapted from the RAGAS Faithfulness metric for agentic
// MCP environments where multi-step K8s tasks (diagnose, patch, verify)
// require chained tool calls with data dependencies.
//
// $$\text{MFS} = \frac{S_{grounded}}{S_{total}}$$
func MultiStepFaithfulnessScore(groundedSteps, totalSteps int) float64 {
	if totalSteps == 0 {
		return 0.0
	}
	return float64(groundedSteps) / float64(totalSteps)
}

// ErrorRecoveryRate (ERR) measures the model's self-correction capability:
// after receiving an error response from the MCP server (e.g., "pod not
// found"), can the model autonomously generate a corrected call without
// additional user intervention?
//
// $$\text{ERR} = \frac{T_{self\_corrected}}{T_{with\_initial\_error}}$$
func ErrorRecoveryRate(selfCorrectedTasks, tasksWithInitialError int) float64 {
	if tasksWithInitialError == 0 {
		return 0.0
	}
	return float64(selfCorrectedTasks) / float64(tasksWithInitialError)
}

// ContextTruncationRate (CTR) measures how often a K8s manifest exceeds the
// model's context window and must be truncated before submission. High CTR
// for small-context models (e.g., 8K tokens) demonstrates a limitation that
// RAG-based selective retrieval can compensate for.
//
// $$\text{CTR} = \frac{T_{truncated}}{T_{total}}$$
func ContextTruncationRate(truncatedTasks, totalTasks int) float64 {
	if totalTasks == 0 {
		return 0.0
	}
	return float64(truncatedTasks) / float64(totalTasks)
}

// BootstrapConfidenceInterval computes a (1-alpha) confidence interval for a
// proportion metric using parametric bootstrap on a Binomial distribution.
// For each of nBootstrap iterations, successes are resampled from
// Bernoulli(p_observed) and the resulting proportion is recorded.
// The CI bounds are taken at the alpha/2 and 1-alpha/2 quantiles of the
// bootstrap distribution. nBootstrap=10000 is the academic standard.
//
// Returns [lower, upper] bounds.
func BootstrapConfidenceInterval(successes, total, nBootstrap int, alpha float64, rng func() float64) [2]float64 {
	if total == 0 || nBootstrap == 0 {
		return [2]float64{0.0, 0.0}
	}
	pObs := float64(successes) / float64(total)
	samples := make([]float64, nBootstrap)

	for i := range samples {
		bootstrapSuccesses := 0
		for j := 0; j < total; j++ {
			if rng() < pObs {
				bootstrapSuccesses++
			}
		}
		samples[i] = float64(bootstrapSuccesses) / float64(total)
	}

	sort.Float64s(samples)
	lowerIdx := int(math.Floor(alpha / 2.0 * float64(nBootstrap)))
	upperIdx := int(math.Ceil((1.0-alpha/2.0)*float64(nBootstrap))) - 1

	if upperIdx >= nBootstrap {
		upperIdx = nBootstrap - 1
	}
	return [2]float64{samples[lowerIdx], samples[upperIdx]}
}

// CliffsData computes Cliff's delta (δ) effect size between two independent
// samples. Unlike Cohen's d, Cliff's δ is non-parametric and appropriate for
// ordinal or non-continuous data such as ESR and FCSR. δ ∈ [-1, 1] where
// δ > 0 indicates stochastic dominance of groupA over groupB. Effect size
// thresholds follow Romano et al. (2006):
//   - |δ| < 0.147: negligible
//   - |δ| < 0.330: small
//   - |δ| < 0.474: medium
//   - |δ| ≥ 0.474: large
//
// $$\delta = \frac{|\{(i,j): a_i > b_j\}| - |\{(i,j): a_i < b_j\}|}{n_a \cdot n_b}$$
func CliffsData(groupA, groupB []float64) float64 {
	na, nb := len(groupA), len(groupB)
	if na == 0 || nb == 0 {
		return 0.0
	}
	dominates, dominated := 0, 0
	for _, a := range groupA {
		for _, b := range groupB {
			if a > b {
				dominates++
			} else if a < b {
				dominated++
			}
		}
	}
	return float64(dominates-dominated) / float64(na*nb)
}

// CliffsEffectSizeLabel returns a human-readable effect size label for a
// given Cliff's δ value following the Romano et al. (2006) thresholds.
func CliffsEffectSizeLabel(delta float64) string {
	abs := math.Abs(delta)
	switch {
	case abs < 0.147:
		return "negligible"
	case abs < 0.330:
		return "small"
	case abs < 0.474:
		return "medium"
	default:
		return "large"
	}
}

// WilcoxonRankSum performs the Wilcoxon rank-sum test (equivalent to the
// Mann-Whitney U test) on two independent samples. This non-parametric test
// determines whether two populations differ without assuming normality,
// which is critical for LLM metrics where distributions are often discrete
// or heavily skewed. The implementation includes the tied-rank variance
// correction following Conover (1999).
//
// Returns the U statistic (min of U_A, U_B) and a two-tailed p-value
// derived from the normal approximation.
//
// $$U_A = R_A - \frac{n_a(n_a+1)}{2},\quad
// z = \frac{U_A - \mu_U}{\sigma_U},\quad
// p = 2\,\Phi(-|z|)$$
//
// where $\sigma_U$ includes the tied-rank correction:
//
// $$\sigma_U = \sqrt{\frac{n_a n_b}{12}
// \left(N+1 - \frac{\sum(t_k^3 - t_k)}{N(N-1)}\right)}$$
func WilcoxonRankSum(groupA, groupB []float64) (uStat float64, pValue float64) {
	na, nb := len(groupA), len(groupB)
	if na == 0 || nb == 0 {
		return 0, 1.0
	}

	type entry struct {
		value float64
		group int
	}

	n := na + nb
	combined := make([]entry, 0, n)
	for _, v := range groupA {
		combined = append(combined, entry{v, 0})
	}
	for _, v := range groupB {
		combined = append(combined, entry{v, 1})
	}
	sort.Slice(combined, func(i, j int) bool {
		return combined[i].value < combined[j].value
	})

	ranks := make([]float64, n)
	var tieCorrection float64
	for i := 0; i < n; {
		j := i
		for j < n && combined[j].value == combined[i].value {
			j++
		}
		avg := float64(i+j+1) / 2.0
		t := float64(j - i)
		tieCorrection += t*t*t - t
		for k := i; k < j; k++ {
			ranks[k] = avg
		}
		i = j
	}

	var rankSumA float64
	for i, e := range combined {
		if e.group == 0 {
			rankSumA += ranks[i]
		}
	}

	uA := rankSumA - float64(na*(na+1))/2.0
	uB := float64(na*nb) - uA
	uStat = math.Min(uA, uB)

	meanU := float64(na*nb) / 2.0
	nf := float64(n)
	sigma := math.Sqrt(float64(na*nb) / 12.0 * (nf + 1.0 - tieCorrection/(nf*(nf-1.0))))
	if sigma == 0 {
		return uStat, 1.0
	}

	z := (uA - meanU) / sigma
	normal := distuv.Normal{Mu: 0, Sigma: 1}
	pValue = 2.0 * normal.Survival(math.Abs(z))

	return uStat, pValue
}

// WilcoxonSignificanceLabel returns the conventional significance label for
// a p-value: "***" (p<0.001), "**" (p<0.01), "*" (p<0.05), "n.s." otherwise.
func WilcoxonSignificanceLabel(pValue float64) string {
	switch {
	case pValue < 0.001:
		return "***"
	case pValue < 0.01:
		return "**"
	case pValue < 0.05:
		return "*"
	default:
		return "n.s."
	}
}

// BonferroniCorrection applies the classical Bonferroni adjustment to a slice
// of raw p-values obtained from m simultaneous hypothesis tests.
//
// Each raw p-value is multiplied by m (the family size) and capped at 1.0,
// controlling the familywise error rate (FWER) at the nominal α level.
// The correction is conservative — it assumes all m tests are independent —
// but is accepted by ACM TOIS, IP&M, and IEEE Access as the minimum required
// disclosure when reporting multiple comparisons.
//
// The returned slice preserves the original ordering of pValues.
//
// $$\tilde{p}_i = \min(1,\; m \cdot p_i)$$
func BonferroniCorrection(pValues []float64) []float64 {
	m := float64(len(pValues))
	out := make([]float64, len(pValues))
	for i, p := range pValues {
		out[i] = math.Min(1.0, p*m)
	}
	return out
}

// HolmBonferroniCorrection applies the Holm (1979) step-down procedure to a
// slice of raw p-values from m simultaneous hypothesis tests.
//
// Holm's method is uniformly more powerful than the classical Bonferroni
// correction: it controls the FWER at the same α while rejecting at least as
// many (often more) null hypotheses. The procedure is required when comparing
// LLM metrics such as ESR, TSA, CHR, and DAAR simultaneously, because the
// naive per-metric Bonferroni inflates the Type I error across the full family.
//
// Algorithm (Holm, 1979):
//  1. Sort the m raw p-values in ascending order, obtaining p_(1) ≤ … ≤ p_(m).
//  2. Compute the adjusted value for rank k as (m − k + 1) × p_(k).
//  3. Enforce monotonicity: p̃_(k) = max(p̃_(k−1), adjusted_(k)).
//  4. Cap at 1.0 and return in the ORIGINAL input order.
//
// $$\tilde{p}_{(k)} = \min\!\left(1,\;\max_{j \le k}\bigl((m-j+1)\,p_{(j)}\bigr)\right)$$
//
// Reference: Holm, S. (1979). A simple sequentially rejective multiple test
// procedure. Scandinavian Journal of Statistics, 6(2), 65–70.
func HolmBonferroniCorrection(pValues []float64) []float64 {
	m := len(pValues)
	if m == 0 {
		return nil
	}

	// Pair each p-value with its original index so we can restore order later.
	type indexed struct {
		p   float64
		idx int
	}
	sorted := make([]indexed, m)
	for i, p := range pValues {
		sorted[i] = indexed{p, i}
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].p < sorted[j].p })

	// Step-down: compute adjusted p-values with monotonicity enforcement.
	adjusted := make([]float64, m)
	runningMax := 0.0
	for k, item := range sorted {
		// rank is 1-indexed; k is 0-indexed.
		candidate := float64(m-k) * item.p // (m − k+1 + 1 − 1) = m − k
		if candidate > runningMax {
			runningMax = candidate
		}
		adjusted[item.idx] = math.Min(1.0, runningMax)
	}

	return adjusted
}

// CountRelevantTokensFromContext counts tokens in the MCP payload JSON that
// also appear in the RAG context text, implementing the T_relevant numerator
// for the CDS metric. Uses case-insensitive word-level overlap.
func CountRelevantTokensFromContext(mcpPayloadJSON, ragContextText string, tokenize Tokenizer) int {
	payloadWords := tokenizeToWords(mcpPayloadJSON)
	contextWords := tokenizeToWords(ragContextText)

	contextSet := make(map[string]bool)
	for _, w := range contextWords {
		contextSet[strings.ToLower(w)] = true
	}

	relevantCount := 0
	for _, w := range payloadWords {
		if contextSet[strings.ToLower(w)] {
			relevantCount++
		}
	}
	return relevantCount
}

// tokenizeToWords splits text on whitespace and JSON structural characters,
// filtering out single-character tokens.
func tokenizeToWords(text string) []string {
	words := strings.FieldsFunc(text, func(r rune) bool {
		return r == ' ' || r == '\n' || r == '\t' || r == '"' ||
			r == '{' || r == '}' || r == ':' || r == ','
	})
	result := make([]string, 0, len(words))
	for _, w := range words {
		if len(w) > 1 {
			result = append(result, w)
		}
	}
	return result
}

// MeanReciprocalRank (MRR) measures how quickly the RAG retriever places the
// first relevant document in the ranking. Unlike P@K and R@K which measure
// whether the relevant document was retrieved at all, MRR penalizes placing
// it at a lower rank position. A standard IR metric required by ACM TOIS.
//
// $$\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\text{rank}_i}$$
//
// where rank_i is the 1-indexed position of the first relevant document for
// query i. If no relevant document was retrieved, the contribution is 0.
func MeanReciprocalRank(ranksOfFirstRelevant []int) float64 {
	if len(ranksOfFirstRelevant) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, rank := range ranksOfFirstRelevant {
		if rank > 0 {
			sum += 1.0 / float64(rank)
		}
	}
	return sum / float64(len(ranksOfFirstRelevant))
}

// NDCGAtK (Normalized Discounted Cumulative Gain) evaluates the quality of
// the RAG retriever's document ranking using graded relevance scores
// (e.g., 3=primary target, 1=related resource, 0=noise). Unlike P@K, NDCG
// penalizes placing a highly relevant document at a low rank position through
// logarithmic discounting. The gold standard ranking metric in ACM TOIS.
//
// $$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}},\quad
// \text{DCG@K} = \sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}$$
func NDCGAtK(retrievedRelevances []float64, idealRelevances []float64, k int) float64 {
	if k == 0 || len(retrievedRelevances) == 0 {
		return 0.0
	}

	dcg := 0.0
	for i := 0; i < k && i < len(retrievedRelevances); i++ {
		dcg += (math.Pow(2, retrievedRelevances[i]) - 1) / math.Log2(float64(i+2))
	}

	idcg := 0.0
	for i := 0; i < k && i < len(idealRelevances); i++ {
		idcg += (math.Pow(2, idealRelevances[i]) - 1) / math.Log2(float64(i+2))
	}

	if idcg == 0.0 {
		return 0.0
	}
	return dcg / idcg
}

// LostInTheMiddleVulnerability (LMV) quantifies a model's sensitivity to
// the position of relevant information within the prompt, based on the
// "Lost in the Middle" phenomenon (Liu et al., 2023). The metric is computed
// by running identical tasks with the key manifest placed at edge positions
// (first or last) versus the middle of the RAG context.
//
//   - LMV > 0.10: strong positional sensitivity, requires RAG chunk reordering
//   - LMV ≈ 0:    uniform context processing
//   - LMV < 0:    paradoxically better middle processing (rare)
//
// $$\text{LMV} = \text{ESR}_{edges} - \text{ESR}_{middle}$$
func LostInTheMiddleVulnerability(esrEdges, esrMiddle float64) float64 {
	return esrEdges - esrMiddle
}

// CostEfficiencyScore (CES) measures the number of successful MCP actions
// per unit of API cost (USD). For locally deployed models (Ollama, vLLM)
// where per-token cost is zero, CES returns +Inf, which is mathematically
// correct and represents the key cost advantage of local deployment.
//
// $$\text{CES} = \frac{E_{success}}{\text{TotalCost}_{\$}}$$
func CostEfficiencyScore(successfulExecutions int, totalCostUSD float64) float64 {
	if totalCostUSD <= 0 {
		return math.Inf(1)
	}
	return float64(successfulExecutions) / totalCostUSD
}

// DestructiveActionAttemptRate (DAAR) measures the fraction of MCP tool calls
// that target a blacklisted destructive operation (e.g., delete_namespace,
// force-delete without grace period). A DAAR of 0% is a prerequisite for
// production deployment and demonstrates that RAG-based intent filtering
// improves operational safety.
//
// $$\text{DAAR} = \frac{A_{destructive}}{A_{total}}$$
func DestructiveActionAttemptRate(destructiveAttempts, totalActionAttempts int) float64 {
	if totalActionAttempts == 0 {
		return 0.0
	}
	return float64(destructiveAttempts) / float64(totalActionAttempts)
}

// ContextCompressionRatio (CCR) measures the effectiveness of context
// optimization techniques (e.g., stripping managedFields, removing empty
// annotations, YAML-to-JSON compression) by comparing the original and
// compressed prompt token counts. CCR directly reduces CTR and API cost.
// A value of 0.5 means the prompt was shortened by 50%.
//
// $$\text{CCR} = \frac{T_{original} - T_{compressed}}{T_{original}}$$
func ContextCompressionRatio(originalTokens, compressedTokens int) float64 {
	if originalTokens == 0 {
		return 0.0
	}
	if compressedTokens >= originalTokens {
		return 0.0
	}
	return float64(originalTokens-compressedTokens) / float64(originalTokens)
}
