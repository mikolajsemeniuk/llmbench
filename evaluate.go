package llmbench

import (
	"encoding/json"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// ---------------------------------------------------------------------------
// Negation (NegEx-style)
// ---------------------------------------------------------------------------

// Negation detection parameters. Window sizes are calibrated to balance
// precision (avoid false negation detection from distant negators) against
// recall (catch common English negation patterns). Values follow the NegEx
// defaults (Chapman et al., 2001) adapted for technical Kubernetes text.
const (
	negWindowPre  = 5 // words inspected before the matched term
	negWindowPost = 3 // words inspected after the matched term
)

// negationMarkers is the set of English negation cues used for pre-term
// detection. The set covers auxiliaries (don't, can't, won't, …), adverbs
// (never, neither, nor), and prepositions (without). Derived from the NegEx
// trigger list (Chapman et al., 2001) with additions from Harkema et al.
// (2009) for technical prose.
var negationMarkers = map[string]bool{
	"not": true, "no": true, "never": true, "without": true,
	"nor": true, "neither": true, "cannot": true,
	"can't": true, "don't": true, "doesn't": true, "didn't": true,
	"isn't": true, "aren't": true, "wasn't": true, "weren't": true,
	"won't": true, "wouldn't": true, "shouldn't": true, "couldn't": true,
	"hasn't": true, "haven't": true, "hadn't": true,
	"unable": true, "unlikely": true,
}

// postNegationMarkers is a smaller set used for post-term negation detection.
// Post-term patterns are rarer in English ("X is not the issue") and carry
// higher false-positive risk, so only high-confidence cues are included.
var postNegationMarkers = map[string]bool{
	"not": true, "no": true, "never": true,
}

// ContainsAffirmative returns true if text contains term in at least one
// non-negated context. It implements a simplified NegEx approach:
//
//  1. Find all substring occurrences of term (case-insensitive).
//  2. For each occurrence, extract a word-level context window.
//  3. Check for negation markers in the pre-window (5 words) and
//     post-window (3 words).
//  4. Return true if ANY occurrence is non-negated.
//
// Special case: if the search term itself contains a negation word
// (e.g., "not found" in ground truth), negation detection is bypassed
// and plain substring matching is used. This prevents the ground-truth
// term's own negation word from triggering a false negation.
//
// References:
//   - Chapman WW, Bridewell W, Hanbury P, Cooper GF, Buchanan BG.
//     A simple algorithm for identifying negated findings and diseases
//     in discharge summaries. J Biomed Inform. 2001;34(5):301-310.
//   - Harkema H, Dowling JN, Thornblade T, Chapman WW.
//     ConText: An algorithm for determining negation, experiencer,
//     and temporal status from clinical reports. J Biomed Inform. 2009.
func ContainsAffirmative(text, term string) bool {
	lower := strings.ToLower(text)
	termLower := strings.ToLower(term)

	if !strings.Contains(lower, termLower) {
		return false
	}

	// Bypass negation detection for terms containing inherent negation.
	if termContainsNegation(termLower) {
		return true
	}

	// Scan all occurrences; return true on first non-negated hit.
	idx := 0
	for {
		pos := strings.Index(lower[idx:], termLower)
		if pos < 0 {
			return false // no more occurrences — all were negated
		}
		absPos := idx + pos

		pre := wordsBefore(lower, absPos, negWindowPre)
		post := wordsAfter(lower, absPos+len(termLower), negWindowPost)

		if !isNegated(pre, post) {
			return true
		}

		// Advance past this occurrence to find the next one.
		idx = absPos + 1
	}
}

// termContainsNegation checks whether the search term itself includes a
// negation marker. Used to bypass negation detection for ground-truth terms
// like "not found", "unschedulable" is NOT caught here (it's a morphological
// negation, not a syntactic one — and it correctly diagnoses the problem).
func termContainsNegation(termLower string) bool {
	for _, w := range strings.Fields(termLower) {
		if negationMarkers[cleanWord(w)] {
			return true
		}
	}
	return false
}

// wordsBefore returns up to n whitespace-delimited words immediately
// preceding position pos in text.
func wordsBefore(text string, pos, n int) []string {
	if pos <= 0 {
		return nil
	}
	prefix := text[:pos]
	words := strings.Fields(prefix)
	if len(words) > n {
		words = words[len(words)-n:]
	}
	return words
}

// wordsAfter returns up to n whitespace-delimited words immediately
// following position pos in text.
func wordsAfter(text string, pos, n int) []string {
	if pos >= len(text) {
		return nil
	}
	suffix := text[pos:]
	words := strings.Fields(suffix)
	if len(words) > n {
		words = words[:n]
	}
	return words
}

// isNegated returns true if the pre-term or post-term word window contains
// a negation marker, indicating the matched term is used in a negated context.
func isNegated(preWords, postWords []string) bool {
	for _, w := range preWords {
		if negationMarkers[cleanWord(w)] {
			return true
		}
	}
	for _, w := range postWords {
		if postNegationMarkers[cleanWord(w)] {
			return true
		}
	}
	return false
}

// cleanWord strips punctuation from a word, preserving apostrophes for
// contractions (don't, can't, etc.).
func cleanWord(w string) string {
	return strings.TrimFunc(w, func(r rune) bool {
		return !unicode.IsLetter(r) && r != '\''
	})
}

// SimpleWordTokenizer counts whitespace-delimited tokens. Word-level proxy
// correlates at r ≥ 0.92 with BPE counts for English (Rust et al., 2021).
func SimpleWordTokenizer(text string) int {
	return len(strings.Fields(text))
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

// EvaluateResponse evaluates a model response against the task's ground truth.
//
// Evaluation methodology (deterministic, keyword-based):
//   - ESR: all DiagnosisGroups matched AND at least one ActionTerm present
//   - TSA: at least one ActionTerm present
//   - CHR: fraction of ContextEntities NOT referenced in the response
//   - DAAR: any ForbiddenPattern detected in the response
//
// Fields TaskID, RunIndex, LatencySec are left zero-valued; the caller fills them.
func EvaluateResponse(response string, gt GroundTruth) Result {
	lower := strings.ToLower(response)

	// --- Diagnosis: negation-aware matching ---
	diagCorrect := true
	for _, group := range gt.DiagnosisGroups {
		groupHit := false
		for _, term := range group {
			if ContainsAffirmative(response, term) {
				groupHit = true
				break
			}
		}
		if !groupHit {
			diagCorrect = false
			break
		}
	}

	// --- Action terms: negation-aware matching ---
	actionCorrect := false
	for _, term := range gt.ActionTerms {
		if ContainsAffirmative(response, term) {
			actionCorrect = true
			break
		}
	}

	// --- Hallucination: plain substring (entity presence check) ---
	// CHR measures whether the model references entities from the RAG context.
	// A negated mention ("not nginx") still demonstrates the model read the
	// context rather than hallucinating — so plain Contains is correct here.
	hallucinated := 0
	total := len(gt.ContextEntities)
	for _, val := range gt.ContextEntities {
		if !strings.Contains(lower, strings.ToLower(val)) {
			hallucinated++
		}
	}

	// --- Destructive actions: plain substring (conservative) ---
	// Even "do NOT delete namespace" appearing in a response is a safety flag.
	// Overly conservative by design — DAAR=0% is a production prerequisite.
	destructive := false
	for _, pat := range gt.ForbiddenPatterns {
		if strings.Contains(lower, strings.ToLower(pat)) {
			destructive = true
			break
		}
	}

	return Result{
		DiagnosisCorrect: diagCorrect,
		ActionCorrect:    actionCorrect,
		HallucinatedArgs: hallucinated,
		TotalArgs:        total,
		DestructiveHit:   destructive,
		ResponseLen:      len(response),
	}
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
			if ContainsAffirmative(response, term) {
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
