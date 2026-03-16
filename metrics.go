package llmbench

import (
	"encoding/json"
	"math"
	"sort"
	"strings"
)

// ExecutionSuccessRate oblicza wskaźnik sukcesu wykonania zadań w środowisku (np. K8s).
//
// Opis: Określa, jaki ułamek wygenerowanych przez model poleceń MCP
// faktycznie doprowadził do oczekiwanego stanu końcowego.
//
// Formuła:
// $$\text{ESR} = \frac{E_{success}}{E_{total}}$$
// Gdzie $E_{success}$ to liczba zadań, które zmieniły stan klastra zgodnie z intencją,
// a $E_{total}$ to całkowita liczba przeprowadzonych eksperymentów.
func ExecutionSuccessRate(successfulExecutions, totalExecutions int) float64 {
	if totalExecutions == 0 {
		return 0.0
	}
	return float64(successfulExecutions) / float64(totalExecutions)
}

// SyntaxValidationRate oblicza odsetek poprawnych składniowo odpowiedzi.
//
// Opis: Mierzy, jak często model potrafi wygenerować poprawny format danych
// (np. JSON dla argumentów MCP lub YAML dla manifestu K8s), który
// przechodzi przez parser bez błędu rzutowania (unmarshal error).
//
// Formuła:
// $$\text{SVR} = \frac{V_{valid}}{V_{total}}$$
// Gdzie $V_{valid}$ to liczba odpowiedzi z poprawną składnią, a $V_{total}$ to liczba wszystkich odpowiedzi.
func SyntaxValidationRate(validResponses, totalResponses int) float64 {
	if totalResponses == 0 {
		return 0.0
	}
	return float64(validResponses) / float64(totalResponses)
}

// ToolSelectionAccuracy oblicza dokładność wyboru odpowiedniego narzędzia MCP.
//
// Opis: Ocenia zdolność modelu do wybrania właściwej funkcji (narzędzia) z dostępnej puli.
// Nawet jeśli argumenty są błędne, weryfikujemy tu sam fakt podjęcia właściwej decyzji
// (np. użycie `get_pod_logs` zamiast `delete_pod`).
//
// Formuła:
// $$\text{TSA} = \frac{T_{correct}}{T_{total}}$$
func ToolSelectionAccuracy(correctSelections, totalSelections int) float64 {
	if totalSelections == 0 {
		return 0.0
	}
	return float64(correctSelections) / float64(totalSelections)
}

// Tokenizer reprezentuje funkcję do zliczania tokenów (np. owrapowaną bibliotekę tiktoken lub model HF).
type Tokenizer func(text string) int

// CalculateTokenEfficiency kompleksowo oblicza TE (Token Efficiency) na podstawie surowej odpowiedzi modelu.
//
// Opis: Ważna metryka przy porównywaniu gadatliwości modeli. Automatyzuje ona
// proces ekstrakcji, minifikacji JSON-a oraz ponownej tokenizacji zminifikowanego
// ładunku (payload), minimalizując w ten sposób szum i formatowanie.
//
// Formuła:
// $$\text{TE} = \frac{\text{Tokens}_{payload}}{\text{Tokens}_{total}}$$
//
// Academic Methodology:
// To eliminate subjective bias in evaluating model verbosity, the Token Efficiency (TE) metric was calculated deterministically.
// We define the 'useful payload' strictly as the machine-actionable JSON parameters required by the Model Context Protocol (MCP).
// For each successful tool call, the extracted JSON object was structurally minified (stripping whitespace and line breaks) to establish the theoretical minimum information entropy.
// This minified string was then re-tokenized using the native tokenizer of the respective model (e.g., tiktoken for OpenAI models, HuggingFace tokenizers for Qwen).
// The TE is the ratio of these payload tokens to the total completion tokens reported by the API.
// If a model failed to produce valid, parsable JSON, its TE for that task was recorded as 0, penalizing the generation of non-actionable text.
func CalculateTokenEfficiency(rawJSON string, totalCompletionTokens int, tokenize Tokenizer) float64 {
	if totalCompletionTokens == 0 {
		return 0.0
	}

	// 1. Ekstrakcja i minifikacja do postaci "Machine-Actionable Payload"
	minifiedPayload, err := ExtractMachineActionablePayload(rawJSON)
	if err != nil {
		// Jeśli JSON nie jest poprawny (np. model zhalucynował zły format), TE wynosi 0.
		return 0.0
	}

	// 2. Re-tokenizacja zminifikowanego tekstu natywnym tokenizerem
	payloadTokens := tokenize(minifiedPayload)

	// 3. Obliczenie efektywności
	return float64(payloadTokens) / float64(totalCompletionTokens)
}

// ExtractMachineActionablePayload wyciąga i minifikuje parametry JSON niezbędne dla MCP.
//
// Opis: Realizuje założenie minimalnej entropii informacji (theoretical minimum information entropy).
// Pomyślne zminifikowanie JSON-a gwarantuje brak sztucznego zawyżania użytecznych tokenów
// przez formatowanie (spacje, taby, znaki nowej linii).
//
// Formuła minimalizacji szumu:
// $$ P_{minified} = \min_{whitespace} \text{JSON}(P_{raw}) $$
func ExtractMachineActionablePayload(rawJSON string) (string, error) {
	var parsed interface{}
	// json.Unmarshal używa wbudowanego parsera do zdekodowania JSON-a
	err := json.Unmarshal([]byte(rawJSON), &parsed)
	if err != nil {
		// Zgodnie z metodologią, jeśli nie jest poprawnym JSON-em, traktujemy to jako błąd (TE = 0)
		return "", err
	}
	// json.Marshal domyślnie minifikuje strukturę (usuwa spacje i taby)
	minified, err := json.Marshal(parsed)
	if err != nil {
		return "", err
	}
	return string(minified), nil
}

// ContextHallucinationRate oblicza wskaźnik halucynacji argumentów.
//
// Opis: W kontekście dużego szumu informacyjnego i optymalizacji K8s MCP kluczowe
// jest, aby model nie zmyślał parametrów, które nie istnieją w kontekście
// (np. nieistniejąca nazwa Poda wyciągnięta z wag zamiast z RAG-a).
// Metryka ta sprawdza jaki ułamek argumentów narzędzia nie występował w dokumencie.
//
// Formuła:
// $$\text{CHR} = \frac{A_{hallucinated}}{A_{total}}$$
// Gdzie $A_{hallucinated}$ to liczba argumentów nieistniejących w kontekście,
// a $A_{total}$ to całkowita liczba wygenerowanych argumentów.
func ContextHallucinationRate(hallucinatedArgs, totalArgs int) float64 {
	if totalArgs == 0 {
		return 0.0
	}
	return float64(hallucinatedArgs) / float64(totalArgs)
}

// SchemaComplianceRate oblicza wskaźnik pełnej zgodności ze schematem narzędzia.
//
// Opis: SVR mierzy tylko poprawność składniową JSON-a (czy to w ogóle JSON).
// SCR (Schema Compliance Rate) określa odsetek odpowiedzi, w których
// payload w 100% pasuje do oczekiwanego schematu JSON narzędzia MCP (JSON Schema),
// posiadając wszystkie wymagane pola i poprawne typy bez dodatkowych halucynacji.
//
// Formuła:
// $$\text{SCR} = \frac{C_{schema\_valid}}{C_{total}}$$
func SchemaComplianceRate(schemaValidResponses, totalResponses int) float64 {
	if totalResponses == 0 {
		return 0.0
	}
	return float64(schemaValidResponses) / float64(totalResponses)
}

// ContextDensityScore (CDS) mierzy stopień wykorzystania dostarczonego kontekstu RAG.
//
// Opis: Weryfikuje, czy model potrafi wyekstrahować kluczowe informacje z dużych manifestów K8s.
// Wysoki CDS przy niskim CHR (Hallucination) dowodzi wyższej inteligencji analitycznej modelu.
//
// Formuła:
// $$ \text{CDS} = \frac{T_{relevant}}{T_{context}} $$
// Gdzie $T_{relevant}$ to liczba tokenów z kontekstu (np. nazwy podów, selektory)
// faktycznie użytych w poprawnym wywołaniu MCP, a $T_{context}$ to całkowita długość okna kontekstowego.
func ContextDensityScore(relevantTokens, contextWindowTokens int) float64 {
	if contextWindowTokens == 0 {
		return 0.0
	}
	return float64(relevantTokens) / float64(contextWindowTokens)
}

// LatencyToActionEfficiency (LAE) definiuje sprawność operacyjną modelu.
//
// Opis: Kluczowa metryka dla IEEE Access. Pokazuje relację sukcesu wykonania (ESR)
// do czasu odpowiedzi. Pozwala wykazać przewagę modeli Qwen/DeepSeek (niski latency)
// nad modelami takimi jak Anthropic/Vertex w zadaniach Real-time DevOps.
//
// Formuła:
// $$ \text{LAE} = \frac{\text{ESR}}{L_{avg}} $$
// Gdzie $L_{avg}$ to średni czas do wykonania akcji (Latency) wyrażony w sekundach.
func LatencyToActionEfficiency(esr float64, avgLatencySeconds float64) float64 {
	if avgLatencySeconds <= 0 {
		return 0.0
	}
	return esr / avgLatencySeconds
}

// =============================================================================
// NOWE METRYKI — wymagane przez recenzentów IEEE/ACM/Elsevier
// =============================================================================

// FirstCallSuccessRate (FCSR) mierzy zdolność modelu do rozwiązania zadania
// bez żadnego retry ani follow-up wywołania.
//
// Opis: Kluczowa metryka dla Real-Time DevOps. Odpowiada na pytanie:
// "Czy model rozwiązał problem za PIERWSZYM razem?" — co jest fundamentalne
// w środowiskach produkcyjnych K8s gdzie retry ma realny koszt operacyjny.
// Recenzenci IEEE zawsze pytają o tę metrykę przy porównaniach LLM.
//
// Formuła:
// $$\text{FCSR} = \frac{T_{solved\_in\_1\_call}}{T_{total}}$$
func FirstCallSuccessRate(solvedInFirstCall, totalTasks int) float64 {
	if totalTasks == 0 {
		return 0.0
	}
	return float64(solvedInFirstCall) / float64(totalTasks)
}

// MeanTimeToRecovery (MTTR) mierzy średni czas od wykrycia błędu przez LLM
// do przywrócenia poprawnego stanu klastra K8s.
//
// Opis: Adaptacja klasycznej metryki SRE (Site Reliability Engineering) do ewaluacji LLM.
// Jest to innowacyjna kontrybucja — mapowanie metryki operacyjnej na zdolność modelu.
// Niskie MTTR przy zachowanym ESR jest dowodem przewagi małych modeli (Qwen/DeepSeek)
// w zadaniach edge DevOps nad dużymi modelami z wysoką latencją (Anthropic/Vertex).
//
// Formuła:
// $$\text{MTTR} = \frac{1}{N} \sum_{i=1}^{N} (t_{resolved,i} - t_{detected,i})$$
// Gdzie czasy są wyrażone w sekundach.
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

// LatencyPercentile oblicza percentyl latencji (p50, p95, p99) z próbki pomiarów.
//
// Opis: Zastępuje lub uzupełnia średnią latencję w LAE. Średnia jest wrażliwa na
// outliers (np. cold start modelu, throttling API). Recenzenci IEEE Access
// wymagają raportowania p95/p99 przy porównaniu modeli w warunkach produkcyjnych.
// p50 = mediana (typowy przypadek), p95 = ogon rozkładu (SLA), p99 = worst-case.
//
// Użycie: percentile = 95.0 dla p95, 99.0 dla p99, 50.0 dla mediany.
func LatencyPercentile(latenciesSec []float64, percentile float64) float64 {
	if len(latenciesSec) == 0 || percentile < 0 || percentile > 100 {
		return 0.0
	}
	sorted := make([]float64, len(latenciesSec))
	copy(sorted, latenciesSec)
	sort.Float64s(sorted)

	// Interpolacja liniowa (metoda zgodna z numpy percentile)
	idx := (percentile / 100.0) * float64(len(sorted)-1)
	lower := int(math.Floor(idx))
	upper := int(math.Ceil(idx))
	if lower == upper {
		return sorted[lower]
	}
	frac := idx - float64(lower)
	return sorted[lower]*(1-frac) + sorted[upper]*frac
}

// RAGPrecisionAtK oblicza precyzję retrievalu RAG dla top-K zwróconych dokumentów.
//
// Opis: KRYTYCZNA metryka dla artykułów o RAG w ACM TOIS i IP&M.
// Mierzy jakość retrievera niezależnie od jakości generacji modelu LLM.
// Bez tej metryki nie można odróżnić czy błędy LLM wynikają ze złego
// rozumienia czy ze złego dostarczenia kontekstu przez retriever.
// retrievedRelevant = liczba dokumentów w top-K które są ground-truth istotne.
//
// Formuła:
// $$\text{P@K} = \frac{|\text{relevant} \cap \text{retrieved@K}|}{K}$$
func RAGPrecisionAtK(retrievedRelevantCount, k int) float64 {
	if k == 0 {
		return 0.0
	}
	return float64(retrievedRelevantCount) / float64(k)
}

// RAGRecallAtK oblicza recall retrievalu RAG dla top-K zwróconych dokumentów.
//
// Opis: Uzupełnienie RAGPrecisionAtK. Razem tworzą pełny obraz jakości
// retrieval pipeline dla manifestów K8s w RAG-u. Recall jest szczególnie
// ważny gdy pominięcie kluczowego manifestu (np. ConfigMap z credentials)
// bezpośrednio powoduje błąd MCP call.
// totalRelevant = całkowita liczba istotnych dokumentów w korpusie.
//
// Formuła:
// $$\text{R@K} = \frac{|\text{relevant} \cap \text{retrieved@K}|}{|\text{relevant}|}$$
func RAGRecallAtK(retrievedRelevantCount, totalRelevantInCorpus int) float64 {
	if totalRelevantInCorpus == 0 {
		return 0.0
	}
	return float64(retrievedRelevantCount) / float64(totalRelevantInCorpus)
}

// RAGFScoreAtK oblicza harmoniczną średnią P@K i R@K (F1@K).
//
// Opis: Syntetyczna metryka RAG łącząca precyzję i recall w jedną liczbę.
// Przydatna do porównań tabelarycznych w artykule gdy miejsce jest ograniczone.
// Beta=1.0 daje klasyczne F1 (równa waga precyzji i recallu).
// Beta=0.5 faworyzuje precyzję (mniej szumu w kontekście).
// Beta=2.0 faworyzuje recall (nie pomijaj kluczowych manifestów).
func RAGFScoreAtK(precisionAtK, recallAtK, beta float64) float64 {
	if precisionAtK+recallAtK == 0 {
		return 0.0
	}
	betaSq := beta * beta
	return (1 + betaSq) * (precisionAtK * recallAtK) / (betaSq*precisionAtK + recallAtK)
}

// RecoveryPlanRationality (RPR) ocenia optymalność planu naprawczego zaproponowanego przez model.
//
// Opis: Mierzy podobieństwo sekwencji narzędzi MCP zaproponowanej przez model
// do optymalnej sekwencji narzędzi (ground-truth). Używa znormalizowanej
// odległości edycji (Levenshtein) na sekwencjach nazw narzędzi.
// RPR=1.0 oznacza idealny plan, RPR=0.0 oznacza całkowicie błędny plan.
//
// Formuła:
// $$\text{RPR} = 1 - \frac{\text{EditDist}(S_{model}, S_{optimal})}{\max(|S_{model}|, |S_{optimal}|)}$$
func RecoveryPlanRationality(modelToolSequence, optimalToolSequence []string) float64 {
	dist := levenshteinDistance(modelToolSequence, optimalToolSequence)
	maxLen := math.Max(float64(len(modelToolSequence)), float64(len(optimalToolSequence)))
	if maxLen == 0 {
		return 1.0 // Oba puste = idealny plan trywialny
	}
	return 1.0 - float64(dist)/maxLen
}

// levenshteinDistance oblicza odległość edycji między dwoma sekwencjami stringów.
// Używana wewnętrznie przez RecoveryPlanRationality.
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

// MultiStepFaithfulnessScore (MFS) mierzy, czy każdy krok agenta bazuje
// na faktycznym wyniku poprzedniego kroku, a nie na "zgadywaniu".
//
// Opis: Adaptacja metryki RAGAS Faithfulness do środowisk agentycznych MCP.
// W wielokrokowych taskach K8s (np. diagnoza → patch → weryfikacja) model
// może "halucynować" wynik kroku N i użyć go w kroku N+1, mimo że
// faktyczny wynik MCP był inny. MFS wykrywa takie odsprzężenie.
// groundedSteps = kroki gdzie argumenty kroku N+1 wynikają z odpowiedzi kroku N.
//
// Formuła:
// $$\text{MFS} = \frac{S_{grounded}}{S_{total}}$$
func MultiStepFaithfulnessScore(groundedSteps, totalSteps int) float64 {
	if totalSteps == 0 {
		return 0.0
	}
	return float64(groundedSteps) / float64(totalSteps)
}

// ErrorRecoveryRate (ERR) mierzy zdolność modelu do samodzielnej korekty błędu MCP.
//
// Opis: Mierzy self-correction — czy model po otrzymaniu error response
// (np. "pod not found", "invalid namespace") potrafi samodzielnie
// wygenerować poprawne wywołanie bez dodatkowej interwencji użytkownika.
// Jest to advantage mniejszych modeli (Qwen/DeepSeek) przy dobrze
// zbudowanym RAG-u, który dostarcza kontekst naprawczy.
//
// Formuła:
// $$\text{ERR} = \frac{T_{self\_corrected}}{T_{with\_initial\_error}}$$
func ErrorRecoveryRate(selfCorrectedTasks, tasksWithInitialError int) float64 {
	if tasksWithInitialError == 0 {
		return 0.0
	}
	return float64(selfCorrectedTasks) / float64(tasksWithInitialError)
}

// ContextTruncationRate (CTR) mierzy, jak często manifest K8s nie mieści się
// w oknie kontekstowym i musi być obcinany przed wysłaniem do modelu.
//
// Opis: Krytyczna metryka dla porównania małych modeli (krótsze okno kontekstowe,
// np. 8k tokenów) vs gigantów (128k+ tokenów). Wysokie CTR u małego modelu
// pokazuje ograniczenie które RAG może kompensować przez selektywny retrieval.
// Jest to kluczowy argument dla Twojej tezy o RAG-assisted small models.
//
// Formuła:
// $$\text{CTR} = \frac{T_{truncated}}{T_{total}}$$
func ContextTruncationRate(truncatedTasks, totalTasks int) float64 {
	if totalTasks == 0 {
		return 0.0
	}
	return float64(truncatedTasks) / float64(totalTasks)
}

// =============================================================================
// STATYSTYKI — wymóg recenzentów, nie opcja
// =============================================================================

// BootstrapConfidenceInterval oblicza 95% przedział ufności metodą bootstrap
// dla dowolnej metryki wyrażonej jako stosunek sukcesów do prób.
//
// Opis: WYMÓG recenzentów ACM TOIS, IP&M i IEEE Access przy porównaniu modeli.
// Bez CI nie można twierdzić że model A jest lepszy od modelu B —
// różnica może być przypadkowa przy małej próbce (n<100 eksperymentów).
// Zwraca [dolna_granica, górna_granica] dla poziomu ufności 1-alpha (domyślnie 0.05 → 95% CI).
//
// Metodologia: Parametryczny bootstrap na rozkładzie Binomialnym.
// Dla każdej próbki losujemy successes ~ Binomial(n, p_observed).
// nBootstrap=10000 jest standardem akademickim.
//
// Zwraca: [lowerBound, upperBound]
func BootstrapConfidenceInterval(successes, total, nBootstrap int, alpha float64, rng func() float64) [2]float64 {
	if total == 0 || nBootstrap == 0 {
		return [2]float64{0.0, 0.0}
	}
	pObs := float64(successes) / float64(total)
	samples := make([]float64, nBootstrap)

	for i := range samples {
		// Symulacja próbki bootstrap: losujemy n obserwacji z rozkładu Bernoulli(p_obs)
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

// CliffsDelta oblicza efekt wielkości Cliff's δ między dwoma próbkami wyników.
//
// Opis: WYMÓG dla IEEE Access i ACM TOIS przy porównaniach LLM.
// Testy statystyczne (t-test, Wilcoxon) informują czy różnica jest istotna,
// ale nie mówią jak DUŻA jest różnica. Cliff's δ ∈ [-1, 1]:
//   - |δ| < 0.147 → efekt znikomy (negligible)
//   - |δ| < 0.330 → efekt mały (small)
//   - |δ| < 0.474 → efekt średni (medium)
//   - |δ| ≥ 0.474 → efekt duży (large)
//
// Interpretacja dla artykułu: δ > 0 oznacza że modelA stochastycznie dominuje modelB.
// Preferowany nad Cohen's d dla danych nieciągłych (jak ESR, FCSR).
//
// Formuła:
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

// CliffsEffectSizeLabel zwraca tekstowy label wielkości efektu wg standardu Romano et al.
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

// =============================================================================
// POMOCNICZE — ekstrakcja tokenów z kontekstu RAG
// =============================================================================

// CountRelevantTokensFromContext liczy tokeny z payload MCP które faktycznie
// pochodzą z dostarczonego kontekstu RAG (token overlap).
//
// Opis: Operacyjna implementacja T_relevant dla metryki CDS.
// Używa prostego token overlap (split by whitespace) między
// zminifikowanym payloadem MCP a surowym tekstem kontekstu RAG.
// W artykule można tę metodę rozszerzyć o BM25 lub cosine similarity
// dla bardziej zaawansowanej wersji.
func CountRelevantTokensFromContext(mcpPayloadJSON, ragContextText string, tokenize Tokenizer) int {
	// Tokenizacja przez split na słowa (uproszczona wersja dla demonstracji)
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

func tokenizeToWords(text string) []string {
	// Prosta tokenizacja — w produkcji zastąp natywnym tokenizerem modelu
	words := strings.FieldsFunc(text, func(r rune) bool {
		return r == ' ' || r == '\n' || r == '\t' || r == '"' ||
			r == '{' || r == '}' || r == ':' || r == ','
	})
	result := make([]string, 0, len(words))
	for _, w := range words {
		if len(w) > 1 { // Filtruj jednoznakowe tokeny interpunkcyjne
			result = append(result, w)
		}
	}
	return result
}

// =============================================================================
// UZUPEŁNIENIE DLA ACM TOIS / IP&M: ZAAWANSOWANY INFORMATION RETRIEVAL
// =============================================================================

// MeanReciprocalRank (MRR) mierzy jak szybko RAG dostarcza pierwszą trafną odpowiedź.
//
// Opis: Kluczowa metryka IR uzupełniająca P@K i R@K. O ile P@K i R@K mierzą
// "czy w ogóle dostarczono właściwy dokument", MRR mierzy "jak wysoko w prompcie
// ten dokument się znalazł". Małe modele (Qwen/DeepSeek) są bardzo wrażliwe na
// pozycję kluczowego manifestu K8s w kontekście — jeśli poprawny chunk jest
// na pozycji 5 zamiast 1, skuteczność spada (patrz: LostInTheMiddleVulnerability).
// Recenzenci ACM TOIS i IP&M traktują MRR jako standard dla systemów IR.
//
// Formuła:
// $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
// Gdzie rank_i to pozycja (1-indexed) pierwszego trafnego dokumentu w i-tym zapytaniu.
// Jeśli żaden dokument nie był trafny dla zapytania i, wkład do sumy wynosi 0.
func MeanReciprocalRank(ranksOfFirstRelevant []int) float64 {
	if len(ranksOfFirstRelevant) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, rank := range ranksOfFirstRelevant {
		if rank > 0 {
			sum += 1.0 / float64(rank)
		}
		// rank <= 0 oznacza brak trafnego dokumentu w wynikach — wkład = 0
	}
	return sum / float64(len(ranksOfFirstRelevant))
}

// NDCGAtK (Normalized Discounted Cumulative Gain) ocenia jakość rankingu RAG-a.
//
// Opis: Złoty standard oceny rankingu w ACM TOIS. Wymaga przypisania stopni
// trafności (relevance grades) do każdego zwróconego dokumentu, np.:
//   - 3 = idealny manifest (zawiera dokładnie szukany Pod/Deployment)
//   - 1 = powiązany (ten sam namespace, inny zasób)
//   - 0 = szum (niezwiązany YAML)
//
// Przewaga nad P@K: NDCG karze za umieszczenie trafnego dokumentu na niskiej
// pozycji (discounting przez log2). Udowadnia, że Twój retriever nie tylko
// znajduje właściwe manifesty K8s, ale układa je optymalnie dla modelu.
// idealRelevances to posortowana malejąco lista idealnych stopni trafności
// (teoretyczna najlepsza kolejność — służy do normalizacji przez IDCG).
//
// Formuła:
// $$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}, \quad
//
//	\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$
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

// LostInTheMiddleVulnerability (LMV) mierzy podatność modelu na fenomen
// "Lost in the Middle" — ignorowanie informacji w środku promptu.
//
// Opis: Innowacyjna metryka specyficzna dla ewaluacji LLM w kontekście RAG.
// Badania (Liu et al., 2023) pokazują że modele językowe (zwłaszcza mniejsze)
// znacznie gorzej wykorzystują informacje umieszczone w środku długiego promptu
// niż na jego początku lub końcu. W kontekście K8s: jeśli kluczowy manifest
// Poda jest na pozycji 3/5 w kontekście RAG, małe modele mogą go pominąć.
//
// Interpretacja wyników:
//   - LMV > 0.10 → model wykazuje silną wrażliwość — wymaga reorderu chunków RAG
//   - LMV ≈ 0    → model równomiernie przetwarza cały kontekst
//   - LMV < 0    → model paradoksalnie lepiej radzi sobie ze środkiem (rzadkie)
//
// Metodologia eksperymentu: uruchom te same taski K8s trzykrotnie, za każdym
// razem zmieniając pozycję kluczowego manifestu (początek/środek/koniec promptu).
// esrEdges = ESR gdy manifest na pozycji 1 lub K (skraje kontekstu).
// esrMiddle = ESR gdy manifest na pozycji środkowej.
//
// Formuła:
// $$\text{LMV} = \text{ESR}_{edges} - \text{ESR}_{middle}$$
func LostInTheMiddleVulnerability(esrEdges, esrMiddle float64) float64 {
	return esrEdges - esrMiddle
}

// =============================================================================
// UZUPEŁNIENIE DLA IEEE ACCESS / ESWA: EKONOMIA I BEZPIECZEŃSTWO
// =============================================================================

// CostEfficiencyScore (CES) mierzy liczbę udanych akcji MCP na jednostkę kosztu.
//
// Opis: Kluczowy argument w artykułach IEEE Access i ESWA porównujących modele
// pod kątem production-readiness. Qwen/DeepSeek mogą mieć ESR niższe o 5-10pp
// od modeli Anthropic/Vertex, ale jeśli ich koszt API jest 50-100x niższy,
// CES jest gigantycznym argumentem za stosowaniem małych modeli w Edge K8s.
// Wynik w jednostkach: liczba udanych akcji MCP per 1 USD wydany na API.
// Dla lokalnych modeli (Ollama, vLLM) totalCostUSD → 0, CES → +Inf,
// co jest matematycznie poprawne i fabularnie silnym argumentem w artykule.
//
// Formuła:
// $$\text{CES} = \frac{E_{success}}{\text{TotalCost}_{\$}}$$
func CostEfficiencyScore(successfulExecutions int, totalCostUSD float64) float64 {
	if totalCostUSD <= 0 {
		// Lokalny model (np. Qwen na vLLM) = koszt infrastruktury, nie per-token.
		// Zwracamy +Inf jako sygnał — w artykule opisz to jako "not applicable (local deployment)".
		return math.Inf(1)
	}
	return float64(successfulExecutions) / totalCostUSD
}

// DestructiveActionAttemptRate (DAAR) mierzy wskaźnik prób wykonania niebezpiecznych akcji K8s.
//
// Opis: Metryka bezpieczeństwa wymagana przez recenzentów ESWA i IEEE Access
// przy artykułach o autonomicznych agentach w środowiskach produkcyjnych.
// Mierzy, jak często model (wskutek halucynacji lub złego RAG-a) próbuje użyć
// narzędzi z czarnej listy bez autoryzacji, np.:
//   - delete_namespace (usunięcie całego namespace'u)
//   - delete_pod bez flagi --grace-period
//   - patch_deployment z image: "latest" (niebezpieczna praktyka)
//
// Niskie DAAR (idealnie 0%) jest wymogiem do wdrożenia produkcyjnego i silnym
// argumentem że RAG z filtrowaniem intencji zwiększa bezpieczeństwo operacyjne.
// destructiveAttempts = liczba wywołań narzędzi z czarnej listy zaobserwowanych
// podczas całego eksperymentu, niezależnie od tego czy się powiodły.
//
// Formuła:
// $$\text{DAAR} = \frac{A_{destructive}}{A_{total}}$$
func DestructiveActionAttemptRate(destructiveAttempts, totalActionAttempts int) float64 {
	if totalActionAttempts == 0 {
		return 0.0
	}

	return float64(destructiveAttempts) / float64(totalActionAttempts)
}

// ContextCompressionRatio (CCR) mierzy efektywność optymalizacji kontekstu RAG.
//
// Opis: Jeśli stosujesz innowacyjne metody budowy i minifikacji kontekstu K8s
// (np. usuwanie pola `managedFields` z manifestów, usuwanie pustych annotacji,
// kompresja wieloliniowych YAML-i do JSON), CCR pokazuje wymierny zysk:
// o ile % skróciłeś prompt nie tracąc informacji potrzebnych do wywołania MCP.
// Bezpośrednio przekłada się na niższy CTR (Context Truncation Rate) i niższy
// koszt API (mniej tokenów promptu = niższy totalCostUSD w CES).
// Wartość 0.0 = brak kompresji, 0.5 = prompt o 50% krótszy, 1.0 = niemożliwe.
//
// Formuła:
// $$\text{CCR} = \frac{T_{original} - T_{compressed}}{T_{original}}$$
func ContextCompressionRatio(originalTokens, compressedTokens int) float64 {
	if originalTokens == 0 {
		return 0.0
	}

	if compressedTokens >= originalTokens {
		return 0.0 // Kompresja nie może zwiększyć rozmiaru — zwróć 0 zamiast wartości ujemnej
	}

	return float64(originalTokens-compressedTokens) / float64(originalTokens)
}
