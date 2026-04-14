package llmbench

import "strings"

// SMARTString computes the SMART metric using string-based sentence matching.
// It treats sentences as basic units and uses ROUGE-L between sentence pairs
// as the matching function (Amplayo et al., 2022).
func SMARTString(reference, candidate string) float64 {
	refSents := splitSentences(reference)
	candSents := splitSentences(candidate)

	if len(refSents) == 0 || len(candSents) == 0 {
		return 0
	}

	// Precision: for each candidate sentence, find best matching reference sentence.
	var precisionSum float64
	for _, cs := range candSents {
		best := 0.0
		for _, rs := range refSents {
			if s := ROUGEL(rs, cs); s > best {
				best = s
			}
		}
		precisionSum += best
	}
	precision := precisionSum / float64(len(candSents))

	// Recall: for each reference sentence, find best matching candidate sentence.
	var recallSum float64
	for _, rs := range refSents {
		best := 0.0
		for _, cs := range candSents {
			if s := ROUGEL(rs, cs); s > best {
				best = s
			}
		}
		recallSum += best
	}
	recall := recallSum / float64(len(refSents))

	if precision+recall == 0 {
		return 0
	}
	return 2 * precision * recall / (precision + recall)
}

// MaxSMARTString returns the best SMART-String score of candidate against all references.
func MaxSMARTString(references []string, candidate string) float64 {
	best := 0.0
	for _, ref := range references {
		if s := SMARTString(ref, candidate); s > best {
			best = s
		}
	}
	return best
}

type SMARTStringScorer struct{}

func NewSMARTStringScorer() *SMARTStringScorer { return &SMARTStringScorer{} }

func (s *SMARTStringScorer) Score(entries []Entry) (ScoreOutput, error) {
	var out ScoreOutput
	for _, e := range entries {
		for mi, mach := range e.MachineSummaries {
			out.Scores = append(out.Scores, MaxSMARTString(e.HumanSummaries, mach))
			out.Relevance = append(out.Relevance, e.Relevance[mi])
			out.Coherence = append(out.Coherence, e.Coherence[mi])
			out.Fluency = append(out.Fluency, e.Fluency[mi])
			out.Consistency = append(out.Consistency, e.Consistency[mi])
		}
	}
	return out, nil
}

// splitSentences splits text into sentences on '.', '!', '?'.
func splitSentences(text string) []string {
	var sents []string
	var current strings.Builder

	for i, r := range text {
		current.WriteRune(r)
		if (r == '.' || r == '!' || r == '?') &&
			(i+1 >= len(text) || text[i+1] == ' ' || text[i+1] == '\n') {
			s := strings.TrimSpace(current.String())
			if len(s) > 0 {
				sents = append(sents, s)
			}
			current.Reset()
		}
	}
	// Remaining text without terminal punctuation.
	if s := strings.TrimSpace(current.String()); len(s) > 0 {
		sents = append(sents, s)
	}
	return sents
}
