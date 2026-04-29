package metrics

import (
	"strings"
	"unicode"
)

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

// abbreviations holds tokens that, when followed by a period, do NOT terminate
// a sentence. Lowercased for case-insensitive matching.
var abbreviations = map[string]bool{
	"mr": true, "mrs": true, "ms": true, "dr": true, "prof": true,
	"sr": true, "jr": true, "st": true, "mt": true, "rev": true,
	"vs": true, "etc": true, "inc": true, "ltd": true, "co": true,
	"corp": true, "no": true, "vol": true, "fig": true, "p": true,
	"pp": true, "e.g": true, "i.e": true, "u.s": true, "u.k": true,
}

// splitSentences splits text on sentence-terminal punctuation followed by
// whitespace, with a guard against common abbreviations.
//
// Operates on runes (not bytes) for correct UTF-8 handling. The guard avoids
// splitting on "Mr." followed by a name, but is intentionally simple — it
// will not handle every edge case (e.g. nested quotes, ellipses).
func splitSentences(text string) []string {
	runes := []rune(text)
	var sents []string
	var current strings.Builder

	for i, r := range runes {
		current.WriteRune(r)

		if r != '.' && r != '!' && r != '?' {
			continue
		}

		// Need whitespace or end-of-text after punctuation to terminate.
		if i+1 < len(runes) && !unicode.IsSpace(runes[i+1]) {
			continue
		}

		// Abbreviation guard: only for periods (not ! or ?).
		if r == '.' && isAbbreviationEnding(current.String()) {
			continue
		}

		s := strings.TrimSpace(current.String())
		if len(s) > 0 {
			sents = append(sents, s)
		}
		current.Reset()
	}

	if s := strings.TrimSpace(current.String()); len(s) > 0 {
		sents = append(sents, s)
	}
	return sents
}

// isAbbreviationEnding reports whether the buffer ends with a known abbreviation
// followed by a period — in which case the period does not terminate the sentence.
func isAbbreviationEnding(buf string) bool {
	trimmed := strings.TrimRight(buf, ".")
	if trimmed == buf {
		return false
	}
	// Take the last whitespace-separated token before the period.
	fields := strings.Fields(trimmed)
	if len(fields) == 0 {
		return false
	}
	last := strings.ToLower(fields[len(fields)-1])
	return abbreviations[last]
}
