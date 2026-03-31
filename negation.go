package llmbench

import (
	"strings"
	"unicode"
)

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
