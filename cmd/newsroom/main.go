// cmd/newsroom converts the Newsroom human-evaluation release into the
// JSONL shape every binary in this repository already reads.
//
// WHY. The previous manuscript evaluated on SummEval alone, i.e. on
// CNN/DailyMail news, which is also the corpus whose lead bias the
// metric's positional prior was built around. Reviewer #4's strongest
// remaining objection was that a single corpus cannot test that
// premise. Newsroom (Grusky et al. 2018) is the natural second corpus:
// its human-evaluation subset rates 7 systems on 60 articles along
// coherence, fluency, informativeness and relevance, it comes from a
// different pool of publishers, and its systems span the extractive
// spectrum -- including `lede3`, a literal lead-3 baseline, which makes
// the extractiveness confound directly visible.
//
// Three rows per (article, system) hold the three raters; they are
// averaged, as in the released analyses.
//
// FIELD MAPPING. Newsroom has no consistency dimension; it has
// informativeness. The converter writes informativeness into the
// `consistency` slot because the downstream code is built around
// SummEval's four names, and every table that reports it must say so.
// Newsroom also ships no human reference summaries, so reference-based
// metrics cannot run on it -- only the reference-free ones, which is
// the comparison of interest anyway.
//
// Usage:
//
//	curl -sLO https://raw.githubusercontent.com/lil-lab/newsroom/master/humaneval/newsroom-human-eval.csv
//	go run ./cmd/newsroom -input newsroom-human-eval.csv -output data/newsroom.jsonl
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type entry struct {
	ID               string    `json:"id"`
	Text             string    `json:"text"`
	MachineSummaries []string  `json:"machine_summaries"`
	HumanSummaries   []string  `json:"human_summaries"`
	Coherence        []float64 `json:"coherence"`
	Consistency      []float64 `json:"consistency"`
	Fluency          []float64 `json:"fluency"`
	Relevance        []float64 `json:"relevance"`
}

func main() {
	var input, output string
	flag.StringVar(&input, "input", "newsroom-human-eval.csv", "path to newsroom-human-eval.csv")
	flag.StringVar(&output, "output", "data/newsroom.jsonl", "output JSONL path")
	flag.Parse()

	f, err := os.Open(input)
	if err != nil {
		log.Fatalf("open %s: %v", input, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		log.Fatalf("read csv: %v", err)
	}
	if len(rows) < 2 {
		log.Fatalf("csv has %d rows", len(rows))
	}
	col := map[string]int{}
	for i, h := range rows[0] {
		col[h] = i
	}
	for _, need := range []string{"ArticleID", "System", "ArticleText", "SystemSummary",
		"CoherenceRating", "FluencyRating", "InformativenessRating", "RelevanceRating"} {
		if _, ok := col[need]; !ok {
			log.Fatalf("csv is missing column %q", need)
		}
	}

	// (article, system) -> accumulated ratings over raters
	type key struct{ doc, sys string }
	type agg struct {
		text, summary              string
		coh, flu, inf, rel, raters float64
	}
	byPair := map[key]*agg{}
	var docOrder []string
	seenDoc := map[string]bool{}
	sysSeen := map[string]bool{}

	for _, row := range rows[1:] {
		k := key{row[col["ArticleID"]], row[col["System"]]}
		a, ok := byPair[k]
		if !ok {
			a = &agg{text: row[col["ArticleText"]], summary: row[col["SystemSummary"]]}
			byPair[k] = a
		}
		a.coh += num(row[col["CoherenceRating"]])
		a.flu += num(row[col["FluencyRating"]])
		a.inf += num(row[col["InformativenessRating"]])
		a.rel += num(row[col["RelevanceRating"]])
		a.raters++
		if !seenDoc[k.doc] {
			seenDoc[k.doc] = true
			docOrder = append(docOrder, k.doc)
		}
		sysSeen[k.sys] = true
	}

	systems := make([]string, 0, len(sysSeen))
	for s := range sysSeen {
		systems = append(systems, s)
	}
	// A fixed system order is required: the loader treats the position
	// in machine_summaries as the SystemID that system-level
	// aggregation groups by.
	sort.Strings(systems)
	sort.Strings(docOrder)

	if dir := filepath.Dir(output); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			log.Fatal(err)
		}
	}
	out, err := os.Create(output)
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	enc := json.NewEncoder(out)

	written, skipped := 0, 0
	for _, doc := range docOrder {
		e := entry{ID: doc, HumanSummaries: []string{}}
		complete := true
		for _, sys := range systems {
			a, ok := byPair[key{doc, sys}]
			if !ok || a.raters == 0 {
				complete = false
				break
			}
			if e.Text == "" {
				e.Text = cleanHTML(a.text)
			}
			e.MachineSummaries = append(e.MachineSummaries, cleanHTML(a.summary))
			e.Coherence = append(e.Coherence, a.coh/a.raters)
			e.Consistency = append(e.Consistency, a.inf/a.raters) // informativeness
			e.Fluency = append(e.Fluency, a.flu/a.raters)
			e.Relevance = append(e.Relevance, a.rel/a.raters)
		}
		if !complete {
			skipped++
			continue
		}
		if err := enc.Encode(e); err != nil {
			log.Fatal(err)
		}
		written++
	}
	fmt.Printf("newsroom: %d articles x %d systems written to %s (%d articles skipped as incomplete)\n",
		written, len(systems), output, skipped)
	fmt.Printf("systems in fixed order: %s\n", strings.Join(systems, ", "))
	fmt.Println("note: the consistency field carries Newsroom's INFORMATIVENESS rating;")
	fmt.Println("      Newsroom ships no reference summaries, so reference-based metrics cannot run on it.")
}

// cleanHTML removes the paragraph markup the release carries inside the
// article text. Sentence splitting downstream keys on punctuation, and
// a stray "</p><p>" glues two sentences into one.
func cleanHTML(s string) string {
	s = strings.NewReplacer("</p><p>", " ", "<p>", " ", "</p>", " ", "<br>", " ", "<br/>", " ").Replace(s)
	for {
		i := strings.Index(s, "<")
		if i < 0 {
			break
		}
		j := strings.Index(s[i:], ">")
		if j < 0 {
			break
		}
		s = s[:i] + " " + s[i+j+1:]
	}
	return strings.Join(strings.Fields(s), " ")
}

func num(s string) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0
	}
	return v
}
