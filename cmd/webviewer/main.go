// cmd/serve serves a simple HTML view of all metric reports in output/.
//
// Usage:
//
//	go run ./cmd/serve
//	# Open http://localhost:8080
//
// Three tabs:
//   - Summary-level correlations
//   - System-level correlations
//   - Run metadata (samples, runtime, throughput, timestamps)
//
// Tailwind via CDN, no build pipeline. Re-reads JSON on every request.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/mikolajsemeniuk/llmbench/pkg/eval"
)

var (
	inputDir string
	addr     string
)

// ── Display ordering matches cmd/tables ───────────────────────────────

var dimensions = []string{"coherence", "consistency", "fluency", "relevance"}

var dimensionShort = map[string]string{
	"coherence":   "Coh",
	"consistency": "Con",
	"fluency":     "Flu",
	"relevance":   "Rel",
}

var metricOrder = []string{
	"bleu", "rouge", "chrf", "meteor", "smartstring",
	"embedscorer",
	"bertscore", "moverscore", "smartmodel",
	"bartscore", "gptscore", "unieval", "geval",
	"bgs",
}

var metricDisplayName = map[string]string{
	"bleu":        "BLEU",
	"rouge":       "ROUGE-L",
	"chrf":        "ChrF",
	"meteor":      "METEOR",
	"smartstring": "SMART-String",
	"embedscorer": "EmbedScorer",
	"bertscore":   "BERTScore",
	"moverscore":  "MoverScore",
	"smartmodel":  "SMART-Model",
	"bartscore":   "BARTScore",
	"gptscore":    "GPTScore",
	"unieval":     "UniEval",
	"geval":       "G-Eval",
	"bgs":         "BGS",
}

var dimensionalMetrics = []struct {
	prefix      string
	displayName string
}{
	{"geval_", "G-Eval"},
	{"unieval_", "UniEval"},
}

// ── Template data types ────────────────────────────────────────────────

type Cell struct {
	Spearman    float64
	Kendall     float64
	SpearmanLow float64
	SpearmanHi  float64
	KendallLow  float64
	KendallHi   float64
	HasCI       bool
}

type CellOrEmpty struct {
	Present bool
	Cell    Cell
}

type Row struct {
	Label string
	Cells [4]CellOrEmpty // ordered: coherence, consistency, fluency, relevance
}

type LevelTable struct {
	Level string
	Rows  []Row
}

// MetadataRow is one row of the run metadata table — covers everything
// from a Report except the per-sample Scores slice (which is too big).
//
// Dimensional metrics (G-Eval, UniEval) appear as 4 separate rows because
// they're 4 separate runs, even though they're collapsed in the correlation
// view. This is intentional: timing matters per run.
type MetadataRow struct {
	Metric         string
	DisplayLabel   string
	Norm           string
	Samples        int
	RuntimeSec     float64
	RuntimeHuman   string  // "1h 23m" or "45s"
	Throughput     float64 // samples/sec
	ThroughputText string  // "12.3/s" or "0.5/s"
	Timestamp      string  // raw RFC3339
	TimestampHuman string  // "2026-05-03 14:30 UTC"
	Age            string  // "3 hours ago"
}

type PageData struct {
	Title    string
	Summary  LevelTable
	System   LevelTable
	Metadata []MetadataRow
}

// ── Main ───────────────────────────────────────────────────────────────

func main() {
	flag.StringVar(&inputDir, "input", "output", "directory containing metric JSON reports")
	flag.StringVar(&addr, "addr", ":8080", "HTTP listen address")
	flag.Parse()

	tmpl, err := template.New("page").Funcs(templateFuncs()).Parse(pageHTML)
	if err != nil {
		log.Fatalf("parse template: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		data, err := buildPageData(inputDir)
		if err != nil {
			http.Error(w, fmt.Sprintf("build page: %v", err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := tmpl.Execute(w, data); err != nil {
			log.Printf("template execute: %v", err)
		}
	})

	log.Printf("Serving metrics view from %s on http://localhost%s", inputDir, addr)
	log.Printf("Reads %s/*.json on every request — edit & refresh.", inputDir)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}

// ── Data assembly ──────────────────────────────────────────────────────

func buildPageData(dir string) (PageData, error) {
	reports, err := loadReports(dir)
	if err != nil {
		return PageData{}, err
	}
	if len(reports) == 0 {
		return PageData{}, fmt.Errorf("no JSON reports in %s", dir)
	}

	return PageData{
		Title:    "llmbench results",
		Summary:  LevelTable{Level: "summary", Rows: buildRows(reports, "summary")},
		System:   LevelTable{Level: "system", Rows: buildRows(reports, "system")},
		Metadata: buildMetadataRows(reports),
	}, nil
}

func loadReports(dir string) ([]eval.Report, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, fmt.Errorf("glob: %w", err)
	}
	sort.Strings(matches)

	reports := make([]eval.Report, 0, len(matches))
	for _, path := range matches {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		var r eval.Report
		if err := json.Unmarshal(raw, &r); err != nil {
			return nil, fmt.Errorf("decode %s: %w", path, err)
		}
		reports = append(reports, r)
	}
	return reports, nil
}

// ── Correlation tables ─────────────────────────────────────────────────

func buildRows(reports []eval.Report, level string) []Row {
	byMetric := make(map[string]eval.Report, len(reports))
	for _, r := range reports {
		byMetric[r.Metric] = r
	}

	rows := make([]Row, 0, len(reports))
	seen := make(map[string]bool)

	for _, dm := range dimensionalMetrics {
		if row, ok := collapseDimensional(byMetric, dm.prefix, dm.displayName, level); ok {
			rows = append(rows, row)
			for _, d := range dimensions {
				seen[dm.prefix+d] = true
			}
		}
	}

	for _, r := range reports {
		if seen[r.Metric] || isDimensionalPartial(r.Metric) {
			continue
		}
		rows = append(rows, Row{
			Label: displayName(r.Metric),
			Cells: cellsFromAllDimensions(r, level),
		})
	}

	sortRows(rows)
	return rows
}

func collapseDimensional(byMetric map[string]eval.Report, prefix, displayName, level string) (Row, bool) {
	var cells [4]CellOrEmpty
	for i, dim := range dimensions {
		rep, ok := byMetric[prefix+dim]
		if !ok {
			return Row{}, false
		}
		corr := selectLevel(rep, level)
		var matched *eval.Dimension
		for j := range corr.Dimensions {
			if corr.Dimensions[j].Name == dim {
				matched = &corr.Dimensions[j]
				break
			}
		}
		if matched == nil {
			return Row{}, false
		}
		cells[i] = CellOrEmpty{Present: true, Cell: makeCell(*matched)}
	}
	return Row{Label: displayName, Cells: cells}, true
}

func cellsFromAllDimensions(r eval.Report, level string) [4]CellOrEmpty {
	var cells [4]CellOrEmpty
	corr := selectLevel(r, level)
	for _, d := range corr.Dimensions {
		for i, dim := range dimensions {
			if d.Name == dim {
				cells[i] = CellOrEmpty{Present: true, Cell: makeCell(d)}
				break
			}
		}
	}
	return cells
}

func makeCell(d eval.Dimension) Cell {
	c := Cell{Spearman: d.Spearman, Kendall: d.KendallTau}
	if d.SpearmanCI != nil && d.KendallTauCI != nil {
		c.HasCI = true
		c.SpearmanLow = d.SpearmanCI.Low
		c.SpearmanHi = d.SpearmanCI.High
		c.KendallLow = d.KendallTauCI.Low
		c.KendallHi = d.KendallTauCI.High
	}
	return c
}

func selectLevel(r eval.Report, level string) eval.Correlation {
	if level == "system" {
		return r.SystemLevel
	}
	return r.SummaryLevel
}

func isDimensionalPartial(metric string) bool {
	for _, dm := range dimensionalMetrics {
		if strings.HasPrefix(metric, dm.prefix) {
			return true
		}
	}
	return false
}

func displayName(metric string) string {
	if v, ok := metricDisplayName[metric]; ok {
		return v
	}
	return metric
}

func sortRows(rows []Row) {
	rank := make(map[string]int, len(metricOrder))
	for i, m := range metricOrder {
		rank[metricDisplayName[m]] = i
	}
	sort.SliceStable(rows, func(i, j int) bool {
		ri, oki := rank[rows[i].Label]
		rj, okj := rank[rows[j].Label]
		switch {
		case oki && okj:
			return ri < rj
		case oki:
			return true
		case okj:
			return false
		default:
			return rows[i].Label < rows[j].Label
		}
	})
}

// ── Metadata table ─────────────────────────────────────────────────────

// buildMetadataRows builds one row per Report. Unlike the correlation
// view, dimensional metrics like geval_coherence stay separate — they're
// 4 separate runs and timing/cost matters per run.
func buildMetadataRows(reports []eval.Report) []MetadataRow {
	rows := make([]MetadataRow, 0, len(reports))
	now := time.Now().UTC()

	for _, r := range reports {
		var throughput float64
		if r.RuntimeSec > 0 {
			throughput = float64(r.Samples) / r.RuntimeSec
		}

		row := MetadataRow{
			Metric:         r.Metric,
			DisplayLabel:   metadataDisplayLabel(r.Metric),
			Norm:           r.Norm,
			Samples:        r.Samples,
			RuntimeSec:     r.RuntimeSec,
			RuntimeHuman:   formatDuration(r.RuntimeSec),
			Throughput:     throughput,
			ThroughputText: formatThroughput(throughput),
			Timestamp:      r.Timestamp,
		}

		if t, err := time.Parse(time.RFC3339, r.Timestamp); err == nil {
			row.TimestampHuman = t.UTC().Format("2006-01-02 15:04 UTC")
			row.Age = formatAge(now.Sub(t))
		} else {
			row.TimestampHuman = r.Timestamp
			row.Age = "—"
		}
		rows = append(rows, row)
	}

	sortMetadataRows(rows)
	return rows
}

// metadataDisplayLabel keeps dimensional suffixes visible (G-Eval coh)
// so timing per dimension is distinguishable.
func metadataDisplayLabel(metric string) string {
	for _, dm := range dimensionalMetrics {
		if strings.HasPrefix(metric, dm.prefix) {
			dim := strings.TrimPrefix(metric, dm.prefix)
			short, ok := dimensionShort[dim]
			if !ok {
				short = dim
			}
			return fmt.Sprintf("%s (%s)", dm.displayName, short)
		}
	}
	return displayName(metric)
}

// sortMetadataRows orders by metricOrder primary, then by dimensional
// position (coh/con/flu/rel) within each dimensional group.
func sortMetadataRows(rows []MetadataRow) {
	rank := make(map[string]int, len(metricOrder))
	for i, m := range metricOrder {
		rank[m] = i
	}
	dimRank := map[string]int{
		"coherence": 0, "consistency": 1, "fluency": 2, "relevance": 3,
	}

	keyOf := func(metric string) (int, int) {
		for _, dm := range dimensionalMetrics {
			if strings.HasPrefix(metric, dm.prefix) {
				dim := strings.TrimPrefix(metric, dm.prefix)
				prefixWithoutTrailing := strings.TrimSuffix(dm.prefix, "_")
				p, ok := rank[prefixWithoutTrailing]
				if !ok {
					p = len(metricOrder)
				}
				return p, dimRank[dim]
			}
		}
		p, ok := rank[metric]
		if !ok {
			p = len(metricOrder)
		}
		return p, 0
	}

	sort.SliceStable(rows, func(i, j int) bool {
		pi, di := keyOf(rows[i].Metric)
		pj, dj := keyOf(rows[j].Metric)
		if pi != pj {
			return pi < pj
		}
		return di < dj
	})
}

// formatDuration renders 4523.5 → "1h 15m". Matches what Reports actually
// store (seconds as float64).
func formatDuration(seconds float64) string {
	if seconds < 0 || math.IsNaN(seconds) {
		return "—"
	}
	d := time.Duration(seconds * float64(time.Second))
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%.1fs", d.Seconds())
	case d < time.Hour:
		mins := int(d.Minutes())
		secs := int(d.Seconds()) - mins*60
		return fmt.Sprintf("%dm %ds", mins, secs)
	default:
		hours := int(d.Hours())
		mins := int(d.Minutes()) - hours*60
		return fmt.Sprintf("%dh %dm", hours, mins)
	}
}

func formatThroughput(samplesPerSec float64) string {
	if samplesPerSec <= 0 || math.IsNaN(samplesPerSec) {
		return "—"
	}
	switch {
	case samplesPerSec >= 100:
		return fmt.Sprintf("%.0f/s", samplesPerSec)
	case samplesPerSec >= 1:
		return fmt.Sprintf("%.1f/s", samplesPerSec)
	default:
		// Below 1/s, show per-minute for readability.
		return fmt.Sprintf("%.1f/min", samplesPerSec*60)
	}
}

// formatAge renders "3 hours ago" / "yesterday" / "5 days ago".
func formatAge(d time.Duration) string {
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%d min ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%d hours ago", int(d.Hours()))
	case d < 48*time.Hour:
		return "yesterday"
	case d < 7*24*time.Hour:
		return fmt.Sprintf("%d days ago", int(d.Hours()/24))
	default:
		return fmt.Sprintf("%d weeks ago", int(d.Hours()/(24*7)))
	}
}

// ── Template helpers ───────────────────────────────────────────────────

func templateFuncs() template.FuncMap {
	return template.FuncMap{
		"fmtVal":  fmtVal,
		"fmtCI":   fmtCI,
		"colorOf": colorOf,
		"isOdd":   func(i int) bool { return i%2 == 1 },
	}
}

func fmtVal(x float64) string {
	if math.IsNaN(x) {
		return "—"
	}
	s := fmt.Sprintf("%.3f", x)
	switch {
	case x >= 0 && x < 1:
		return strings.TrimPrefix(s, "0")
	case x > -1 && x < 0:
		return "-" + strings.TrimPrefix(s[1:], "0")
	default:
		return s
	}
}

func fmtCI(low, high float64) string {
	return fmt.Sprintf("[%s, %s]", fmtVal(low), fmtVal(high))
}

func colorOf(x float64) string {
	switch {
	case math.IsNaN(x):
		return ""
	case x >= 0.5:
		return "bg-green-200"
	case x >= 0.3:
		return "bg-green-100"
	case x >= 0.15:
		return "bg-green-50"
	case x > -0.05:
		return ""
	case x > -0.2:
		return "bg-red-50"
	default:
		return "bg-red-100"
	}
}

// ── HTML template ──────────────────────────────────────────────────────

const pageHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{.Title}}</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
.cell-ci-wrap { position: relative; }
.cell-ci {
  display: none; position: absolute; bottom: 100%; left: 50%;
  transform: translateX(-50%); white-space: nowrap;
  padding: 4px 8px; background: #1f2937; color: white;
  font-size: 0.7rem; border-radius: 4px; z-index: 10;
  margin-bottom: 4px;
}
.cell-ci-wrap:hover .cell-ci { display: block; }
</style>
</head>
<body class="bg-gray-50 min-h-screen">
<div class="max-w-7xl mx-auto p-6">
  <header class="mb-6">
    <h1 class="text-3xl font-bold text-gray-800">{{.Title}}</h1>
    <p class="text-sm text-gray-500 mt-1">
      {{len .Metadata}} runs loaded. Edit JSON in output/ and refresh.
    </p>
  </header>

  <nav class="mb-6 flex gap-2 flex-wrap">
    <button data-tab="summary" class="tab-btn px-4 py-2 rounded font-medium bg-blue-600 text-white">
      Summary-level
    </button>
    <button data-tab="system" class="tab-btn px-4 py-2 rounded font-medium bg-gray-200 text-gray-700 hover:bg-gray-300">
      System-level
    </button>
    <button data-tab="metadata" class="tab-btn px-4 py-2 rounded font-medium bg-gray-200 text-gray-700 hover:bg-gray-300">
      Run metadata
    </button>
  </nav>

  <div data-panel="summary">
    {{template "corrTable" .Summary.Rows}}
    <p class="text-xs text-gray-400 mt-2">
      Spearman ρ and Kendall τ. Hover any cell for 95% bootstrap CI.
    </p>
  </div>
  <div data-panel="system" class="hidden">
    {{template "corrTable" .System.Rows}}
    <p class="text-xs text-gray-400 mt-2">
      Spearman ρ and Kendall τ. With N=16 systems, system-level CI are wide.
    </p>
  </div>
  <div data-panel="metadata" class="hidden">
    {{template "metadataTable" .Metadata}}
    <p class="text-xs text-gray-400 mt-2">
      One row per Report. Dimensional metrics (G-Eval, UniEval) appear as 4 separate runs.
    </p>
  </div>
</div>

<script>
(function() {
  const buttons = document.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('[data-panel]');
  const activeCls = 'tab-btn px-4 py-2 rounded font-medium bg-blue-600 text-white';
  const idleCls = 'tab-btn px-4 py-2 rounded font-medium bg-gray-200 text-gray-700 hover:bg-gray-300';

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      buttons.forEach(b => b.className = (b.dataset.tab === target ? activeCls : idleCls));
      panels.forEach(p => {
        if (p.dataset.panel === target) p.classList.remove('hidden');
        else p.classList.add('hidden');
      });
    });
  });
})();
</script>
</body>
</html>

{{define "corrTable"}}
<div class="bg-white rounded-lg shadow overflow-x-auto">
  <table class="min-w-full text-sm">
    <thead class="bg-gray-100 text-gray-700">
      <tr>
        <th rowspan="2" class="text-left px-4 py-3 font-semibold">Metric</th>
        <th colspan="2" class="text-center px-2 py-2 font-semibold border-l border-gray-200">Coh</th>
        <th colspan="2" class="text-center px-2 py-2 font-semibold border-l border-gray-200">Con</th>
        <th colspan="2" class="text-center px-2 py-2 font-semibold border-l border-gray-200">Flu</th>
        <th colspan="2" class="text-center px-2 py-2 font-semibold border-l border-gray-200">Rel</th>
      </tr>
      <tr class="text-xs text-gray-500">
        <th class="font-mono font-normal px-2 py-1 border-l border-gray-200">ρ</th>
        <th class="font-mono font-normal px-2 py-1">τ</th>
        <th class="font-mono font-normal px-2 py-1 border-l border-gray-200">ρ</th>
        <th class="font-mono font-normal px-2 py-1">τ</th>
        <th class="font-mono font-normal px-2 py-1 border-l border-gray-200">ρ</th>
        <th class="font-mono font-normal px-2 py-1">τ</th>
        <th class="font-mono font-normal px-2 py-1 border-l border-gray-200">ρ</th>
        <th class="font-mono font-normal px-2 py-1">τ</th>
      </tr>
    </thead>
    <tbody>
      {{range $i, $row := .}}
      <tr class="{{if isOdd $i}}bg-gray-50{{end}} hover:bg-blue-50">
        <td class="px-4 py-2 font-medium text-gray-800 whitespace-nowrap">{{$row.Label}}</td>
        {{range $row.Cells}}
          {{if .Present}}
          <td class="cell-ci-wrap px-2 py-2 text-right font-mono border-l border-gray-100 {{colorOf .Cell.Spearman}}">
            {{fmtVal .Cell.Spearman}}
            {{if .Cell.HasCI}}<span class="cell-ci">ρ {{fmtCI .Cell.SpearmanLow .Cell.SpearmanHi}}</span>{{end}}
          </td>
          <td class="cell-ci-wrap px-2 py-2 text-right font-mono {{colorOf .Cell.Kendall}}">
            {{fmtVal .Cell.Kendall}}
            {{if .Cell.HasCI}}<span class="cell-ci">τ {{fmtCI .Cell.KendallLow .Cell.KendallHi}}</span>{{end}}
          </td>
          {{else}}
          <td class="px-2 py-2 text-right text-gray-300 border-l border-gray-100">—</td>
          <td class="px-2 py-2 text-right text-gray-300">—</td>
          {{end}}
        {{end}}
      </tr>
      {{end}}
    </tbody>
  </table>
</div>
{{end}}

{{define "metadataTable"}}
<div class="bg-white rounded-lg shadow overflow-x-auto">
  <table class="min-w-full text-sm">
    <thead class="bg-gray-100 text-gray-700">
      <tr>
        <th class="text-left px-4 py-3 font-semibold">Metric</th>
        <th class="text-right px-3 py-3 font-semibold">Samples</th>
        <th class="text-right px-3 py-3 font-semibold">Runtime</th>
        <th class="text-right px-3 py-3 font-semibold">Throughput</th>
        <th class="text-left px-3 py-3 font-semibold">Norm</th>
        <th class="text-left px-3 py-3 font-semibold">Run timestamp</th>
        <th class="text-left px-3 py-3 font-semibold">Age</th>
      </tr>
    </thead>
    <tbody>
      {{range $i, $row := .}}
      <tr class="{{if isOdd $i}}bg-gray-50{{end}} hover:bg-blue-50">
        <td class="px-4 py-2 font-medium text-gray-800 whitespace-nowrap">{{$row.DisplayLabel}}</td>
        <td class="px-3 py-2 text-right font-mono">{{$row.Samples}}</td>
        <td class="px-3 py-2 text-right font-mono">{{$row.RuntimeHuman}}</td>
        <td class="px-3 py-2 text-right font-mono text-gray-600">{{$row.ThroughputText}}</td>
        <td class="px-3 py-2 text-gray-600">{{$row.Norm}}</td>
        <td class="px-3 py-2 font-mono text-xs text-gray-600">{{$row.TimestampHuman}}</td>
        <td class="px-3 py-2 text-gray-500">{{$row.Age}}</td>
      </tr>
      {{end}}
    </tbody>
  </table>
</div>
{{end}}
`
