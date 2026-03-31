package llmbench

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
)

func TestParseReportFileJSON_compare(t *testing.T) {
	t.Parallel()

	raw := []byte(`{
		"model_a": "ollama/a",
		"model_b": "ollama/b",
		"generated_at": "2020-01-01T00:00:00Z",
		"aggregate": [],
		"per_level": [],
		"per_task": [],
		"raw": {"a": {}, "b": {}}
	}`)

	v, err := ParseReportFileJSON(raw)
	if err != nil {
		t.Fatalf("ParseReportFileJSON: %v", err)
	}

	if !v.IsCompare {
		t.Fatal("want IsCompare true")
	}

	if v.Compare.ModelA != "ollama/a" || v.Compare.ModelB != "ollama/b" {
		t.Errorf("got ModelA=%q ModelB=%q", v.Compare.ModelA, v.Compare.ModelB)
	}
}

func TestParseReportFileJSON_single(t *testing.T) {
	t.Parallel()

	raw := []byte(`{
		"metadata": {"provider": "ollama", "model": "m"},
		"aggregate": {"esr": 0.5},
		"per_level": [],
		"rag_quality": {},
		"per_task": [],
		"runs": []
	}`)

	v, err := ParseReportFileJSON(raw)
	if err != nil {
		t.Fatalf("ParseReportFileJSON: %v", err)
	}

	if v.IsCompare {
		t.Fatal("want IsCompare false")
	}

	if v.Single.Metadata.Provider != "ollama" || v.Single.Metadata.Model != "m" {
		t.Errorf("metadata: %+v", v.Single.Metadata)
	}
}

func TestParseReportFileJSON_invalidJSON(t *testing.T) {
	t.Parallel()

	_, err := ParseReportFileJSON([]byte(`{`))
	if err == nil {
		t.Fatal("want error for invalid JSON")
	}
}

func TestFormatHTMLDelta(t *testing.T) {
	t.Parallel()

	t.Run("zero", func(t *testing.T) {
		t.Parallel()

		if got := FormatHTMLDelta(0); got != "0.0000" {
			t.Errorf("got %q", got)
		}
	})

	t.Run("positive", func(t *testing.T) {
		t.Parallel()

		if got := FormatHTMLDelta(0.0842); got != "+0.0842" {
			t.Errorf("got %q", got)
		}
	})

	t.Run("negative", func(t *testing.T) {
		t.Parallel()

		if got := FormatHTMLDelta(-0.0312); got != "−0.0312" {
			t.Errorf("got %q", got)
		}
	})
}

func TestSanitizeMetricsForJSON(t *testing.T) {
	t.Parallel()

	m := Metrics{
		CES:  math.Inf(1),
		LAE:  math.NaN(),
		MTTR: math.Inf(-1),
	}
	SanitizeMetricsForJSON(&m)
	if m.CES != -1 || m.LAE != -1 || m.MTTR != -1 {
		t.Errorf("got CES=%g LAE=%g MTTR=%g, want -1 for all", m.CES, m.LAE, m.MTTR)
	}
}

func TestCountTasksByLevel(t *testing.T) {
	t.Parallel()

	tasks := []Task{
		{ID: "1", Level: LevelDiagnostic},
		{ID: "2", Level: LevelDiagnostic},
		{ID: "3", Level: LevelRepair},
	}

	t.Run("L1_count", func(t *testing.T) {
		t.Parallel()

		if n := CountTasksByLevel(tasks, LevelDiagnostic); n != 2 {
			t.Errorf("got %d, want 2", n)
		}
	})

	t.Run("L3_empty", func(t *testing.T) {
		t.Parallel()

		if n := CountTasksByLevel(tasks, LevelMultiStep); n != 0 {
			t.Errorf("got %d, want 0", n)
		}
	})
}

func TestPrintReportSummary_containsProvider(t *testing.T) {
	t.Parallel()

	r := Report{
		Metadata: Metadata{Provider: "ollama", Model: "qwen", TotalTasks: 1, RunsPerTask: 1, TotalRuns: 1},
		Metrics:  Metrics{ESR: 0.5, ESRCI: [2]float64{0.4, 0.6}, TSA: 0.5, CHR: 0.1},
	}
	var buf bytes.Buffer
	PrintReportSummary(&buf, r, "/tmp/out.json")
	s := buf.String()
	if !strings.Contains(s, "ollama") || !strings.Contains(s, "qwen") || !strings.Contains(s, "/tmp/out.json") {
		t.Fatalf("unexpected output:\n%s", s)
	}
}

func TestWriteReportJSON_roundTrip(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := dir + "/r.json"
	r := Report{
		Metadata: Metadata{Provider: "p", Model: "m", TotalTasks: 0, RunsPerTask: 0, TotalRuns: 0},
		Metrics:  Metrics{},
	}
	if err := WriteReportJSON(path, r); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	var back Report
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if back.Metadata.Provider != "p" {
		t.Errorf("Provider = %q", back.Metadata.Provider)
	}
}
