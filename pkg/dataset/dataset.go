// Package dataset embeds the human-rated corpora into the binary so that
// every cmd/<metric> tool runs out of the box. Each corpus is <name>.jsonl
// in the SummEval layout (one article per line with its machine summaries,
// references and per-summary ratings) and is decoded by
// pkg/eval.LoadDataset:
//
//	summeval     SummEval (Fabbri et al. 2021), four dimensions
//	frank_cnndm  FRANK CNN/DM part, factuality stored as consistency
//	frank_xsum   FRANK XSum part, factuality stored as consistency
//	rose_cnndm   RoSE CNN/DM test, ACU stored as relevance
//
// The three transfer corpora are written by prepare.py (make
// transfer-data); dimensions a corpus does not annotate are 0.
package dataset

import "embed"

//go:embed *.jsonl
var FS embed.FS

// Default is the corpus every tool reads unless told otherwise.
const Default = "summeval"

// Titles are the corpora's names as printed in tables.
var Titles = map[string]string{
	"summeval":    "SummEval",
	"frank_cnndm": "FRANK (CNN/DM)",
	"frank_xsum":  "FRANK (XSum)",
	"rose_cnndm":  "RoSE (CNN/DM)",
}
