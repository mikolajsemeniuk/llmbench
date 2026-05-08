// Package dataset embeds the SummEval release JSONL into the binary
// so that every cmd/<metric> tool runs out-of-the-box without
// requiring an external data download. The default consumer is
// pkg/eval.NewDataset, which decodes the embedded file via the
// io/fs interface; callers that want to point at a different JSONL
// (a different snapshot, a subsetted file, a custom corpus) can pass
// their own fs.FS to NewDataset and bypass this embedded copy
// entirely. SummevalDefaultPath is the canonical filename within
// the embed.FS.
package dataset

import "embed"

//go:embed summeval.jsonl
var Summeval embed.FS

const SummevalDefaultPath = "summeval.jsonl"
