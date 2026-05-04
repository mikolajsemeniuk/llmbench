package dataset

import "embed"

//go:embed summeval.jsonl
var Summeval embed.FS

const SummevalDefaultPath = "summeval.jsonl"
