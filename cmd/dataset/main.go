package main

import (
	"fmt"

	"github.com/mikolajsemeniuk/llmbench"
)

func main() {
	path := "../../model_annotations.aligned.scored.jsonl"
	records, err := llmbench.NewDataset(path)
	if err != nil {
		fmt.Printf("Błąd: %v\n", err)
		return
	}

	if len(records) > 0 {
		r := records[0]
		fmt.Printf("Wczytano rekordów: %d\n", len(records))
		fmt.Printf("ID: %s\n", r.ID)
		fmt.Printf("Liczba podsumowań maszynowych: %d\n", len(r.MachineSummaries))
		fmt.Printf("Pierwsza ocena spójności (Coherence): %f\n", r.Coherence[0])
	}
}
