package llmbench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

type Dataset struct {
	ID               string    `json:"id"`
	Text             string    `json:"text"`
	MachineSummaries []string  `json:"machine_summaries"`
	HumanSummaries   []string  `json:"human_summaries"`
	Relevance        []float64 `json:"relevance"`
	Coherence        []float64 `json:"coherence"`
	Fluency          []float64 `json:"fluency"`
	Consistency      []float64 `json:"consistency"`
}

func NewDataset(filePath string) ([]Dataset, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var results []Dataset
	scanner := bufio.NewScanner(file)

	// SummEval ma bardzo długie teksty (artykuły CNN),
	// więc ustawiamy duży bufor (np. 2MB), aby uniknąć błędu "token too long"
	const maxCapacity = 2 * 1024 * 1024
	buf := make([]byte, maxCapacity)
	scanner.Buffer(buf, maxCapacity)

	for scanner.Scan() {
		var record Dataset
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			return nil, fmt.Errorf("error unmarshaling line: %v", err)
		}

		results = append(results, record)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return results, nil
}
