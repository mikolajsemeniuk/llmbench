.PHONY: benchmark-cpu
benchmark-cpu:
	@for cmd in bleu rouge chrf meteor smartstring; do go run ./cmd/$$cmd || exit 1; done

.PHONY: benchmark-modelsrv
benchmark-modelsrv:
	go run ./cmd/bertscore
	go run ./cmd/moverscore
	go run ./cmd/bartscore
	go run ./cmd/unieval -dimension consistency

.PHONY: benchmark-ollama
benchmark-ollama:
	@for cmd in embedscorer geval smartmodel; do go run ./cmd/$$cmd || exit 1; done
	go run ./cmd/geval -dimension coherence
	go run ./cmd/geval -dimension consistency
	go run ./cmd/geval -dimension fluency
	go run ./cmd/geval -dimension relevance

.PHONY: tables-summary
tables-summary:
	go run ./cmd/tables -ci -level summary -output paper/summary.tex

.PHONY: tables-system
tables-system:
	go run ./cmd/tables -ci -level system  -output paper/system.tex
