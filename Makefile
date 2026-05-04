.PHONY: benchmark
benchmark: benchmark-lexical benchmark-modelsrv benchmark-ollama

.PHONY: benchmark-lexical
benchmark-lexical:
	go run ./cmd/bleu
	go run ./cmd/rouge
	go run ./cmd/chrf
	go run ./cmd/meteor
	go run ./cmd/smartstring

.PHONY: benchmark-modelsrv
benchmark-modelsrv:
	go run ./cmd/bertscorer
	go run ./cmd/moverscorer
	go run ./cmd/bartscorer
	go run ./cmd/gptscorer
	go run ./cmd/unieval -dimension coherence
	go run ./cmd/unieval -dimension consistency
	go run ./cmd/unieval -dimension fluency
	go run ./cmd/unieval -dimension relevance

.PHONY: benchmark-ollama
benchmark-ollama:
	go run ./cmd/embedscorer
	go run ./cmd/smartmodel
	go run ./cmd/bgs
	go run ./cmd/geval -dimension coherence
	go run ./cmd/geval -dimension consistency
	go run ./cmd/geval -dimension fluency
	go run ./cmd/geval -dimension relevance

.PHONY: paper
paper: paper-summary paper-system

.PHONY: paper-summary
paper-summary:
	go run ./cmd/paper -ci -level summary -output paper/summary.tex

.PHONY: paper-system
paper-system:
	go run ./cmd/paper -ci -level system  -output paper/system.tex
