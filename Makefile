.PHONY: benchmark
benchmark:
	@for cmd in bleu rouge chrf meteor smartstring; do go run ./cmd/$$cmd || exit 1; done

.PHONY: tables-summary
tables-summary:
	go run ./cmd/tables -ci -level summary -output paper/summary.tex

.PHONY: tables-system
tables-system:
	go run ./cmd/tables -ci -level system  -output paper/system.tex
