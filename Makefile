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
paper: paper-summary paper-system paper-ablation paper-comparisons

.PHONY: paper-summary
paper-summary:
	go run ./cmd/paper -ci -level summary -output paper/summary.tex

.PHONY: paper-system
paper-system:
	go run ./cmd/paper -ci -level system  -output paper/system.tex

.PHONY: paper-ablation
paper-ablation:
	go run ./cmd/ablation -input ablation -output paper/ablation.tex -json paper/ablation.json

.PHONY: paper-comparisons
paper-comparisons:
	go run ./cmd/compare -metric bgs -baselines unieval,bertscore,geval,smartmodel,chrf -bootstrap 5000 -output paper/comparisons.tex -json paper/comparisons.json

.PHONY: benchmark-bgs
benchmark-bgs:
	go run ./cmd/bgs

# benchmark-ablation regenerates the per-variant snapshots in ablation/
# that back paper/ablation.tex. One file per cell of the β × salience-frac
# grid plus a recall-only baseline (precision side disabled). All runs
# use -bootstrap 0 — ablation point estimates only, no per-cell CI.
.PHONY: benchmark-ablation
benchmark-ablation:
	@mkdir -p ablation
	go run ./cmd/bgs -beta 1 -salience-frac 0.30 -bootstrap 0 -output ablation/bgs_beta1.json
	go run ./cmd/bgs -beta 2 -salience-frac 0.30 -bootstrap 0 -output ablation/bgs_beta2.json
	go run ./cmd/bgs -beta 3 -salience-frac 0.30 -bootstrap 0 -output ablation/bgs_beta3.json
	go run ./cmd/bgs -beta 1 -salience-frac 0.10 -bootstrap 0 -output ablation/bgs_salience10.json
	go run ./cmd/bgs -beta 1 -salience-frac 0.50 -bootstrap 0 -output ablation/bgs_salience50.json
	go run ./cmd/bgs -beta 1 -salience-frac 1.00 -bootstrap 0 -output ablation/bgs_salience100.json
	go run ./cmd/bgs -recall-only -bootstrap 0 -output ablation/bgs_recall_only.json
