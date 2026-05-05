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
	go run ./cmd/ablation -input ablation -lambda-star $(BGS_LAMBDA) -output paper/ablation.tex -json paper/ablation.json

.PHONY: paper-comparisons
paper-comparisons:
	go run ./cmd/compare -metric bgs -baselines unieval,bertscore,geval,smartmodel,chrf -bootstrap 5000 -output paper/comparisons.tex -json paper/comparisons.json

# Canonical lead-bias λ selected on the dev split (first 50 articles).
# This is the value used for the headline run in output/bgs.json
# (consumed by paper/summary.tex). To reproduce the selection, run
# `make benchmark-ablation-dev-lead` and check the dev-mean Spearman ρ
# is maximised at λ=0.5.
BGS_LAMBDA ?= 0.5

# benchmark-bgs runs the canonical metric on the full SummEval set
# with the dev-selected λ above. Writes the report consumed by
# paper/summary.tex and paper/comparisons.tex.
.PHONY: benchmark-bgs
benchmark-bgs:
	go run ./cmd/bgs -doc-split all -lead-bias-lambda $(BGS_LAMBDA) -output output/bgs.json

# benchmark-bgs-paper is the *full reproduction pipeline* a reviewer
# can run end-to-end. It (1) regenerates the BGS lead-bias dev sweep,
# (2) re-runs the test-split verification, (3) re-runs the canonical
# metric on the full set, (4) re-renders every paper artifact
# (summary, system, ablation, comparisons). Allow ~10 minutes total
# on Apple silicon with Ollama warmed up.
.PHONY: benchmark-bgs-paper
benchmark-bgs-paper: benchmark-ablation benchmark-bgs paper

# benchmark-ablation regenerates the per-variant snapshots in
# ablation/ that back paper/ablation.tex. Two conceptually distinct
# components: the recall-only baseline (λ=0) and the lead-bias sweep
# (λ>0). Each runs on both dev and test splits.
.PHONY: benchmark-ablation
benchmark-ablation: benchmark-ablation-recall benchmark-ablation-lead

# Recall baseline — the metric with NO lead-bias prior (λ=0). Two
# runs (dev + test) producing bgs_recall_{dev,test}.json. Establishes
# the reference point against which the lead-bias sweep is compared.
.PHONY: benchmark-ablation-recall
benchmark-ablation-recall:
	@mkdir -p ablation
	go run ./cmd/bgs -doc-split first50 -lead-bias-lambda 0 -bootstrap 0 -output ablation/bgs_recall_dev.json
	go run ./cmd/bgs -doc-split last50  -lead-bias-lambda 0 -bootstrap 0 -output ablation/bgs_recall_test.json

# Lead-bias sweep — λ ∈ {0.25, 0.5, 1.0, 2.0} on both dev and test.
# Selected λ* is the value that maximises dev-mean Spearman ρ; the
# matching test-split row carries ★ in the rendered table.
.PHONY: benchmark-ablation-lead
benchmark-ablation-lead:
	@mkdir -p ablation
	go run ./cmd/bgs -doc-split first50 -lead-bias-lambda 0.25 -bootstrap 0 -output ablation/bgs_lead_dev_l025.json
	go run ./cmd/bgs -doc-split first50 -lead-bias-lambda 0.5  -bootstrap 0 -output ablation/bgs_lead_dev_l050.json
	go run ./cmd/bgs -doc-split first50 -lead-bias-lambda 1.0  -bootstrap 0 -output ablation/bgs_lead_dev_l100.json
	go run ./cmd/bgs -doc-split first50 -lead-bias-lambda 2.0  -bootstrap 0 -output ablation/bgs_lead_dev_l200.json
	go run ./cmd/bgs -doc-split last50  -lead-bias-lambda 0.25 -bootstrap 0 -output ablation/bgs_lead_test_l025.json
	go run ./cmd/bgs -doc-split last50  -lead-bias-lambda 0.5  -bootstrap 0 -output ablation/bgs_lead_test_l050.json
	go run ./cmd/bgs -doc-split last50  -lead-bias-lambda 1.0  -bootstrap 0 -output ablation/bgs_lead_test_l100.json
