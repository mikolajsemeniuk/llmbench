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
	go run ./cmd/ablation -input ablation \
		-k-star $(BGS_K) \
		-lambda-star $(BGS_LAMBDA) \
		-alpha-star $(BGS_ALPHA) \
		-gamma-star $(BGS_GAMMA) \
		-output paper/ablation.tex -json paper/ablation.json

.PHONY: paper-comparisons
paper-comparisons:
	go run ./cmd/compare -metric bgs -baselines unieval,bertscore,geval,smartmodel,chrf -bootstrap 5000 -output paper/comparisons.tex -json paper/comparisons.json

# Canonical hyperparameters selected on the dev split (first 50 articles).
# These are the values used for the headline run in output/bgs.json
# (consumed by paper/summary.tex). To reproduce the selection, run
# `make benchmark-ablation-dev-*` and check the dev-mean Spearman ρ
# is maximised at the values below. Of the four tested components,
# only lead-bias (λ) improved dev-mean over the recall baseline.
BGS_K       ?= 1
BGS_LAMBDA  ?= 0.5
BGS_ALPHA   ?= 0
BGS_GAMMA   ?= 0

# benchmark-bgs runs the canonical metric on the full SummEval set
# with the dev-selected hyperparameters above. Writes the report
# consumed by paper/summary.tex and paper/comparisons.tex.
.PHONY: benchmark-bgs
benchmark-bgs:
	go run ./cmd/bgs -doc-split all \
		-recall-top-k $(BGS_K) \
		-lead-bias-lambda $(BGS_LAMBDA) \
		-coverage-alpha $(BGS_ALPHA) \
		-redundancy-gamma $(BGS_GAMMA) \
		-output output/bgs.json

# benchmark-bgs-paper is the *full reproduction pipeline* a reviewer
# can run end-to-end. It (1) regenerates every BGS-related ablation
# snapshot, (2) re-runs the canonical metric, (3) re-renders every
# paper artifact (summary, system, ablation, comparisons). Allow
# ~30 minutes total on Apple silicon with Ollama warmed up.
.PHONY: benchmark-bgs-paper
benchmark-bgs-paper: benchmark-ablation benchmark-bgs paper

# benchmark-ablation regenerates the per-variant snapshots in
# ablation/ that back paper/ablation.tex. Four orthogonal dev sweeps
# (lead-bias λ, top-k k, coverage α, diversity γ) plus test-split
# verifications plus legacy F_β reproductions. All runs use
# -bootstrap 0 (ablation point estimates).
.PHONY: benchmark-ablation
benchmark-ablation: benchmark-ablation-dev-lead benchmark-ablation-dev-topk benchmark-ablation-dev-coverage benchmark-ablation-dev-diversity benchmark-ablation-test benchmark-ablation-legacy

# Dev sweep — lead bias λ. The accepted (canonical) component:
# multiplies each cosine cos(c_j, s_i) by exp(−λ · i / n) before the
# argmax. λ=0 is the position-agnostic recall baseline.
.PHONY: benchmark-ablation-dev-lead
benchmark-ablation-dev-lead:
	@mkdir -p ablation
	go run ./cmd/bgs -doc-split first50 -lead-bias-lambda 0.25 -bootstrap 0 -output ablation/bgs_lead_dev_l025.json
	go run ./cmd/bgs -doc-split first50 -lead-bias-lambda 0.5  -bootstrap 0 -output ablation/bgs_lead_dev_l050.json
	go run ./cmd/bgs -doc-split first50 -lead-bias-lambda 1.0  -bootstrap 0 -output ablation/bgs_lead_dev_l100.json
	go run ./cmd/bgs -doc-split first50 -lead-bias-lambda 2.0  -bootstrap 0 -output ablation/bgs_lead_dev_l200.json

# Dev sweep — top-k recall (rejected on dev, kept as ablation row).
# Replaces max with mean of top-k cosines. k=1 reduces to mean-of-max.
.PHONY: benchmark-ablation-dev-topk
benchmark-ablation-dev-topk:
	@mkdir -p ablation
	go run ./cmd/bgs -doc-split first50 -recall-top-k 2  -bootstrap 0 -output ablation/bgs_topk_dev_k02.json
	go run ./cmd/bgs -doc-split first50 -recall-top-k 3  -bootstrap 0 -output ablation/bgs_topk_dev_k03.json
	go run ./cmd/bgs -doc-split first50 -recall-top-k 5  -bootstrap 0 -output ablation/bgs_topk_dev_k05.json
	go run ./cmd/bgs -doc-split first50 -recall-top-k 10 -bootstrap 0 -output ablation/bgs_topk_dev_k10.json

# Dev sweep #1: coverage exponent α, with diversity disabled (γ=0).
# Sweep grid: {0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0}. The α=0 row is the
# recall-only baseline.
.PHONY: benchmark-ablation-dev-coverage
benchmark-ablation-dev-coverage:
	@mkdir -p ablation
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 0    -redundancy-gamma 0 -bootstrap 0 -output ablation/bgs_cov_dev_a000.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 0.25 -redundancy-gamma 0 -bootstrap 0 -output ablation/bgs_cov_dev_a025.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 0.5  -redundancy-gamma 0 -bootstrap 0 -output ablation/bgs_cov_dev_a050.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 1.0  -redundancy-gamma 0 -bootstrap 0 -output ablation/bgs_cov_dev_a100.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 1.5  -redundancy-gamma 0 -bootstrap 0 -output ablation/bgs_cov_dev_a150.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 2.0  -redundancy-gamma 0 -bootstrap 0 -output ablation/bgs_cov_dev_a200.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 3.0  -redundancy-gamma 0 -bootstrap 0 -output ablation/bgs_cov_dev_a300.json

# Dev sweep #2: diversity exponent γ (within-summary redundancy
# penalty), with coverage disabled (α=0). Sweep grid: {0.1, 0.25,
# 0.5, 0.75, 1.0, 1.5, 2.0}. γ=0 is the same recall baseline as the
# coverage sweep's a000 row, so it's not duplicated here.
.PHONY: benchmark-ablation-dev-diversity
benchmark-ablation-dev-diversity:
	@mkdir -p ablation
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 0 -redundancy-gamma 0.10 -bootstrap 0 -output ablation/bgs_div_dev_g010.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 0 -redundancy-gamma 0.25 -bootstrap 0 -output ablation/bgs_div_dev_g025.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 0 -redundancy-gamma 0.50 -bootstrap 0 -output ablation/bgs_div_dev_g050.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 0 -redundancy-gamma 0.75 -bootstrap 0 -output ablation/bgs_div_dev_g075.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 0 -redundancy-gamma 1.00 -bootstrap 0 -output ablation/bgs_div_dev_g100.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 0 -redundancy-gamma 1.50 -bootstrap 0 -output ablation/bgs_div_dev_g150.json
	go run ./cmd/bgs -doc-split first50 -coverage-alpha 0 -redundancy-gamma 2.00 -bootstrap 0 -output ablation/bgs_div_dev_g200.json

# Test-split verification rows for the ablation table. The recall
# baseline (λ=0, k=1, α=γ=0) and the canonical lead-bias λ*=0.5 are
# the headline rows. Adjacent λ and select rejected-component rows
# are included as illustration of how the test split responds.
.PHONY: benchmark-ablation-test
benchmark-ablation-test:
	@mkdir -p ablation
	go run ./cmd/bgs -doc-split last50 -lead-bias-lambda 0    -bootstrap 0 -output ablation/bgs_cov_test_a000.json
	go run ./cmd/bgs -doc-split last50 -lead-bias-lambda 0.25 -bootstrap 0 -output ablation/bgs_lead_test_l025.json
	go run ./cmd/bgs -doc-split last50 -lead-bias-lambda 0.5  -bootstrap 0 -output ablation/bgs_lead_test_l050.json
	go run ./cmd/bgs -doc-split last50 -lead-bias-lambda 1.0  -bootstrap 0 -output ablation/bgs_lead_test_l100.json
	go run ./cmd/bgs -doc-split last50 -coverage-alpha 0.5    -bootstrap 0 -output ablation/bgs_cov_test_a050.json
	go run ./cmd/bgs -doc-split last50 -coverage-alpha 1.0    -bootstrap 0 -output ablation/bgs_cov_test_a100.json
	go run ./cmd/bgs -doc-split last50 -redundancy-gamma 0.10 -bootstrap 0 -output ablation/bgs_div_test_g010.json
	go run ./cmd/bgs -doc-split last50 -redundancy-gamma 0.25 -bootstrap 0 -output ablation/bgs_div_test_g025.json

# Legacy precision-side reproductions on both splits, for the ablation
# table comparison row.
.PHONY: benchmark-ablation-legacy
benchmark-ablation-legacy:
	@mkdir -p ablation
	go run ./cmd/bgs -doc-split first50 -legacy-precision -beta 1 -salience-frac 0.30 -bootstrap 0 -output ablation/bgs_legacy_dev_b1.json
	go run ./cmd/bgs -doc-split first50 -legacy-precision -beta 2 -salience-frac 0.30 -bootstrap 0 -output ablation/bgs_legacy_dev_b2.json
	go run ./cmd/bgs -doc-split last50  -legacy-precision -beta 1 -salience-frac 0.30 -bootstrap 0 -output ablation/bgs_legacy_test_b1.json
	go run ./cmd/bgs -doc-split last50  -legacy-precision -beta 2 -salience-frac 0.30 -bootstrap 0 -output ablation/bgs_legacy_test_b2.json
