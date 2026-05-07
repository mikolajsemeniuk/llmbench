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
	go run ./cmd/lgs
	go run ./cmd/geval -dimension coherence
	go run ./cmd/geval -dimension consistency
	go run ./cmd/geval -dimension fluency
	go run ./cmd/geval -dimension relevance

.PHONY: paper
paper: paper-summary paper-system paper-ablation paper-embedder-ablation paper-comparisons

.PHONY: paper-summary
paper-summary:
	go run ./cmd/paper -ci -level summary -output paper/summary.tex

.PHONY: paper-system
paper-system:
	go run ./cmd/paper -ci -level system  -output paper/system.tex

.PHONY: paper-ablation
paper-ablation:
	go run ./cmd/ablation -input ablation -lambda-star $(LGS_LAMBDA) -output paper/ablation.tex

.PHONY: paper-embedder-ablation
paper-embedder-ablation:
	go run ./cmd/embedder -input ablation -canonical $(LGS_EMBED_MODEL) -output paper/embedder_ablation.tex

.PHONY: paper-comparisons
paper-comparisons:
	go run ./cmd/compare -metric lgs -baselines unieval,bertscore,geval,smartmodel,chrf,embedscorer -bootstrap 5000 -output paper/comparisons.tex

# Canonical hyperparameters. λ is the lead-bias decay (selected on
# the dev split — see benchmark-ablation-lead). LGS_EMBED_MODEL is
# the canonical sentence-embedder used for the headline run; the
# embedder ablation (benchmark-embedder-ablation) compares it against
# alternatives to show the result is not specific to one backbone.
LGS_LAMBDA      ?= 0.5
LGS_EMBED_MODEL ?= nomic-embed-text

# benchmark-lgs runs the canonical metric on the full SummEval set
# with the dev-selected λ and the canonical embedder. Writes the
# report consumed by paper/summary.tex and paper/comparisons.tex.
.PHONY: benchmark-lgs
benchmark-lgs:
	go run ./cmd/lgs -doc-split all -lead-bias-lambda $(LGS_LAMBDA) -embed-model $(LGS_EMBED_MODEL) -output output/lgs.json

# benchmark-lgs-paper is the *full reproduction pipeline* a reviewer
# can run end-to-end. It (1) regenerates the LGS lead-bias dev sweep
# and test verification, (2) regenerates the embedder ablation,
# (3) re-runs the canonical metric on the full set, (4) re-renders
# every paper artifact (summary, system, ablation, embedder ablation,
# comparisons). Allow ~25 minutes total on Apple silicon with Ollama
# warmed up.
.PHONY: benchmark-lgs-paper
benchmark-lgs-paper: benchmark-ablation benchmark-embedder-ablation benchmark-lgs paper

# benchmark-ablation regenerates the per-variant snapshots in
# ablation/ that back paper/ablation.tex. Two conceptually distinct
# components: the recall-only baseline (λ=0) and the lead-bias sweep
# (λ>0). Each runs on both dev and test splits.
.PHONY: benchmark-ablation
benchmark-ablation: benchmark-ablation-recall benchmark-ablation-lead

# Recall baseline — the metric with NO lead-bias prior (λ=0). Two
# runs (dev + test) producing lgs_recall_{dev,test}.json. Establishes
# the reference point against which the lead-bias sweep is compared.
.PHONY: benchmark-ablation-recall
benchmark-ablation-recall:
	@mkdir -p ablation
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0 -bootstrap 0 -output ablation/lgs_recall_dev.json
	go run ./cmd/lgs -doc-split last50  -lead-bias-lambda 0 -bootstrap 0 -output ablation/lgs_recall_test.json

# Lead-bias sweep — λ ∈ {0.25, 0.5, 1.0, 2.0} on both dev and test.
# Selected λ* is the value that maximises dev-mean Spearman ρ; the
# matching test-split row carries ★ in the rendered table.
.PHONY: benchmark-ablation-lead
benchmark-ablation-lead:
	@mkdir -p ablation
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0.25 -bootstrap 0 -output ablation/lgs_lead_dev_l025.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0.5  -bootstrap 0 -output ablation/lgs_lead_dev_l050.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 1.0  -bootstrap 0 -output ablation/lgs_lead_dev_l100.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 2.0  -bootstrap 0 -output ablation/lgs_lead_dev_l200.json
	go run ./cmd/lgs -doc-split last50  -lead-bias-lambda 0.25 -bootstrap 0 -output ablation/lgs_lead_test_l025.json
	go run ./cmd/lgs -doc-split last50  -lead-bias-lambda 0.5  -bootstrap 0 -output ablation/lgs_lead_test_l050.json
	go run ./cmd/lgs -doc-split last50  -lead-bias-lambda 1.0  -bootstrap 0 -output ablation/lgs_lead_test_l100.json
	go run ./cmd/lgs -doc-split last50  -lead-bias-lambda 2.0  -bootstrap 0 -output ablation/lgs_lead_test_l200.json

# Finer lead-bias sweep around the dev optimum — preempts the
# reviewer concern that λ*=0.5 is a noise pick. The coarse dev sweep
# (benchmark-ablation-lead) puts λ=0.25 at .251 and λ=0.5 at .252
# (gap of .001, no CI). This finer sweep covers
# λ ∈ {0.3, 0.4, 0.6, 0.75} (the missing points around the optimum)
# WITH bootstrap CI=5000 on each per-dimension ρ. λ=0.5 is also
# re-run with bootstrap so all five points around the optimum carry
# CIs comparable to each other. Output files use `_b5k` suffix to
# distinguish them from the no-bootstrap coarse sweep snapshots.
.PHONY: benchmark-ablation-lead-finer
benchmark-ablation-lead-finer:
	@mkdir -p ablation
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0.30 -bootstrap 5000 -output ablation/lgs_lead_dev_l030_b5k.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0.40 -bootstrap 5000 -output ablation/lgs_lead_dev_l040_b5k.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0.50 -bootstrap 5000 -output ablation/lgs_lead_dev_l050_b5k.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0.60 -bootstrap 5000 -output ablation/lgs_lead_dev_l060_b5k.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0.75 -bootstrap 5000 -output ablation/lgs_lead_dev_l075_b5k.json

# Test-side counterpart of the finer sweep. Same five λ values as
# benchmark-ablation-lead-finer, but on last50 (test split) with
# bootstrap CI=5000. Used to verify that the smooth-and-flat dev
# surface around the optimum transfers to the held-out test split.
.PHONY: benchmark-ablation-lead-finer-test
benchmark-ablation-lead-finer-test:
	@mkdir -p ablation
	go run ./cmd/lgs -doc-split last50 -lead-bias-lambda 0.30 -bootstrap 5000 -output ablation/lgs_lead_test_l030_b5k.json
	go run ./cmd/lgs -doc-split last50 -lead-bias-lambda 0.40 -bootstrap 5000 -output ablation/lgs_lead_test_l040_b5k.json
	go run ./cmd/lgs -doc-split last50 -lead-bias-lambda 0.50 -bootstrap 5000 -output ablation/lgs_lead_test_l050_b5k.json
	go run ./cmd/lgs -doc-split last50 -lead-bias-lambda 0.60 -bootstrap 5000 -output ablation/lgs_lead_test_l060_b5k.json
	go run ./cmd/lgs -doc-split last50 -lead-bias-lambda 0.75 -bootstrap 5000 -output ablation/lgs_lead_test_l075_b5k.json

# Cross-embedder lead-bias verification — repeats the dev λ sweep
# with a second embedder (bge-m3, 567M, the largest in our roster)
# to confirm the lead-bias contribution is not specific to
# nomic-embed-text. Outputs files with `_bge` suffix so they can be
# distinguished from the canonical nomic sweep. Used in narrative,
# not currently rendered into a paper table.
.PHONY: benchmark-ablation-lead-bge
benchmark-ablation-lead-bge:
	@mkdir -p ablation
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0    -embed-model bge-m3 -bootstrap 0 -output ablation/lgs_lead_dev_l000_bge.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0.25 -embed-model bge-m3 -bootstrap 0 -output ablation/lgs_lead_dev_l025_bge.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 0.5  -embed-model bge-m3 -bootstrap 0 -output ablation/lgs_lead_dev_l050_bge.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 1.0  -embed-model bge-m3 -bootstrap 0 -output ablation/lgs_lead_dev_l100_bge.json
	go run ./cmd/lgs -doc-split first50 -lead-bias-lambda 2.0  -embed-model bge-m3 -bootstrap 0 -output ablation/lgs_lead_dev_l200_bge.json

# Embedder ablation — runs the canonical metric (λ=$(LGS_LAMBDA)) on
# the full SummEval set with four sentence-embedder backbones.
# Establishes that LGS is a metric design (sentence-level grounding +
# lead-bias prior), not specific to nomic-embed-text. Output is
# rendered into paper/embedder_ablation.tex by `paper-embedder-ablation`.
# Requires the four embedders to be available in Ollama:
#   ollama pull nomic-embed-text mxbai-embed-large bge-m3 all-minilm
.PHONY: benchmark-embedder-ablation
benchmark-embedder-ablation:
	@mkdir -p ablation
	go run ./cmd/lgs -doc-split all -lead-bias-lambda $(LGS_LAMBDA) -embed-model nomic-embed-text  -bootstrap 0 -output ablation/lgs_embedder_nomic.json
	go run ./cmd/lgs -doc-split all -lead-bias-lambda $(LGS_LAMBDA) -embed-model mxbai-embed-large -bootstrap 0 -output ablation/lgs_embedder_mxbai.json
	go run ./cmd/lgs -doc-split all -lead-bias-lambda $(LGS_LAMBDA) -embed-model bge-m3            -bootstrap 0 -output ablation/lgs_embedder_bge.json
	go run ./cmd/lgs -doc-split all -lead-bias-lambda $(LGS_LAMBDA) -embed-model all-minilm        -bootstrap 0 -output ablation/lgs_embedder_minilm.json
