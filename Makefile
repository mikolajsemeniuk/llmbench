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
	$(MAKE) benchmark-geval

# G-Eval is stochastic at temperature>0. We average $(GEVAL_RUNS)
# independent runs at $(GEVAL_TEMPERATURE) to bring the methodology
# closer to the original paper (which averaged 20 GPT-4 samples) and
# to obtain across-run mean±std for the LLM-judge variance reported
# alongside the canonical correlation. With runs=1 the configuration
# collapses to the legacy greedy-decoding single-run baseline.
GEVAL_RUNS ?= 3
GEVAL_TEMPERATURE ?= 0.7
GEVAL_BASE_SEED ?= 42
.PHONY: benchmark-geval
benchmark-geval:
	go run ./cmd/geval -dimension coherence   -runs $(GEVAL_RUNS) -temperature $(GEVAL_TEMPERATURE) -base-seed $(GEVAL_BASE_SEED)
	go run ./cmd/geval -dimension consistency -runs $(GEVAL_RUNS) -temperature $(GEVAL_TEMPERATURE) -base-seed $(GEVAL_BASE_SEED)
	go run ./cmd/geval -dimension fluency     -runs $(GEVAL_RUNS) -temperature $(GEVAL_TEMPERATURE) -base-seed $(GEVAL_BASE_SEED)
	go run ./cmd/geval -dimension relevance   -runs $(GEVAL_RUNS) -temperature $(GEVAL_TEMPERATURE) -base-seed $(GEVAL_BASE_SEED)

.PHONY: paper
paper: paper-summary paper-system paper-ablation paper-embedders paper-comparisons paper-frontier paper-lambdaci paper-friedman paper-splitrobust paper-lambdadim

# The correlation tables carry a bootstrap CI in every cell. Emitting
# Spearman and Kendall side by side makes the table roughly twice as wide
# as the elsarticle text block, so each coefficient gets its own table:
# Spearman in the results section, Kendall in the appendix.
.PHONY: paper-summary
paper-summary:
	go run ./cmd/paper -ci -level summary -coeffs spearman -output paper/summary.gen.tex
	go run ./cmd/paper -ci -level summary -coeffs kendall \
		-label tab:correlations_summary_kendall -output paper/summary_kendall.gen.tex

.PHONY: paper-system
paper-system:
	go run ./cmd/paper -ci -level system  -coeffs spearman -output paper/system.gen.tex
	go run ./cmd/paper -ci -level system  -coeffs kendall \
		-label tab:correlations_system_kendall -output paper/system_kendall.gen.tex

.PHONY: paper-ablation
paper-ablation:
	go run ./cmd/ablation -input ablation -lambda-star $(LGS_LAMBDA) -output paper/ablation.gen.tex

.PHONY: paper-embedders
paper-embedders:
	go run ./cmd/embedder -input ablation -canonical $(LGS_EMBED_MODEL) -output paper/embedders.gen.tex

.PHONY: paper-comparisons
paper-comparisons:
	go run ./cmd/compare -metric lgs -baselines unieval,bertscore,geval,smartmodel,chrf,embedscorer -bootstrap 5000 -output paper/comparisons.gen.tex

.PHONY: paper-frontier
paper-frontier:
	go run ./cmd/frontier -input output -output paper/frontier.gen.tex

# paper-lambdaci quantifies how much of the reported lambda* is signal
# and how much is the luck of drawing 50 development articles. It
# re-runs the whole selection procedure on each cluster-bootstrap
# resample and reports P(argmax) per grid value, plus paired
# cross-backbone and same-backbone controls. Reads the per-sample
# scores already in ablation/ -- no embedder passes, so it is cheap to
# re-run after any change to the sweep.
.PHONY: paper-lambdaci
paper-lambdaci:
	go run ./cmd/lambdaci -input ablation -bootstrap 5000 \
		-backbone-a $(LGS_EMBED_MODEL) -backbone-b bge-m3 \
		-output paper/lambdaci.gen.tex

# paper-confound reports how much of each metric's correlation with
# human ratings is explained by plain extractiveness -- the fraction of
# candidate bigrams copied verbatim out of the source. Motivated by the
# lead-window controls below: they rival far more expensive metrics on
# raw correlation, and this target is what separates "the cheap
# baseline is as good" from "the benchmark rewards copying".
.PHONY: paper-confound
paper-confound:
	go run ./cmd/confound -input output -bootstrap 2000 \
		-ours lgs -output paper/confound.gen.tex

# Omnibus + post-hoc significance testing across the whole metric
# pool (reviewers #1 #5 and #5 #9): Friedman test blocked by article
# (100 blocks, not the 4 dimensions -- see cmd/friedman for why),
# Wilcoxon signed-rank of LGS against every other metric with
# Holm correction, and the Nemenyi critical difference on average
# rank. Reads output/*.json only, no Ollama or model server needed.
.PHONY: paper-friedman
paper-friedman:
	go run ./cmd/friedman \
		-metrics bleu,rouge,chrf,meteor,smartstring,embedscorer,bertscore,moverscore,smartmodel,bartscore,gptscore,unieval,geval,lgs,lead3sent,lead5sent,lead3whole \
		-output paper/friedman.gen.tex

# Random-split robustness of the lambda* selection protocol
# (reviewer #3 m1): the manuscript selects lambda* on the JSONL-order
# 50/50 dev/test split; this re-runs the whole selection procedure on
# thousands of random 50/50 article splits by merging the per-sample
# scores already in ablation/lgs_recall_*.json and
# ablation/lgs_lead_{dev,test}_l*.json, so it needs neither Ollama nor
# the model server. Depends on benchmark-ablation-recall,
# benchmark-ablation-lead and benchmark-ablation-lead-finer(-test)
# having been run at least once.
.PHONY: paper-splitrobust
paper-splitrobust:
	go run ./cmd/splitrobust -input ablation -splits 2000 -output paper/splitrobust.gen.tex

# Per-dimension lead-bias exponent decision matrix (reviewer #5 #7):
# the ablation table selects one lambda* by mean rho across all four
# dimensions; this selects the dev-argmax lambda separately per
# dimension and verifies it on test. Reads ablation/*.json only.
.PHONY: paper-lambdadim
paper-lambdadim:
	go run ./cmd/lambdadim -input ablation -output paper/lambdadim.gen.tex

# Lead-window controls. No embedder, no model, no hyperparameter: keep
# the first k source sentences and match them with ROUGE-L. `sent`
# reuses LGS's own mean-of-max aggregation so the comparison isolates
# the embedder; `whole` is the cruder single-block variant. These are
# the baselines a reference-free lead-biased metric has to beat, and
# they are CPU-only -- neither Ollama nor the model server is needed.
.PHONY: benchmark-lead
benchmark-lead:
	go run ./cmd/leadbaseline -lead-k 3 -mode sent  -output output/lead3sent.json
	go run ./cmd/leadbaseline -lead-k 5 -mode sent  -output output/lead5sent.json
	go run ./cmd/leadbaseline -lead-k 3 -mode whole -output output/lead3whole.json

.PHONY: benchmark-lead-sweep
benchmark-lead-sweep:
	@mkdir -p ablation
	go run ./cmd/leadbaseline -lead-k 1 -mode sent  -bootstrap 0 -output ablation/lead1sent.json
	go run ./cmd/leadbaseline -lead-k 2 -mode sent  -bootstrap 0 -output ablation/lead2sent.json
	go run ./cmd/leadbaseline -lead-k 3 -mode sent  -bootstrap 0 -output ablation/lead3sent.json
	go run ./cmd/leadbaseline -lead-k 5 -mode sent  -bootstrap 0 -output ablation/lead5sent.json
	go run ./cmd/leadbaseline -lead-k 1 -mode whole -bootstrap 0 -output ablation/lead1whole.json
	go run ./cmd/leadbaseline -lead-k 2 -mode whole -bootstrap 0 -output ablation/lead2whole.json
	go run ./cmd/leadbaseline -lead-k 3 -mode whole -bootstrap 0 -output ablation/lead3whole.json
	go run ./cmd/leadbaseline -lead-k 5 -mode whole -bootstrap 0 -output ablation/lead5whole.json

# Canonical hyperparameters. λ is the lead-bias decay (selected on
# the dev split — see benchmark-ablation-lead). LGS_EMBED_MODEL is
# the canonical sentence-embedder used for the headline run; the
# embedder ablation (benchmark-embedder-ablation) compares it against
# alternatives to show the result is not specific to one backbone.
LGS_LAMBDA      ?= 0.5
LGS_EMBED_MODEL ?= nomic-embed-text

# benchmark-lgs runs the canonical metric on the full SummEval set
# with the dev-selected λ and the canonical embedder. Writes the
# report consumed by paper/summary.gen.tex and paper/comparisons.gen.tex.
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
# ablation/ that back paper/ablation.gen.tex. Two conceptually distinct
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
# rendered into paper/embedders.gen.tex by `paper-embedders`.
# Requires the four embedders to be available in Ollama:
#   ollama pull nomic-embed-text mxbai-embed-large bge-m3 all-minilm
.PHONY: benchmark-embedder-ablation
benchmark-embedder-ablation:
	@mkdir -p ablation
	go run ./cmd/lgs -doc-split all -lead-bias-lambda $(LGS_LAMBDA) -embed-model nomic-embed-text  -bootstrap 0 -output ablation/lgs_embedder_nomic.json
	go run ./cmd/lgs -doc-split all -lead-bias-lambda $(LGS_LAMBDA) -embed-model mxbai-embed-large -bootstrap 0 -output ablation/lgs_embedder_mxbai.json
	go run ./cmd/lgs -doc-split all -lead-bias-lambda $(LGS_LAMBDA) -embed-model bge-m3            -bootstrap 0 -output ablation/lgs_embedder_bge.json
	go run ./cmd/lgs -doc-split all -lead-bias-lambda $(LGS_LAMBDA) -embed-model all-minilm        -bootstrap 0 -output ablation/lgs_embedder_minilm.json
