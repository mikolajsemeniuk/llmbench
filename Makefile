# The venv used on the GB10 when present; override with PYTHON=<python>.
PYTHON        ?= $(or $(wildcard $(HOME)/.venvs/llmbench/bin/python),python3)
OLLAMA_MODELS  = nomic-embed-text qwen2.5:7b-instruct-q4_K_M
DIMS           = coherence consistency fluency relevance

# Transfer corpora and the dimension each one annotates.
CORPORA          = frank_cnndm frank_xsum rose_cnndm
DIM_frank_cnndm  = consistency
DIM_frank_xsum   = consistency
DIM_rose_cnndm   = relevance

# Python dependencies, Ollama models, and the AlignScore weights converted
# once to safetensors (the server never unpickles the checkpoint).
.PHONY: setup
setup:
	$(PYTHON) -m pip install -r cmd/modelsrv/requirements.txt
	for m in $(OLLAMA_MODELS); do ollama pull $$m || exit 1; done
	$(PYTHON) cmd/modelsrv/alignscore_convert.py

# Starts Ollama (unless it already runs) and the model server, detached
# from the shell. The server loads every model before it listens. Fails
# at once when PYTHON lacks the server's dependencies.
.PHONY: up
up:
	@$(PYTHON) -c "import torch, transformers, flask" 2>/dev/null || { echo "$(PYTHON) lacks the model-server dependencies: run make setup, or pass PYTHON=<venv python>"; exit 1; }
	curl -sf localhost:11434/api/tags >/dev/null || (setsid ollama serve >/dev/null 2>&1 &)
	curl -sf localhost:9200/health >/dev/null || (cd cmd/modelsrv && setsid $(PYTHON) app.py >/dev/null 2>&1 &)

# Waits up to 10 minutes for the model server (failing at once if its
# process is gone), then checks that it loaded every model and that Ollama
# has the models the benchmark uses.
.PHONY: health
health:
	@for i in $$(seq 120); do curl -sf localhost:9200/health >/dev/null && break; pgrep -f '[a]pp\.py' >/dev/null || { echo "model server: not running (make up)"; exit 1; }; sleep 5; done
	@curl -sf localhost:9200/health | grep -q '"status": *"ok"' || { echo "model server: not ready"; curl -s localhost:9200/health; exit 1; }
	@echo "model server: ok"
	@for m in $(OLLAMA_MODELS); do curl -sf localhost:11434/api/tags | grep -q "\"$$m" || { echo "ollama: $$m missing"; exit 1; }; done
	@echo "ollama: ok"

.PHONY: down
down:
	fuser -k 9200/tcp || true

# Every score on every corpus, then every table.
.PHONY: reproduce
reproduce: benchmark transfer paper

# Rebuilds pkg/dataset/*.jsonl from their sources (byte-identical to the
# committed files).
.PHONY: data
data:
	$(PYTHON) pkg/dataset/prepare.py

# Removes every generated result: scores, ablations, the LNC calibration,
# the cached PSC paraphrases and the rendered tables. make clean data
# reproduce then rebuilds the whole repository state.
.PHONY: clean
clean:
	rm -rf output ablation paper/*.gen.tex

.PHONY: benchmark
benchmark: benchmark-lexical benchmark-lead benchmark-modelsrv benchmark-ollama benchmark-lnc

.PHONY: benchmark-lexical
benchmark-lexical:
	go run ./cmd/bleu
	go run ./cmd/rouge
	go run ./cmd/chrf
	go run ./cmd/meteor
	go run ./cmd/smartstring

# Lead-window controls: the first k source sentences matched with
# ROUGE-L, no model at all. CPU only.
.PHONY: benchmark-lead
benchmark-lead:
	go run ./cmd/leadbaseline -lead-k 3 -mode sent
	go run ./cmd/leadbaseline -lead-k 5 -mode sent
	go run ./cmd/leadbaseline -lead-k 3 -mode whole

# CCM, NLIG and SPL are LNC's building blocks, reported on their own as
# the negative results / components of the analysis.
.PHONY: benchmark-modelsrv
benchmark-modelsrv:
	go run ./cmd/bertscorer
	go run ./cmd/moverscorer
	go run ./cmd/bartscorer
	go run ./cmd/gptscorer
	for d in $(DIMS); do go run ./cmd/unieval -dimension $$d || exit 1; done
	go run ./cmd/alignscore
	go run ./cmd/ccm
	go run ./cmd/nlig
	go run ./cmd/spl

# G-Eval is stochastic at temperature>0: GEVAL_RUNS independent runs at
# GEVAL_TEMPERATURE are averaged (the original paper averaged 20 GPT-4
# samples), which also gives the across-run mean±std of the LLM judge.
# PSC needs both servers: Ollama writes the source paraphrase (cached in
# ablation/summeval/psc_paraphrases.json), the model server scores.
GEVAL_RUNS        ?= 3
GEVAL_TEMPERATURE ?= 0.7
.PHONY: benchmark-ollama
benchmark-ollama:
	go run ./cmd/embedscorer
	go run ./cmd/smartmodel
	go run ./cmd/lgs
	for d in $(DIMS); do go run ./cmd/geval -dimension $$d -runs $(GEVAL_RUNS) -temperature $(GEVAL_TEMPERATURE) || exit 1; done
	go run ./cmd/psc

# LNC on SummEval: computes the features, fits the ridge calibration on
# the first 50 articles only and freezes it in
# ablation/summeval/lnc_calibration.json.
.PHONY: benchmark-lnc
benchmark-lnc:
	go run ./cmd/lnc

# LNC applies the frozen SummEval calibration on every corpus but
# SummEval. G-Eval runs once, greedily, to fit the compute budget.
.PHONY: transfer
transfer: $(addprefix transfer-,$(CORPORA))

transfer-%:
	go run ./cmd/lnc -dataset $*
	go run ./cmd/alignscore -dataset $*
	go run ./cmd/nlig -dataset $*
	go run ./cmd/lgs -dataset $*
	go run ./cmd/leadbaseline -dataset $* -lead-k 3 -mode sent
	go run ./cmd/leadbaseline -dataset $* -lead-k 5 -mode sent
	go run ./cmd/unieval -dataset $* -dimension $(DIM_$*)
	go run ./cmd/geval -dataset $* -dimension $(DIM_$*) -runs 1 -temperature 0

.PHONY: paper
paper: paper-summary paper-system paper-comparisons paper-confound paper-friedman paper-lncrobust paper-transfer

# Every SummEval table is computed on the held-out last 50 articles, the
# half LNC is not calibrated on, for every metric alike.
#
# Each correlation table carries a bootstrap CI per cell, so Spearman and
# Kendall are separate tables (side by side they overflow the page).
.PHONY: paper-summary
paper-summary:
	go run ./cmd/paper -doc-split last50 -ci -level summary -coeffs spearman -output paper/summary.gen.tex
	go run ./cmd/paper -doc-split last50 -ci -level summary -coeffs kendall -label tab:correlations_summary_kendall -output paper/summary_kendall.gen.tex

.PHONY: paper-system
paper-system:
	go run ./cmd/paper -doc-split last50 -ci -level system -coeffs spearman -output paper/system.gen.tex
	go run ./cmd/paper -doc-split last50 -ci -level system -coeffs kendall -label tab:correlations_system_kendall -output paper/system_kendall.gen.tex

# LNC is calibrated on the first 50 articles, so its paired comparisons
# run on the held-out last 50, on raw and on copy-partialled Spearman.
LNC_BASELINES = unieval,geval,alignscore,nlig,lgs,bertscore,gptscore,bartscore,lead5sent
.PHONY: paper-comparisons
paper-comparisons:
	go run ./cmd/compare -metric lnc -doc-split last50 -baselines $(LNC_BASELINES) -bootstrap 5000 -output paper/lnc_comparisons.gen.tex
	go run ./cmd/compare -metric lnc -doc-split last50 -partial -baselines $(LNC_BASELINES) -bootstrap 5000 -output paper/lnc_comparisons_partial.gen.tex

# How much of each metric's correlation is extractiveness (copy rate),
# and the cost--quality Pareto frontier on the raw and partialled axes.
.PHONY: paper-confound
paper-confound:
	go run ./cmd/confound -doc-split last50 -bootstrap 2000 -ours lnc -output paper/confound.gen.tex

# Friedman test blocked by article, Wilcoxon of LNC against every other
# metric (Holm), Nemenyi critical difference.
.PHONY: paper-friedman
paper-friedman:
	go run ./cmd/friedman -doc-split last50 -target lnc \
		-metrics bleu,rouge,chrf,meteor,smartstring,embedscorer,bertscore,moverscore,smartmodel,bartscore,gptscore,unieval,geval,alignscore,nlig,lgs,lead3sent,lead5sent,lnc \
		-output paper/friedman.gen.tex

# Random article splits, system-disjoint splits and the same ridge over
# existing metrics (reads the LNC feature dump).
.PHONY: paper-lncrobust
paper-lncrobust:
	go run ./cmd/lncrobust -output paper/lnc_robust.gen.tex

.PHONY: paper-transfer
paper-transfer: $(addprefix paper-transfer-,$(CORPORA))

paper-transfer-%:
	go run ./cmd/compare -dataset $* -dims $(DIM_$*) -partial -metric lnc -baselines alignscore,nlig,lgs,lead3sent,lead5sent,unieval,geval -bootstrap 5000 -output paper/transfer_$*_comparisons.gen.tex
	go run ./cmd/confound -dataset $* -dims $(DIM_$*) -bootstrap 2000 -ours lnc -output paper/transfer_$*_confound.gen.tex

# The manuscript: main.tex only includes the numbered section files and
# the rendered *.gen.tex tables.
.PHONY: pdf
pdf:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
