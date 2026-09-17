# LLMBench

Correlation benchmark for summarization metrics on the [SummEval](https://huggingface.co/datasets/mteb/summeval) dataset, and reference implementation of **LGS** — an *efficient, reference-free* embedding-only metric for summary quality.

Reference-based metrics (BERTScore, MoverScore, SMART-Model, BLEU, ROUGE-L, ChrF, METEOR, SMART-String, EmbedScorer, BARTScore, GPTScore) are scored against the human-annotated reference summaries. Source-based metrics (G-Eval, LGS) score the candidate against the source article and need no reference. UniEval uses both source and reference. All correlations and paired-bootstrap comparisons are aggregated by `cmd/paper` and `cmd/compare` into the LaTeX tables under `paper/`.

## LGS — formulation

For source `D` and candidate summary `C` split into sentences:

```
w(i)  = exp(-λ · i / n)                    # exponential lead-bias prior
score = mean over c_j of  max_i  w(i) · cos(emb(c_j), emb(s_i))
```

For each candidate sentence `c_j`, find the best-matching source sentence after the cosine has been weighted by source position (`i = 0` is the lead, `n` is source length). Average over candidate sentences. `λ` is the only hyperparameter.

The mean-of-max grounding ("is each summary sentence anchored in some source sentence?") penalises hallucination on the candidate side. The exponential lead-bias prior on the source side is motivated by the well-documented lead bias of CNN/DailyMail-style news (Kedzie et al. 2018, Grusky et al. 2018): salient content concentrates near the article start. The decay `exp(−λ · i / n)` provides a deterministic source-side salience signal that is cheaper than iterative TextRank/LexRank algorithms.

**Held-out selection of λ**. SummEval is split by article into a 50-article development set (first 50 docs in dataset order) and a 50-article test set (last 50). On dev, λ ∈ {0, 0.25, 0.5, 1.0, 2.0} is swept and the value maximising mean Spearman ρ across the four SummEval dimensions is selected. The dev winner is **λ\* = 0.5**: dev mean ρ rises from .233 (λ=0) to .252 (+.019), with all four dimensions improving (coh +.021, con +.007, flu +.011, rel +.036). On the held-out test split λ\*=0.5 also beats the no-prior baseline (mean ρ .319 vs .314).

**How much of λ\* is reproducible** (`make paper-lambdaci`, `paper/lambdaci.gen.tex`). The selection procedure is itself run inside the cluster bootstrap: on each of 5,000 resamples of the dev articles the whole sweep is recomputed and its argmax recorded. Two results, and they point in opposite directions:

- *Switching the prior on is robust.* λ=0 is selected in **0.7%** of dev resamples and **0.3%** of test resamples.
- *The exponent is not identifiable.* λ=0.3 wins 27.6%, λ=0.4 wins 20.4%, the reported λ\*=0.5 wins **17.8%**, λ=0.25 wins 17.5%. The protocol returns the interval **[0.25, 0.6]**, not a point.

Controls on the grid common to all configurations: a fixed configuration compared against *itself* on two independent draws reselects the same λ only **43.5%** of the time (disagreeing symmetrically, 28.6% / 27.9%) — that is the noise floor. Cross-backbone agreement is **37.8%**, i.e. barely below it, so disagreement alone proves nothing; what does separate is direction (nomic picks the larger λ in 61.9% of resamples vs 0.4% reverse). Caveat: changing only *which* 50 articles are used produces the same asymmetry (52.8% vs 0.9%), so the backbone is not isolated as the cause.

## LGS — positioning

LGS is reference-free and runs on a small Ollama embedder (`nomic-embed-text`, 137M params). It is **not** intended to beat UniEval — UniEval still wins on raw correlation. Within the pool of *learned/embedding* metrics, LGS:

- Beats **EmbedScorer** (the whole-text cosine baseline using the *same* embedder) significantly on coh / con / flu (Δρ = +.117 / +.171 / +.116, p<.001), tie on rel — this isolates the contribution of sentence-level grounding + lead-bias prior over a plain whole-text cosine.
- Beats **SMART-Model** significantly on coh / con / flu (Δρ = +.068 / +.134 / +.089, p ≤ .005), tie on rel — same family (sentence-level + embedder), so this isolates the value of reference-freeness + lead-bias prior.
- Beats **BERTScore** significantly on consistency (Δρ=+.118, p<.001), ties on coh / flu / rel — while using a much smaller embedder and no human reference.
- Beats **ChrF** significantly on coherence (Δρ=+.237, p<.001).
- Outscores BLEU, ROUGE-L, METEOR, SMART-String, MoverScore, BARTScore on the point estimate of every SummEval dimension.
- Loses to UniEval on every dimension (p<.001) and to G-Eval on consistency (p<.001); other G-Eval comparisons are statistical ties. This is openly reported in `paper/comparisons.gen.tex`.
- All eight significant comparisons above survive Benjamini-Hochberg correction over the 24 (baseline, dimension) cells of `paper/comparisons.gen.tex`; the weakest survivor is coherence vs SMART-Model at q=.010. `cmd/compare` reports both raw p and adjusted q.
- **Robust to embedder choice** (`paper/embedders.gen.tex`). The canonical metric is run with four sentence-embedder backbones spanning a ~24× parameter range — `all-minilm` (23M), `nomic-embed-text` (137M, headline), `mxbai-embed-large` (335M), `bge-m3` (567M). Mean Spearman ρ stays in [.284, .312]; per-dimension profile shifts (nomic dominates coh / rel, others con / flu) but the metric design is not specific to one backbone.

**But this comparison is confounded, and the honest picture is weaker.** SummEval's mean-of-dimension correlation is heavily driven by *extractiveness* — how much a candidate copies verbatim from the source — and a four-line lead-window heuristic with no model at all (`cmd/leadbaseline`) beats LGS on raw mean Spearman ρ at ~950× lower cost (`paper/frontier.gen.tex`, `paper/ablation.gen.tex` §Pareto). Controlling for extractiveness (`cmd/confound`, `paper/confound.gen.tex`) narrows this: on the confound-partialled axis LGS is not statistically distinguishable from the cheapest lead baseline (Lead-3, Δρ=+.045, p=.064) and ties BERTScore (Δρ=−.043, p=.074), though it does show a marginal edge over Lead-5 (Δρ=+.044, p=.033). A Friedman test across the full 17-metric pool (14 published + 3 lead controls), blocked by article rather than by dimension for statistical power, confirms this is not a fluke of one comparison (`cmd/friedman`, `paper/friedman.gen.tex`): LGS is significantly behind UniEval on every dimension and not reliably ahead of the lead-window controls on consistency or fluency. See `improvements.txt` for the full account and `paper/main.tex` §"Pareto analysis" / §"Extractiveness confound" for the reconciled claims.

The paper's contribution is a reference-free metric with one tunable hyperparameter (lead-bias λ) selected via held-out methodology that prior metric papers skip — but the lead-bias prior itself contributes only +.005 mean ρ held-out and its exponent is not reliably identifiable from a 50-article split (see λ\* discussion above and `paper/splitrobust.gen.tex`), so the paper's central empirical claim is methodological honesty about a confounded benchmark, not a metric that is unambiguously better than the cheapest available baseline.

## Entailment go/no-go probe (unreleased, gates a possible Paper 2)

`improvements.txt` §6 flags cosine similarity as the wrong signal type for the consistency dimension: cosine measures semantic proximity, not entailment, and a paraphrase and a subtle contradiction can sit at nearly the same cosine distance. Consistency is LGS's worst dimension (.227 raw ρ), which is exactly what an entailment signal should help with. `cmd/nliscorer` swaps LGS's own aggregation (mean over candidate sentences of max over source sentences) from embedding cosine to NLI entailment probability, scored by a small cross-encoder (`cross-encoder/nli-deberta-v3-small`, ~140M) served from a new `/nli` endpoint on `cmd/modelsrv` — no Ollama needed, CPU or GPU.

```sh
make benchmark-nli      # writes output/nli.json (needs cmd/modelsrv running with /nli loaded)
make go-no-go-nli       # writes paper/confound-nli-probe.gen.tex + prints the per-dimension breakdown
```

Decision rule: read the **consistency** column of the per-dimension partial-rho breakdown. Every cosine/lexical/positional signal in the pool plateaus around a confound-controlled partial ρ of ~.23 (`paper/confound.gen.tex`). If NLI clears that ceiling on consistency, entailment is a genuinely different signal class and is worth building a metric around (Paper 2 in `improvements.txt` §6). If it plateaus at the same ceiling, the problem was never the signal type. This does not touch the canonical `paper/confound.gen.tex` table — its output is a separate probe file.

## Reproducing the paper

Every number in `paper/*.gen.tex` regenerates from Make targets. Steps below assume Ollama on `localhost:11434` with `nomic-embed-text` pulled and the model server (`cmd/modelsrv`) running on port 9200 for the reference-based baselines.

```sh
# 1. Regenerate every baseline's per-sample scores in output/.
#    Skip this if output/*.json is already populated.
make benchmark                  # ~30 min

# 2. Reproduce the LGS ablations + canonical end-to-end:
#    (a) recall baseline (λ=0) on dev and test → lgs_recall_{dev,test}.json
#    (b) lead-bias sweep λ ∈ {0.25, 0.5, 1.0, 2.0} on dev and test
#        → lgs_lead_{dev,test}_l*.json
#    (c) embedder ablation: canonical λ run with 4 sentence-embedders
#        → lgs_embedder_{nomic,mxbai,bge,minilm}.json
#    (d) canonical run on the full set with the dev-selected λ*
#        → output/lgs.json
#    (e) re-render every paper/*.tex.
make benchmark-lgs-paper        # ~25 min

# 3. (Optional) Run only one ablation slice — useful when a reviewer
#    wants to verify a single component without re-running the full grid:
make benchmark-ablation-recall      # recall baseline only (2 runs, ~2 min)
make benchmark-ablation-lead        # lead-bias sweep only (8 runs, ~7 min)
make benchmark-embedder-ablation    # 4 embedders, full set (~7 min)

# 4. (Optional) Inspect the snapshots and rendered tables:
ls ablation/                        # lgs_recall_*.json + lgs_lead_*.json + lgs_embedder_*.json
cat paper/ablation.gen.tex paper/embedders.gen.tex

# 5. (Optional) Statistics-only targets. These read the per-sample
#    scores already stored in output/ and ablation/, so they need
#    neither Ollama nor the model server:
make paper-lambdaci                 # λ selection uncertainty (~3 min)
make paper-comparisons              # paired bootstrap + BH-FDR (~2 min)
make paper-confound                 # extractiveness confound (~2 min)
make benchmark-lead                 # lead-window controls (~1 min)
make paper-friedman                 # Friedman + Wilcoxon over the full pool (~1 min)
make paper-splitrobust              # λ* random-split robustness (~1 min)
```

The canonical hyperparameters are encoded as Make variables: `LGS_LAMBDA=0.5` (dev-selected) and `LGS_EMBED_MODEL=nomic-embed-text` (the headline embedder). To rerun the canonical with a different choice — for instance, λ=0.25 or a larger embedder:

```sh
make benchmark-lgs LGS_LAMBDA=0.25
make benchmark-lgs LGS_EMBED_MODEL=mxbai-embed-large
```

The dev/test split is by dataset order (deterministic from `eval.NewDataset`'s JSONL emission order — no random seed). The article-level split is implemented in `applyDocSplit` in `cmd/lgs/main.go`.

## Requirements

- Go 1.26+
- Python 3.10+ with pip (for the model server: BERTScore, MoverScore, BARTScore, UniEval, GPTScore)
- [Ollama](https://ollama.com) running locally on port 11434 (for EmbedScorer, SMART-Model, LGS, G-Eval). Pull the default models first:
  ```sh
  ollama pull nomic-embed-text
  ollama pull qwen2.5:7b-instruct-q4_K_M
  ```
  For the embedder ablation (`make benchmark-embedder-ablation`), additionally pull:
  ```sh
  ollama pull mxbai-embed-large
  ollama pull bge-m3
  ollama pull all-minilm
  ```

## Metrics

| Metric       | Family       | Backend       | Input            | cmd                |
|--------------|--------------|---------------|------------------|--------------------|
| BLEU         | lexical      | CPU           | reference        | `cmd/bleu`         |
| ROUGE-L      | lexical      | CPU           | reference        | `cmd/rouge`        |
| ChrF         | lexical      | CPU           | reference        | `cmd/chrf`         |
| METEOR       | lexical      | CPU           | reference        | `cmd/meteor`       |
| SMART-String | lexical      | CPU           | reference        | `cmd/smartstring`  |
| EmbedScorer  | embedding    | Ollama        | reference        | `cmd/embedscorer`  |
| SMART-Model  | embedding    | Ollama        | reference        | `cmd/smartmodel`   |
| BERTScore    | model        | model server  | reference        | `cmd/bertscorer`   |
| MoverScore   | model        | model server  | reference        | `cmd/moverscorer`  |
| BARTScore    | model        | model server  | reference        | `cmd/bartscorer`   |
| GPTScore     | model        | model server  | reference        | `cmd/gptscorer`    |
| G-Eval       | LLM judge    | Ollama        | source           | `cmd/geval`        |
| UniEval      | model        | model server  | source + ref     | `cmd/unieval`      |
| LGS (ours)   | embedding    | Ollama        | source           | `cmd/lgs`          |

G-Eval and UniEval are run once per SummEval dimension (`coherence`, `consistency`, `fluency`, `relevance`).

## Run

Start the model server:

```sh
cd cmd/modelsrv
pip3 install -r requirements.txt
python3 app.py
```

Run the full benchmark (CPU + model server + Ollama):

```sh
make benchmark
```

Or run a subset:

```sh
make benchmark-lexical    # BLEU, ROUGE, ChrF, METEOR, SMART-String
make benchmark-modelsrv   # BERTScore, MoverScore, BARTScore, GPTScore, UniEval (4 dims)
make benchmark-ollama     # EmbedScorer, SMART-Model, G-Eval (4 dims)
```

Each cmd writes a JSON report to `output/<metric>.json` containing per-sample scores plus summary-level and system-level Pearson, Spearman, and Kendall correlations against the SummEval human annotations (with bootstrap 95% CI by default).

## Generate paper tables

```sh
make paper
```

Writes the correlation tables from the JSON reports in `output/`. Each correlation table carries a bootstrap CI in every cell, so Spearman and Kendall are emitted as **separate** tables — side by side they are about twice as wide as the `elsarticle` text block:

| File | Contents | LaTeX label |
|------|----------|-------------|
| `paper/summary.gen.tex`         | summary-level Spearman ρ | `tab:correlations_summary` |
| `paper/summary_kendall.gen.tex` | summary-level Kendall τ  | `tab:correlations_summary_kendall` |
| `paper/system.gen.tex`          | system-level Spearman ρ  | `tab:correlations_system` |
| `paper/system_kendall.gen.tex`  | system-level Kendall τ   | `tab:correlations_system_kendall` |

`cmd/paper` takes `-coeffs` (`spearman`, `kendall`, `pearson`; comma-separated) and `-label` to override the default label.

## Service endpoints

The model server exposes the following endpoints on port 9200.

### BERTScore

Token-level BERTScore (RoBERTa-large). Returns precision, recall, F1.

```sh
curl -X POST http://localhost:9200/bertscore \
     -H "Content-Type: application/json" \
     -d '{"reference": "A Pod is the smallest unit in Kubernetes", "candidate": "A Pod is the basic deployable unit in K8s"}'
```

### MoverScore

Word Mover's Distance with contextual RoBERTa embeddings. Returns similarity score in [0, 1].

```sh
curl -X POST http://localhost:9200/moverscore \
     -H "Content-Type: application/json" \
     -d '{"reference": "A Deployment provides declarative updates for Pods and ReplicaSets.", "candidate": "Deployments manage ReplicaSets and enable declarative Pod updates."}'
```

### BARTScore

BART-based generation likelihood scoring. Returns log-probability normalized to [0, 1].

```sh
curl -X POST http://localhost:9200/bartscore \
     -H "Content-Type: application/json" \
     -d '{"reference": "A Pod is the smallest deployable unit in Kubernetes.", "candidate": "Pods are the smallest deployable units in K8s."}'
```

### UniEval

T5-based Boolean QA evaluator. Dimensions: `coherence`, `consistency`, `fluency`, `relevance`, `overall`, `all`.

```sh
curl -X POST http://localhost:9200/unieval \
     -H "Content-Type: application/json" \
     -d '{"reference": "If a container exceeds its memory limit, the kubelet terminates it with OOMKilled status.", "candidate": "When a Pod uses more memory than allowed, Kubernetes kills it.", "dimension": "overall"}'
```

### GPTScore

GPT-2 log-probability scoring normalized to [0, 1].

```sh
curl -X POST http://localhost:9200/gptscore \
     -H "Content-Type: application/json" \
     -d '{"reference": "A Pod is the smallest deployable unit in Kubernetes.", "candidate": "A Pod is the smallest unit of deployment in Kubernetes."}'
```

[MIT](LICENSE)
