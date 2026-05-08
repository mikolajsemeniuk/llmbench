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

## LGS — positioning

LGS is reference-free and runs on a small Ollama embedder (`nomic-embed-text`, 137M params). It is **not** intended to beat UniEval — UniEval still wins on raw correlation. The claim is that LGS:

- Beats **EmbedScorer** (the whole-text cosine baseline using the *same* embedder) significantly on coh / con / flu (Δρ = +.117 / +.171 / +.116, p<.001), tie on rel — this isolates the contribution of sentence-level grounding + lead-bias prior over a plain whole-text cosine.
- Beats **SMART-Model** significantly on coh / con / flu (Δρ = +.068 / +.134 / +.089, p ≤ .005), tie on rel — same family (sentence-level + embedder), so this isolates the value of reference-freeness + lead-bias prior.
- Beats **BERTScore** significantly on consistency (Δρ=+.118, p<.001), ties on coh / flu / rel — while using a much smaller embedder and no human reference.
- Beats **ChrF** significantly on coherence (Δρ=+.237, p<.001).
- Outscores BLEU, ROUGE-L, METEOR, SMART-String, MoverScore, BARTScore on the point estimate of every SummEval dimension.
- Loses to UniEval on every dimension (p<.001) and to G-Eval on consistency (p<.001); other G-Eval comparisons are statistical ties. This is openly reported in `paper/comparisons.gen.tex`.
- **Robust to embedder choice** (`paper/embedders.gen.tex`). The canonical metric is run with four sentence-embedder backbones spanning a ~24× parameter range — `all-minilm` (23M), `nomic-embed-text` (137M, headline), `mxbai-embed-large` (335M), `bge-m3` (567M). Mean Spearman ρ stays in [.284, .312]; per-dimension profile shifts (nomic dominates coh / rel, others con / flu) but the metric design is not specific to one backbone.

The paper's contribution is "a reference-free metric you can deploy with a single Ollama model, with one tunable hyperparameter (lead-bias λ) selected via held-out methodology that prior metric papers skip" — efficiency, reference-freeness, methodological honesty, and an empirically-validated structural prior.

## Reproducing the paper

Every number in `paper/{summary,system,ablation,comparisons}.tex` regenerates from Make targets. Steps below assume Ollama on `localhost:11434` with `nomic-embed-text` pulled and the model server (`cmd/modelsrv`) running on port 9200 for the reference-based baselines.

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

Writes `paper/summary.gen.tex` and `paper/system.gen.tex` from the JSON reports in `output/`.

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

### Reranker (port 8010)

Cross-encoder reranker for semantic relevance scoring.

```sh
curl -X POST http://localhost:8010/v1/rerank \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Co to jest uczenie maszynowe?",
       "documents": [
         "Uczenie maszynowe to dział sztucznej inteligencji.",
         "Przepis na szarlotkę wymaga jabłek i mąki.",
         "Algorytmy ML pozwalają systemom uczyć się na podstawie danych."
       ],
       "top_n": 3
     }'
```

## License

[MIT](LICENSE)
