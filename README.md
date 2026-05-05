# LLMBench

Correlation benchmark for summarization metrics on the [SummEval](https://huggingface.co/datasets/mteb/summeval) dataset, and reference implementation of **BGS** (Bidirectional Grounding Score) — an *efficient, reference-free* embedding-only metric for summary quality.

Reference-based metrics (BERTScore, MoverScore, SMART-Model, BLEU, ROUGE-L, ChrF, METEOR, SMART-String, EmbedScorer, BARTScore, GPTScore) are scored against the human-annotated reference summaries. Source-based metrics (G-Eval, BGS) score the candidate against the source article and need no reference. UniEval uses both source and reference. All correlations and paired-bootstrap comparisons are aggregated by `cmd/paper` and `cmd/compare` into the LaTeX tables under `paper/`.

## BGS positioning

BGS is reference-free and runs on a small Ollama embedder (`nomic-embed-text`, 137M params). It is **not** intended to beat UniEval or G-Eval — those win on raw correlation. The claim is that BGS:

- Beats **BERTScore** significantly on consistency (Δρ=+.094, p<.001), ties on coh / flu / rel — while using a much smaller embedder and no human reference.
- Beats **SMART-Model** significantly on coh / con / flu (Δρ ∈ [+.045, +.110]).
- Beats **ChrF** significantly on coherence (Δρ=+.214).
- Outscores BLEU, ROUGE-L, METEOR, SMART-String, MoverScore, BARTScore, EmbedScorer on the point estimate of every SummEval dimension.
- Loses to UniEval on every dimension and to G-Eval on consistency. This is openly reported in `paper/comparisons.tex`.

The paper's contribution is "a reference-free metric you can deploy with a single Ollama model" — efficiency and reference-freeness, not SOTA correlation.

## Requirements

- Go 1.26+
- Python 3.10+ with pip (for the model server: BERTScore, MoverScore, BARTScore, UniEval, GPTScore)
- [Ollama](https://ollama.com) running locally on port 11434 (for EmbedScorer, SMART-Model, BGS, G-Eval). Pull the default models first:
  ```sh
  ollama pull nomic-embed-text
  ollama pull qwen2.5:7b-instruct-q4_K_M
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
| BGS (ours)   | embedding    | Ollama        | source           | `cmd/bgs`          |

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

Writes `paper/summary.tex` and `paper/system.tex` from the JSON reports in `output/`.

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
