# LLMBench

Dataset: https://huggingface.co/datasets/mteb/summeval

## Set up Metrics server

```sh
docker compose up --build -d
```

## Metrics (CLI)

All commands accept `-input` (path to SummEval JSONL) and `-output` (path for JSON report, `-` for stdout).

### BLEU

```sh
go run ./cmd/bleu -input model_annotations.aligned.scored.jsonl -output output/bleu.json
```

### ROUGE

```sh
go run ./cmd/rouge -input model_annotations.aligned.scored.jsonl -output output/rouge.json
```

### METEOR

```sh
go run ./cmd/meteor -input model_annotations.aligned.scored.jsonl
```

### ChrF

```sh
go run ./cmd/chrf -input model_annotations.aligned.scored.jsonl
```

### SMART-String

```sh
go run ./cmd/smartstring -input model_annotations.aligned.scored.jsonl
```

### SMART-Model (embedding cosine similarity via Ollama)

```sh
go run ./cmd/smartmodel -input model_annotations.aligned.scored.jsonl -host http://localhost:11434 -embed nomic-embed-text
```

### EmbedScorer (sentence-level cosine similarity via Ollama)

```sh
go run ./cmd/embedscorer -input model_annotations.aligned.scored.jsonl -host http://localhost:11434 -embed nomic-embed-text
```

### BARTScore (log-probability via Ollama)

```sh
go run ./cmd/bartscorer -input model_annotations.aligned.scored.jsonl -host http://localhost:11434 -model qwen2.5:3b-instruct -output output/bartscorer.json
```

### BERTScore (canonical RoBERTa-large via metrics server)

```sh
go run ./cmd/bertscorer -input model_annotations.aligned.scored.jsonl -host http://localhost:9200 -output output/bertscorer.json
```

### MoverScore (Word Mover's Distance via metrics server)

```sh
go run ./cmd/moverscorer -input model_annotations.aligned.scored.jsonl -host http://localhost:9200 -output output/moverscorer.json
```

### UniEval (T5-based Boolean QA via metrics server)

```sh
go run ./cmd/unieval -input model_annotations.aligned.scored.jsonl -host http://localhost:9200 -dimension overall -output output/unieval.json
```

### GPTScore (log-probability via metrics server)

```sh
go run ./cmd/gptscorer -input model_annotations.aligned.scored.jsonl -host http://localhost:9200 -output output/gptscorer.json
```

### G-Eval (LLM-as-judge via Ollama)

```sh
go run ./cmd/geval -input model_annotations.aligned.scored.jsonl -host http://localhost:11434 -judge qwen2.5:7b-instruct-q4_K_M -output output/geval.json
```

### GPTScore

```sh
curl -X POST http://localhost:9200/gptscore \
  -H "Content-Type: application/json" \
  -d '{"reference": "A Pod is the smallest deployable unit in Kubernetes. It can contain one or more containers that share storage and network resources.", "candidate": "A Pod is the smallest unit of deployment in Kubernetes. It represents a group of one or more containers with shared storage and network."}'
```

### BERTScore
Canonical token-level BERTScore (RoBERTa-large). Returns precision, recall, F1.
```sh
curl -X POST http://localhost:9200/bertscore \
     -H "Content-Type: application/json" \
     -d '{"reference": "A Pod is the smallest unit in Kubernetes", "candidate": "A Pod is the basic deployable unit in K8s"}'
```

### MoverScore
Word Mover's Distance with contextual RoBERTa embeddings. Returns similarity score [0, 1].
```sh
curl -X POST http://localhost:9200/moverscore \
     -H "Content-Type: application/json" \
     -d '{"reference": "A Deployment provides declarative updates for Pods and ReplicaSets.", "candidate": "Deployments manage ReplicaSets and enable declarative Pod updates."}'
```

### UniEval
T5-based Boolean QA evaluator. Dimensions: `coherence`, `consistency`, `fluency`, `relevance`, `overall`, `all`.
```sh
curl -X POST http://localhost:9200/unieval \
     -H "Content-Type: application/json" \
     -d '{"reference": "If a container exceeds its memory limit, the kubelet terminates it with OOMKilled status.", "candidate": "When a Pod uses more memory than allowed, Kubernetes kills it.", "dimension": "overall"}'
```

All dimensions at once:
```sh
curl -X POST http://localhost:9200/unieval \
     -H "Content-Type: application/json" \
     -d '{"reference": "A ReplicaSet ensures a specified number of pod replicas are running at any given time.", "candidate": "ReplicaSets maintain a stable set of replica Pods running at all times.", "dimension": "all"}'
```

### AlignScore
Unified alignment scoring (NLI+QA based). Returns similarity score [0, 1].
```sh
curl -X POST http://localhost:9200/alignscore \
     -H "Content-Type: application/json" \
     -d '{"reference": "A Pod is the smallest deployable unit in Kubernetes.", "candidate": "Pods are the basic building blocks of Kubernetes workloads."}'
```

## Reranker endpoint
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

---

## Metric taxonomy

```
Bez sieci neuronowych:    BLEU, ROUGE, METEOR, chrF
                          └── porównanie n-gramów / stringów

Encoder (BERT-scale):     BERTScore, MoverScore, BLEURT, AlignScore, UniEval
                          └── embeddingi + similarity / fine-tuned regresja / Boolean QA

Hybrydowe:                SMART (string lub model), BARTScore (seq2seq scoring)
                          └── formuła + opcjonalny model

LLM (scoring):            GPTScore
                          └── log-probability generowania

LLM (judge):              G-Eval, Fusion-Eval, Prometheus 1/2, MT-Bench
                          └── prompt → ocena / feedback

LLM (atomowa):            FActScore
                          └── dekompozycja → retrieval → weryfikacja

Fuzja:                    Faithfulness Metric Fusion, Multi-Layered Evaluation
                          └── model drzewiasty / voting na wyjściach różnych metryk
```

## Correlation results (SummEval, summary-level Spearman ρ / Kendall τ)

| Metryka        | Coherence    | Consistency  | Fluency      | Relevance    |
|                |   ρ   |  τ   |   ρ   |  τ   |   ρ   |  τ   |   ρ   |  τ   |
|----------------|-------|------|-------|------|-------|------|-------|------|
| BLEU-4         | 0.102 | 0.08 | 0.137 | 0.10 | 0.066 | 0.05 | 0.206 | 0.16 |
| ROUGE-L        | 0.163 | 0.12 | 0.140 | 0.11 | 0.108 | 0.08 | 0.202 | 0.15 |
