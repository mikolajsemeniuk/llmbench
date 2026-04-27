# LLMBench

Dataset: https://huggingface.co/datasets/mteb/summeval

## Set up Metrics server

```sh
docker compose up --build -d
```

## Run benchmark

```sh
make benchmark
```

## Metrics server endpoints (port 9200)

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

### GPTScore

GPT-2 log-probability scoring normalized to [0, 1]. Returns similarity score.

```sh
curl -X POST http://localhost:9200/gptscore \
     -H "Content-Type: application/json" \
     -d '{"reference": "A Pod is the smallest deployable unit in Kubernetes. It can contain one or more containers that share storage and network resources.", "candidate": "A Pod is the smallest unit of deployment in Kubernetes. It represents a group of one or more containers with shared storage and network."}'
```

## Reranker endpoint (port 8010)

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
