# LLMBench

## BLEU

```sh
go run ./cmd/bleu  -input testdata/samples.json -output bleu_llama.json
```

## ROUGE

```sh
go run ./cmd/rouge -input testdata/samples.json -output rouge_llama.json
```

# BERTScore

```sh
go run ./cmd/bertscore -input testdata/samples.json -embed nomic-embed-text -output bert_llama.json
```

# G-Eval

```sh
go run ./cmd/geval -input testdata/samples.json -judge qwen2.5:7b-instruct-q4_K_M -output geval_llama.json
```
