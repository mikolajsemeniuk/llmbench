# LLMBench

## BLEU

```sh
go run ./cmd/bleu  -input testdata/samples.json -output output/bleu.json
```

## ROUGE

```sh
go run ./cmd/rouge -input testdata/samples.json -output output/rouge.json
```

# BERTScore

```sh
go run ./cmd/bertscore -input testdata/samples.json -embed nomic-embed-text -output output/bert.json
```

# G-Eval

```sh
go run ./cmd/geval -input testdata/samples.json -judge qwen2.5:7b-instruct-q4_K_M -output output/geval.json
```
