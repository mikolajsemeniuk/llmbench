# Model 1 — Qwen

```sh
go run ./cmd/evaluate/main.go -provider ollama -model qwen2.5:3b-instruct -runs 5 -output qwen.json
```

# Model 2 — LLama Pro

```sh
go run ./cmd/evaluate/main.go -provider ollama -model llama-pro:latest -runs 5 -output llama-pro.json
```

## Compare results

```sh
go run ./cmd/compare/main.go -a qwen.json -b llama-pro.json -output compare.json
```

## Show result

```sh
go run ./cmd/reportv2/main.go -file compare.json
```
