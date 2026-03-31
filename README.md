# Model 1 — Qwen

```sh
go run ./cmd/evaluate/main.go -provider ollama -model qwen2.5:3b-instruct -runs 1 -output qwen.json
```

# Model 2 — LLama Pro

```sh
go run ./cmd/evaluate/main.go -provider ollama -model llama-pro:latest -runs 1 -output llama-pro.json
```

## Compare results

```sh
go run ./cmd/compare/main.go -a qwen.json -b llama-pro.json -output compare.json
```

## Show result

```sh
go run ./cmd/report/main.go -file compare.json
```

## Show latex

```sh
go run cmd/latex/main.go -file compare.json -output tables.tex
```

cenoj72302@fun4k.com
Semafor4!

na stoku
