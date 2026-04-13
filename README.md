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

## MCP

```sh
go run ./cmd/mcp
```

## Agent

```sh
go run ./cmd/agent
```

### MCP (ready to use) 

```sh
docker run --rm -p 9090:9090 -v ~/.kube:/home/nonroot/.kube:ro quay.io/containers/kubernetes_mcp_server:latest --port 9090 --kubeconfig /home/nonroot/.kube/config
```

### MCP inspector

```sh
# Streamable HTTP: http://host.docker.internal:9090/mcp
docker run --rm -p 6274:6274 -p 6277:6277 -e HOST=0.0.0.0 -e MCP_PROXY_HOST=0.0.0.0 ghcr.io/modelcontextprotocol/inspector
```

### Dataset

https://huggingface.co/datasets/mteb/summeval

### Temp

```sh
cp ~/.kube/config /tmp/kube-mcp-config
sed -i '' 's|127.0.0.1|host.docker.internal|g' /tmp/kube-mcp-config
sed -i '' 's|certificate-authority-data:.*|insecure-skip-tls-verify: true|g' /tmp/kube-mcp-config

docker run --rm -p 9090:9090 \
  -v /tmp/kube-mcp-config:/home/nonroot/.kube/config:ro \
  quay.io/containers/kubernetes_mcp_server:latest \
  --port 9090 \
  --kubeconfig /home/nonroot/.kube/config
  
docker run --rm -p 6274:6274 -p 6277:6277 -e HOST=0.0.0.0 -e MCP_PROXY_HOST=0.0.0.0 ghcr.io/modelcontextprotocol/inspector


cat /tmp/kube-mcp-config
apiVersion: v1
clusters:
- cluster:
    insecure-skip-tls-verify: true
    server: https://host.docker.internal:49660
  name: docker-desktop
contexts:
- context:
    cluster: docker-desktop
    user: docker-desktop
  name: docker-desktop
current-context: docker-desktop
kind: Config
users:
- name: docker-desktop
  user:
    client-certificate-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURLVENDQWhHZ0F3SUJBZ0lJTnZHYVZrbUt0WUV3RFFZSktvWklodmNOQVFFTEJRQXdGVEVUTUJFR0ExVUUKQXhNS2EzVmlaWEp1WlhSbGN6QWVGdzB5TmpBek1qSXhOVEkzTlRsYUZ3MHlOekF6TWpJeE5UTXlOVGxhTUR3eApIekFkQmdOVkJBb1RGbXQxWW1WaFpHMDZZMngxYzNSbGNpMWhaRzFwYm5NeEdUQVhCZ05WQkFNVEVHdDFZbVZ5CmJtVjBaWE10WVdSdGFXNHdnZ0VpTUEwR0NTcUdTSWIzRFFFQkFRVUFBNElCRHdBd2dnRUtBb0lCQVFEU1pPYWUKT2JvNTNQaW1KUm1UMTFHTkdGTlNUdngyb1VLQVlWQTlsRC9jYktXTlIyaVlFSWRWNWVLS3lSMngxaHkyRE9zOApZeXMySGlwdVZnSVVNTlFyTXdWc0pxK2ZmLys4bGtyOFF3TXl6QXlpSDhkclJ1emx5dEt6NXBpclMwbWRkdjVmCllpOE4xbEo3NS9UZEU0M1hWTkFFS1k4RVc2eGRCd1U1WG13bnZETWR3R1d2bXBVWTRFZmtQQjNNbUJHUXNyZkUKNTRyczAraWp0Y1FiS2FtQ3RSYTQwR3F3VXhxNnZkWW04aEpkRm05M3hyNmpCYm4wVUlSbEQ1NS9rKzd3TlovMQowc25SQzdJSWJXV0haUjBYbnlGVzRHU0R0YmRmWUMrRmZ3WGhpUHQrUHFmRTBNenU1TDdXNUhLdE9nTmc4alQxCm14cU1TYS9kK2U2bDl2ODFBZ01CQUFHalZqQlVNQTRHQTFVZER3RUIvd1FFQXdJRm9EQVRCZ05WSFNVRUREQUsKQmdnckJnRUZCUWNEQWpBTUJnTlZIUk1CQWY4RUFqQUFNQjhHQTFVZEl3UVlNQmFBRkZ4TUFFWXZGMXpDRFhObQpCcC9SSWhGRStqRlVNQTBHQ1NxR1NJYjNEUUVCQ3dVQUE0SUJBUUI5U1pwN1RpZmVhaEJ2S0dTNmVIZTN1NlFFCnphMWk5b0JYSUhOQWhWU0Zrb3gxVE9BQWY2VGRoTkhtY3JnVjk3c3NCOUJHZWNhaWpPbDdwazZSWllGV1pjMWUKQ1dGbzhIVzZMLzFra1VGZG9CeFgyZW9UQUttYWNtVUZTbXBQS1hGYllWOGJGRUFiRlYwendCK3d3amF4aElYcApLUnQ0MmI2aXJka1lQcjN0d1NNVjV0SGdPanZUSWhsNXRKOW5KMzhWakxTZ1ZGbFo1cklGRXpiMzBTM0l3Z3hUCjZQM3lrY2NpYVhUNXoyUHAwMnpwTzY2OTZjcDNUeTNLMFpRQzVld1J5NW14NElMbERZRjNNeUFzYnZDYVg0SWgKTmxBeHYybm02d1BPblcySW0yeWtmU0VhTGF6VnNOK05TOTRGZHdDL1V6bkUyMHpRVjJ5eUZrR0dlZVdkCi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K
    client-key-data: LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlFcEFJQkFBS0NBUUVBMG1UbW5qbTZPZHo0cGlVWms5ZFJqUmhUVWs3OGRxRkNnR0ZRUFpRLzNHeWxqVWRvCm1CQ0hWZVhpaXNrZHNkWWN0Z3pyUEdNck5oNHFibFlDRkREVUt6TUZiQ2F2bjMvL3ZKWksvRU1ETXN3TW9oL0gKYTBiczVjclNzK2FZcTB0Sm5YYitYMkl2RGRaU2UrZjAzUk9OMTFUUUJDbVBCRnVzWFFjRk9WNXNKN3d6SGNCbApyNXFWR09CSDVEd2R6SmdSa0xLM3hPZUs3TlBvbzdYRUd5bXBnclVXdU5CcXNGTWF1cjNXSnZJU1hSWnZkOGErCm93VzU5RkNFWlErZWY1UHU4RFdmOWRMSjBRdXlDRzFsaDJVZEY1OGhWdUJrZzdXM1gyQXZoWDhGNFlqN2ZqNm4KeE5ETTd1UysxdVJ5clRvRFlQSTA5WnNhakVtdjNmbnVwZmIvTlFJREFRQUJBb0lCQUJCSXBQSWNHbHRvekluaQpGelVId09wckMrVXo4eVdrc2pGSTF0MWRRRnErd2ZWMFVHdkI0N3owajVzWm1Sc1pna2F6Z0pxL2tOUkljMnBFCndvdkRnbU1HNDVBM3A2SWJBRi9INjIwbzFwemVOQ0FjSzQxcHJqalVLNGltL0FGTisyMmM0Y1Uwem9XRFVBWlgKc1VPbkdYUlFFOUF1cUpZTmJsZ1RqTEQxcTVOTGxyYXBzYXQ2ZGViZHlDc1ovSEJkMDlTbmFmaUo4ZlpRSFJSegp1NVlIa0tnT0Z2Z1IxRDAyUnBxbmZURDcwNGJtdXNMZFd4Mlk0LzlYWlUvWjdJSWFPOTJ4bmRMNlNKbWg3emRCCmd5Rmk2anFNSTRzMC9ZNU91TUpuQVVaMmFLK2R1UDVKOTkraGZheERDQ0RTQUFKYXRrSWJqSkk5dFViay9HL0kKdzRiVi9NRUNnWUVBNzIwWVFabjBhMDB4eUZwZW9aMVhEb1BRTmpKdG5KbXlvN0ZQT0w3TkhYczVFWnIraUoxNAo5R0ZSZ1o1TTdmTFJteldQY3hza3hOa0NZN2EweXhDcisrelpLVm9jVytZQUJjS2F4cTNvT2FJY05OV28vMm9FCldYbGZxT3B0VmVpSHoyZHBZdENWTThYRStHeU9Kd2ZFT2NKY2Facjg4U0twcTFITnF2YXJ6eEVDZ1lFQTRQVlQKWXAvMnJWcWxKSWZKd2Z2QzRMVGVyQ3ZPMkxHYkVTTGd2ekJDa2lOdzFNOEw3TGlBZmE2NnQ3dlNGY3JuM1NVZgpZK0JlY2kwVm1pSDhkbmRsNDllWXBkRnNoM1hMUDFVVUQzaERoaEVCNG5YcTlqeVdUOEszKzdTMDkzeTZKYTNPCnBrd0lNQlA1cnl4Tm0wREphenVzaGFITXhSQ2QyUUR1NlRaa2RlVUNnWUFkVmdLbzF4SkpxM1cwRk02UGd0WE4KNDN5NWgwaEM3ZG9qa0hBaWhjNWdGRjhUdHlnRTJUYWV5dVhQdWZPM0hBOXVzd3RXa1RiYUg3VFpQdU84RmRqYwp6MUowYktRWTVuK09OUi85eEFVMk9wUzJMSSsrYStFSWpZU1pEOUJCdkhJWGlaWXlFMWlVdFdEREI1b0xVanBLCjBYTzlSTTVGUlhnQWs4OWRhVWYzNFFLQmdRRE00MlJCRFlTMHV6eHlHeUxOaFNvblUxVUQ1eHFNRHFjM1lsYmsKaTJYMmlFVDU3bUhrQnQ4d21YWUNaaFNnT0tBWnNQZjRGYUN2eVJSRnYvS2JTNEFIbHBPM1l4aS8vNjlRVFlMcAozQlZVQkNWOVJ2enJySjhTb2p6RUNnQlE4TTd5Qm0yUzFPa1lNUGxXYkxsNlQvV2pyMFFncWc5QTVUTi9NL1JsCmdGN2JhUUtCZ1FDZG5RS3hLWkh6MEpka0dWdzk3Y3FuaHZPVnpxQmxGL2d4Z0NXQWJkUngrYlExbmltd2p3UmMKRTZublJqWDRhSzNZUWwvd00ycU1jWkJCRlNvZGs2bi8xZnA0YWFqaDJ6WFVxQjJZV0pPN2hTa2NSUldBMXJGSgpkbk4yTWdtc1lpTUhIUkd4d0srUHRZeUIrdjVTTDVnU3ZMTnArMTZvQlJYenZjK2Y0Mk9xTmc9PQotLS0tLUVORCBSU0EgUFJJVkFURSBLRVktLS0tLQo=
    
    
    
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
