# LLMBench

Correlation benchmark for summarisation metrics, and the reference implementation of **LNC** (Likelihood–NLI Composite). LNC is a reference-free metric that scores the four SummEval dimensions with two small open models [22, 23].

The benchmark covers:

- 21 metrics on SummEval [1], the Lead-k controls included;
- three transfer corpora: FRANK CNN/DM, FRANK XSum [2] and RoSE CNN/DM [3].

Everything is run and timed on one machine: an ASUS Ascent GX10 with an NVIDIA GB10.

## Setup

Installs the Python dependencies, pulls the Ollama models and converts the AlignScore weights. Pass `PYTHON=<venv python>` when your venv is not active.

```sh
make setup
```

## Start

Starts Ollama and the model server in the background.

```sh
make up
```

## Check

Waits until every model is loaded.

```sh
make health
curl localhost:9200/health
```

## Run all

Computes every score on every corpus, then every table. This takes about 16 h; run it inside `tmux`.

```sh
make reproduce
```

## Run parts

```sh
make benchmark             # SummEval: every metric, then the LNC calibration
make benchmark-lexical     # BLEU, ROUGE-L, ChrF, METEOR, SMART-String
make benchmark-lead        # Lead-3 / Lead-5 controls
make benchmark-modelsrv    # BERTScore, MoverScore, BARTScore, GPTScore, UniEval, AlignScore, CCM, NLIG, SPL
make benchmark-ollama      # EmbedScorer, SMART-Model, LGS, G-Eval, PSC
make benchmark-lnc         # LNC, fitting the calibration on the first 50 SummEval articles
make transfer              # FRANK and RoSE with the frozen LNC calibration
make transfer-rose_cnndm   # one transfer corpus
make paper                 # every table; needs no server
```

One metric on one corpus:

```sh
go run ./cmd/lnc -dataset frank_cnndm
```

## Reproduce from scratch

Deletes every result, rebuilds the corpora from their sources (byte-identical to the committed files) and recomputes everything.

```sh
make clean data reproduce
```

## Stop

```sh
make down
```

## Layout

| path | contents |
|---|---|
| `pkg/dataset/<name>.jsonl` | embedded corpora: `summeval`, `frank_cnndm`, `frank_xsum`, `rose_cnndm` |
| `output/<dataset>/<metric>[_<dimension>].json` | per-sample scores and their correlations with the human ratings |
| `ablation/<dataset>/` | variants, raw dumps, LNC features; the LNC calibration is in `ablation/summeval/` |
| `paper/*.gen.tex` | tables rendered by `make paper` |

## Metrics

| metric | input | cmd | reference |
|---|---|---|---|
| BLEU | reference | `cmd/bleu` | [4] |
| ROUGE-L | reference | `cmd/rouge` | [5] |
| ChrF | reference | `cmd/chrf` | [6] |
| METEOR | reference | `cmd/meteor` | [7] |
| SMART-String, SMART-Model | reference | `cmd/smartstring`, `cmd/smartmodel` | [8] |
| EmbedScorer | reference | `cmd/embedscorer` | [24] |
| BERTScore | reference | `cmd/bertscorer` | [9] |
| MoverScore | reference | `cmd/moverscorer` | [10] |
| BARTScore | reference | `cmd/bartscorer` | [11] |
| GPTScore | reference | `cmd/gptscorer` | [12] |
| UniEval | source + reference | `cmd/unieval` | [13] |
| G-Eval | source | `cmd/geval` | [14] |
| AlignScore | source | `cmd/alignscore` | [15] |
| NLIG (SummaC-ZS) | source | `cmd/nlig` | [16] |
| Lead-3, Lead-5 | source | `cmd/leadbaseline` | — |
| LGS | source | `cmd/lgs` | — |
| CCM | source | `cmd/ccm` | [17, 18] |
| PSC | source | `cmd/psc` | — |
| SPL | source | `cmd/spl` | — |
| **LNC** | source | `cmd/lnc` | [11, 16, 17, 19, 21] |

Statistics tools:

- `cmd/paper`: correlation tables;
- `cmd/compare`: paired bootstrap;
- `cmd/confound`: extractiveness confound [20] and the Pareto frontier;
- `cmd/friedman`: Friedman, Wilcoxon and Nemenyi tests;
- `cmd/lncrobust`: LNC controls.

## Bibliography

1. A. R. Fabbri, W. Kryściński, B. McCann, C. Xiong, R. Socher, D. Radev. SummEval: Re-evaluating Summarization Evaluation. *TACL*, 2021.
2. A. Pagnoni, V. Balachandran, Y. Tsvetkov. Understanding Factuality in Abstractive Summarization with FRANK: A Benchmark for Factuality Metrics. *NAACL*, 2021.
3. Y. Liu, A. R. Fabbri, P. Liu, Y. Zhao, L. Nan, R. Han, S. Han, S. Joty, C.-S. Wu, C. Xiong, D. Radev. Revisiting the Gold Standard: Grounding Summarization Evaluation with Robust Human Evaluation. *ACL*, 2023.
4. K. Papineni, S. Roukos, T. Ward, W.-J. Zhu. BLEU: a Method for Automatic Evaluation of Machine Translation. *ACL*, 2002.
5. C.-Y. Lin. ROUGE: A Package for Automatic Evaluation of Summaries. *Text Summarization Branches Out*, 2004.
6. M. Popović. chrF: character n-gram F-score for automatic MT evaluation. *WMT*, 2015.
7. S. Banerjee, A. Lavie. METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. *ACL Workshop on Intrinsic and Extrinsic Evaluation Measures*, 2005.
8. R. K. Amplayo, P. J. Liu, Y. Zhao, S. Narayan. SMART: Sentences as Basic Units for Text Evaluation. *ICLR*, 2023.
9. T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, Y. Artzi. BERTScore: Evaluating Text Generation with BERT. *ICLR*, 2020.
10. W. Zhao, M. Peyrard, F. Liu, Y. Gao, C. M. Meyer, S. Eger. MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance. *EMNLP*, 2019.
11. W. Yuan, G. Neubig, P. Liu. BARTScore: Evaluating Generated Text as Text Generation. *NeurIPS*, 2021.
12. J. Fu, S.-K. Ng, Z. Jiang, P. Liu. GPTScore: Evaluate as You Desire. *NAACL*, 2024.
13. M. Zhong, Y. Liu, D. Yin, Y. Mao, Y. Jiao, P. Liu, C. Zhu, H. Ji, J. Han. Towards a Unified Multi-Dimensional Evaluator for Text Generation. *EMNLP*, 2022.
14. Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, C. Zhu. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. *EMNLP*, 2023.
15. Y. Zha, Y. Yang, R. Li, Z. Hu. AlignScore: Evaluating Factual Consistency with a Unified Alignment Function. *ACL*, 2023.
16. P. Laban, T. Schnabel, P. N. Bennett, M. A. Hearst. SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization. *TACL*, 2022.
17. Z. Xie, T. Cohn, J. H. Lau. DeltaScore: Fine-Grained Story Evaluation with Perturbations. *Findings of EMNLP*, 2023.
18. E. Mitchell, Y. Lee, A. Khazatsky, C. D. Manning, C. Finn. DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. *ICML*, 2023.
19. G. I. Winata et al. MetaMetrics: Calibrating Metrics for Generation Tasks Using Human Preferences. *ICLR*, 2025.
20. E. Durmus, F. Ladhak, T. Hashimoto. Spurious Correlations in Reference-Free Evaluation of Text Generation. *ACL*, 2022.
21. Q. Jia et al. Zero-shot Faithfulness Evaluation for Text Summarization with Foundation Language Model. *EMNLP*, 2023.
22. P. He, J. Gao, W. Chen. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. *ICLR*, 2023.
23. Qwen Team. Qwen2.5 Technical Report. *arXiv:2412.15115*, 2024.
24. Z. Nussbaum, J. X. Morris, B. Duderstadt, A. Mulyar. Nomic Embed: Training a Reproducible Long Context Text Embedder. *arXiv:2402.01613*, 2024.

[MIT](LICENSE)
