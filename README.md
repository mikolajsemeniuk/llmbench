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

**How much of λ\* is reproducible** (`make paper-lambdaci`, `paper/lambdaci.gen.tex`). The selection procedure is itself run inside the cluster bootstrap: on each of 5,000 resamples of the dev articles the whole sweep is recomputed and its argmax recorded. Two results, and they point in opposite directions:

- *Switching the prior on is robust.* λ=0 is selected in **0.7%** of dev resamples and **0.3%** of test resamples.
- *The exponent is not identifiable.* λ=0.3 wins 27.6%, λ=0.4 wins 20.4%, the reported λ\*=0.5 wins **17.8%**, λ=0.25 wins 17.5%. The protocol returns the interval **[0.25, 0.6]**, not a point.

Controls on the grid common to all configurations: a fixed configuration compared against *itself* on two independent draws reselects the same λ only **43.5%** of the time (disagreeing symmetrically, 28.6% / 27.9%) — that is the noise floor. Cross-backbone agreement is **37.8%**, i.e. barely below it, so disagreement alone proves nothing; what does separate is direction (nomic picks the larger λ in 61.9% of resamples vs 0.4% reverse). Caveat: changing only *which* 50 articles are used produces the same asymmetry (52.8% vs 0.9%), so the backbone is not isolated as the cause.

## LGS — positioning

LGS is reference-free and runs on a small Ollama embedder (`nomic-embed-text`, 137M params). It is **not** intended to beat UniEval — UniEval still wins on raw correlation. Within the pool of *learned/embedding* metrics, LGS:

- Beats **EmbedScorer** (the whole-text cosine baseline using the *same* embedder) significantly on coh / con / flu (Δρ = +.117 / +.171 / +.116, p<.001), tie on rel — this isolates the contribution of sentence-level grounding + lead-bias prior over a plain whole-text cosine.
- Beats **SMART-Model** significantly on coh / con / flu (Δρ = +.068 / +.134 / +.089, p ≤ .005), tie on rel — same family (sentence-level + embedder), so this isolates the value of reference-freeness + lead-bias prior.
- Beats **BERTScore** significantly on consistency (Δρ=+.118, p<.001), ties on coh / flu / rel — while using a much smaller embedder and no human reference.
- Beats **ChrF** significantly on coherence (Δρ=+.237, p<.001).
- Outscores BLEU, ROUGE-L, METEOR, SMART-String, MoverScore, BARTScore on the point estimate of every SummEval dimension.
- Loses to UniEval on every dimension (p<.001) and to G-Eval on consistency (p<.001); other G-Eval comparisons are statistical ties. This is openly reported in `paper/comparisons.gen.tex`.
- All eight significant comparisons above survive Benjamini-Hochberg correction over the 24 (baseline, dimension) cells of `paper/comparisons.gen.tex`; the weakest survivor is coherence vs SMART-Model at q=.010. `cmd/compare` reports both raw p and adjusted q.
- **Robust to embedder choice** (`paper/embedders.gen.tex`). The canonical metric is run with four sentence-embedder backbones spanning a ~24× parameter range — `all-minilm` (23M), `nomic-embed-text` (137M, headline), `mxbai-embed-large` (335M), `bge-m3` (567M). Mean Spearman ρ stays in [.284, .312]; per-dimension profile shifts (nomic dominates coh / rel, others con / flu) but the metric design is not specific to one backbone.

**But this comparison is confounded, and the honest picture is weaker.** SummEval's mean-of-dimension correlation is heavily driven by *extractiveness* — how much a candidate copies verbatim from the source — and a four-line lead-window heuristic with no model at all (`cmd/leadbaseline`) beats LGS on raw mean Spearman ρ at ~950× lower cost (`paper/frontier.gen.tex`, `paper/ablation.gen.tex` §Pareto). Controlling for extractiveness (`cmd/confound`, `paper/confound.gen.tex`) narrows this: on the confound-partialled axis LGS is not statistically distinguishable from the cheapest lead baseline (Lead-3, Δρ=+.045, p=.064) and ties BERTScore (Δρ=−.043, p=.074), though it does show a marginal edge over Lead-5 (Δρ=+.044, p=.033). A Friedman test across the full 17-metric pool (14 published + 3 lead controls), blocked by article rather than by dimension for statistical power, confirms this is not a fluke of one comparison (`cmd/friedman`, `paper/friedman.gen.tex`): LGS is significantly behind UniEval on every dimension and not reliably ahead of the lead-window controls on consistency or fluency. See `improvements.txt` for the full account and `paper/main.tex` §"Pareto analysis" / §"Extractiveness confound" for the reconciled claims.

The paper's contribution is a reference-free metric with one tunable hyperparameter (lead-bias λ) selected via held-out methodology that prior metric papers skip — but the lead-bias prior itself contributes only +.005 mean ρ held-out and its exponent is not reliably identifiable from a 50-article split (see λ\* discussion above and `paper/splitrobust.gen.tex`), so the paper's central empirical claim is methodological honesty about a confounded benchmark, not a metric that is unambiguously better than the cheapest available baseline.

## CCM — Copy-invariant Contrastive Margin

LGS's source-similarity signal is dominated by extractiveness (see above), and every variant built from source–candidate similarity plateaus at the same confound-partialled ceiling. CCM swaps the signal class. For source `x`, candidate `y` and `K` automatically generated perturbations `y'_k` of the candidate itself:

```
score = log P(y | x) − mean_k log P(y'_k | x)
```

under a small causal LM (`Qwen/Qwen2.5-1.5B`, served by the model server at `/ccm`). The perturbations are local, meaning-changing edits (`pkg/metrics/ccm.go`):

- **entity**: swap a proper noun for another one from the source.
- **number**: change a number.
- **negation**: add or remove a negation after an auxiliary verb.
- **order**: swap two sentences.

`y` and every `y'_k` therefore have almost the same copy rate. Whatever makes `log P(y|x)` high only because `y` copies the source also makes `log P(y'_k|x)` high, and cancels out in the difference.

- **Cost.** One source pass per article (its KV cache is reused for all 16 candidates × (K+1) variants), plus one short unconditional pass for the ablations.
- **Hyperparameters.** `K=8` and the canonical variant (plain conditional margin) are fixed a priori. Nothing is tuned on SummEval.
- **Prior art.** DeltaScore (Xie et al., Findings EMNLP 2023) uses `log p(s|c) − log p(s'|c)` with a single perturbation for story evaluation. DetectGPT / Fast-DetectGPT average over K perturbations, but unconditionally and for machine-text detection. CCM applies the K-perturbation, source-conditioned form to summarisation, with factual perturbation families aimed at copy-invariance. The copy/extractiveness motivation follows Durmus et al. (ACL 2022) and Ladhak et al. (ACL 2022).

```sh
make benchmark-ccm      # needs the model server; writes output/ccm.json and ablation/ccm_*.json
```

The same forward passes also yield three ablations and a per-sample dump:

- `ablation/ccm_logp.json`: `log P(y|x)/|y|` with no contrast (BARTScore s→h style, same LM).
- `ablation/ccm_zmargin.json`: the margin divided by the std of the perturbed likelihoods.
- `ablation/ccm_pmimargin.json`: the conditional margin minus the same margin without the source.
- `ablation/ccm_raw.json`: every log-likelihood, perturbation family and perturbed text.

**Result: the canonical margin is not copy-invariant.** It scores raw ρ̄ .225, partial ρ̄ .124 and ρ_copy .43, below LGS on both axes. A plausible cause: with the source in context, the LM predicts copied spans almost deterministically. Perturbing a copied span therefore costs more likelihood than perturbing a paraphrase, so the margin rewards copying instead of cancelling it. The entity and number families carry almost no consistency signal (partial ρ ≈ .10).

## CCM-D — per-dimension CCM

CCM-D reuses CCM's forward passes and assigns one signal to each SummEval dimension (`pkg/metrics/ccm.go`):

| dimension   | signal |
|-------------|--------|
| coherence   | z-normalised margin over all perturbation families |
| consistency | `log P(y|x) / |y|` |
| fluency     | `log P(y|x)` |
| relevance   | `log P(y|x)/|y| / σ₁ + margin(order) / σ₂` (σ from the dev split) |

**Selection protocol.** The assignment was chosen on the dev half (first 50 articles), per dimension, by copy-partialled Spearman ρ, from a fixed pool of seven signals:

- `log P(y|x)/|y|` and `log P(y|x)`
- `log P(y)/|y|`
- order margin and negation margin
- z-margin
- `log P(y|x)/|y|` + order margin

It was then verified once on the held-out last 50 articles. Caveat: on coherence the z-margin beat the `log P(y|x)/|y|` + order combination by only .001 on dev.

**Held-out test half (last 50 articles).** Mean Spearman ρ with the matching-dimension scorer for UniEval and G-Eval:

| metric    | raw ρ̄ | partial ρ̄ |
|-----------|-------|-----------|
| UniEval   | .473  | .424 |
| **CCM-D** | **.393** | **.297** |
| G-Eval    | .338  | .300 |
| BERTScore | .269  | .269 |
| LGS       | .319  | .230 |
| GPTScore  | .290  | .194 |
| Lead-5    | .292  | .147 |

**Full set** (`make paper-ccmd`; this includes the dev half used for selection):

- Raw ρ̄ .388, partial ρ̄ .290 [.260, .318], ρ_copy .74.
- Paired bootstrap with BH-FDR over 32 cells against UniEval, G-Eval, LGS, BERTScore, GPTScore, BARTScore, Lead-5 and Lead-3: 16 significant wins, 14 ties, 2 losses. Both losses are to UniEval (coherence, relevance).
- Against G-Eval: a win on fluency, ties elsewhere.

**Open issues:**

- **Cost is not yet like-for-like.** CCM-D's 335 ms/sample was measured on a GB10 GPU (Qwen2.5-1.5B, fp32 weights with TF32 matmuls). The other rows are Apple-CPU numbers from the original runs.
- **Still largely a copy detector.** ρ_copy is .74, driven by the likelihood signals.
- **The signals have prior art.** `log P(y|x)` scoring: BARTScore s→h, GPTScore, FFLM, HaRiM+. Sentence-order contrast: DeltaScore, Laban et al. 2021. The contribution is the dimension-wise assembly and the protocol, not a new signal.

## PSC — Paraphrased-Source Conditioning

PSC tests the mechanism behind CCM's failure directly. It recomputes every CCM signal as `log P(· | x̃)`, where `x̃` is a meaning-preserving paraphrase of the source:

- **Paraphrase generation.** `qwen2.5:7b-instruct` (Ollama) rewrites the source once per article, in 3-sentence chunks at temperature 0. The result is cached in `ablation/psc_paraphrases.json`.
- **Paired design.** Perturbations, seed and scoring LM are identical to CCM, so PSC vs CCM differs only in the conditioning text.
- **Pre-registered primary metric.** `pscd_<dim>` applies the CCM-D recipe unchanged.

```sh
make benchmark-psc   # needs Ollama + model server
make paper-pscd      # paired bootstrap + confound table
```

**Prior art.** Freitag et al. (EMNLP 2020), Bawden et al. (2020) and Para-Ref (NAACL 2024) paraphrase the *reference* to reduce overlap bias. Prism (Thompson & Post, EMNLP 2020) documents the copy bonus in likelihood scoring. Dreyer et al. (EACL Findings 2023) adjust factuality metrics for abstractiveness after the fact. We found no prior work that paraphrases the *conditioning source*.

**Result: the mechanism is confirmed, but the metric does not win.** Full set:

| metric | raw ρ̄ | partial ρ̄ | ρ_copy (mean over dims) | ms/sample |
|---|---|---|---|---|
| CCM-D (source x) | .388 | .290 | .54 | 335 |
| PSC-D (paraphrase x̃) | .272 | .257 [.226, .288] | .12 | 736 (364 paraphrase + 372 scoring) |
| `log P(y|x)/|y|` | .355 | .235 | .74 | — |
| `log P(y|x̃)/|y|` | .260 | .223 | .18 | — |

- **Paraphrasing removes most of the copy bonus.** The paraphrase shares 21% of its bigrams with the source. ρ_copy of the likelihood drops from .74 to .18.
- **The copy advantage is mostly copying.** The per-summary diagnostic `(log P(y|x) − log P(y|x̃))/|y|` correlates .57 with copy rate.
- **But the removed part also carried signal.**
  - The partial axis does not improve: .290 → .257 on the full set, .297 → .239 on the held-out test half.
  - Consistency and fluency lose significantly against CCM-D: Δρ −.21 and −.23.
  - Paired comparisons over 28 cells: 2 wins (BERTScore on consistency, Lead-5 on relevance), 17 ties, 9 losses.
- **Paraphrase quality is a confound.** Manual inspection finds occasional small factual drifts, e.g. an added "During an earlier part of the season".

## NLIG — NLI grounding (SummaC-ZS construction)

`cmd/nlig` checks every summary sentence against every 1- and 2-sentence source window. It uses DeBERTa-v3-large NLI (MNLI/FEVER/ANLI/LingNLI/WANLI, model server `/nli`) and scores `mean_j max_w P(e) − P(c)`. This is SummaC-ZS (Laban et al., TACL 2022), not a new metric. It answers the reviewers' point that cosine is not entailment.

- **Consistency is its strength.** Raw ρ .438, partial .362 on the full set and .413 on the test half. That is the best consistency signal in the pool, above UniEval (.300) and G-Eval (.395 partial).
- **The other dimensions are weak.** Mean partial is .189.
- **Cost:** 467 ms/sample on GB10.

## SPL — splice-point likelihood (negative result)

The median SummEval candidate copies 92% of its bigrams, so the hypothesis was that quality is decided at the splice points between copied fragments. `cmd/spl` scores `mean log P(w | x, prefix)` only over words that are not a continuation of a source bigram. It is cheap: 45 ms/sample, one source pass, no perturbations.

**It does not work.** Canonical raw .123 and partial .139. The words *inside* copied fragments carry more signal (partial .197). The per-word log-probs sum back to CCM's sequence log-prob within 0.03 nats, so this is not an implementation error.

## LNC — Likelihood–NLI Composite

`cmd/lnc` computes 10 cheap, reference-free features end to end. It then applies a per-dimension ridge combination (α = 10, fixed a priori). The calibration is fitted on the dev half only and frozen in `ablation/lnc_calibration.json`.

| source | features |
|---|---|
| CCM | `log P(y|x)/|y|`, `log P(y|x)`, `log P(y)/|y|`, order margin, negation margin, z-margin |
| NLIG | mean and min |
| SPL | splice and inside-copy log-prob |

**How the design was chosen:**

- Ridge was preferred over picking the best single signal or pair per dimension by 5-fold article-level cross-validation inside dev: .301 vs .295.
- The test half was not used for that choice.
- A population-dependent rank feature was dropped so the metric is computable per sample.

```sh
make benchmark-lnc                 # fit on first 50 articles, score all 1600
make benchmark-lnc LNC_FIT=false   # apply the frozen calibration
make paper-lnc                     # paired bootstrap on the held-out half + confound table
```

**Held-out test half** (last 50 articles, never used for fitting). Mean summary-level Spearman ρ; UniEval and G-Eval use their matching-dimension scorer:

| metric | raw ρ̄ | partial ρ̄ | partial coh / con / flu / rel |
|---|---|---|---|
| UniEval | .473 | .424 | .550 / .289 / .426 / .429 |
| **LNC** | **.437** | **.354** | .375 / .360 / .333 / .347 |
| G-Eval (7B, 3 runs) | .338 | .300 | .313 / .398 / .177 / .313 |
| CCM-D | .393 | .297 | .319 / .237 / .315 / .317 |
| BERTScore | .269 | .269 | .425 / .105 / .128 / .420 |
| LGS | .319 | .230 | .380 / .118 / .045 / .377 |
| Lead-5 | .292 | .147 | .190 / .135 / .062 / .202 |

**Paired bootstrap on the test half only** (`cmd/compare -doc-split last50`, BH-FDR over 32 cells): 18 significant wins, 13 ties, 1 loss (UniEval on coherence).

- **vs UniEval:** wins on consistency (+.104).
- **vs G-Eval:** wins on fluency (+.275), ties on coherence, consistency and relevance.
- **vs other metrics:** beats CCM-D on coherence and consistency, and beats LGS, BERTScore and GPTScore on consistency and fluency.
- **vs Lead-5 and BARTScore:** wins on every dimension.

**Split robustness.** The same fit was repeated on 200 random 50/50 article splits:

- LNC: partial .331 [.302, .356], raw .420 [.392, .445].
- It beats G-Eval on raw in 100% of splits and on partial in 71%.
- It beats BERTScore in 100% and UniEval in 0%.
- The canonical-split test number (.354) is on the favourable side of this distribution. **.331 is the honest headline.**

**LNC-D (debiased variant, `make benchmark-lncd`).** Same features and α. The ridge target is the human rank with the copy-rate rank projected out; copy rate is used only for fitting, never as a feature. The feature dump is reused, so no model calls are needed.

| test half | raw ρ̄ | partial ρ̄ | ρ_copy |
|---|---|---|---|
| LNC | .437 | .354 | .55 |
| LNC-D | .398 | .354 | .25 |

LNC-D matches LNC on the extractiveness-controlled axis while depending far less on copying. On the raw axis it gives up ground: paired comparisons on the test half give 7 wins, 14 ties, 3 losses. The losses are to UniEval on coherence and to LNC on consistency and relevance.

**Open issues:**

- **Still extractiveness-correlated.** ρ_copy is .64 on the full set, from the likelihood features. LNC-D reduces this.
- **Cost:** 854 ms/sample on GB10 (CCM 340 + NLI 469 + SPL 45). This is not comparable to the Apple-CPU baseline costs until those are re-measured on the same hardware (`gb10/`).
- **Single corpus.** The calibration is fitted on SummEval.
- **No new primitive.** Prior art: MetaMetrics (learned metric combination), SummaC, BARTScore/FFLM, DeltaScore.

## LNC-fast — the same composite at a quarter of the cost

`make benchmark-lncf` runs the LNC pipeline and protocol unchanged. Three cost cuts were fixed a priori, not tuned:

- The CCM LM runs in bf16 (`CCM_DTYPE=bfloat16` on the model server).
- K=4 perturbations instead of 8.
- NLI is run only against each summary sentence's 4 source windows with the highest word overlap (`-nli-topk 4`).

The calibration is refitted on the dev half only.

| | GB10 ms/sample | test raw ρ̄ | test partial ρ̄ | 200 random splits partial ρ̄ |
|---|---|---|---|---|
| LNC | 854 (CCM 340 + NLI 469 + SPL 45) | .437 | .354 | .331 |
| **LNC-fast** | **227** (CCM 118 + NLI 85 + SPL 25) | **.433** | **.350** | **.323** [.298, .347] |

**Paired bootstrap on the test half** (36 cells, BH-FDR): 17 wins, 18 ties, 1 loss (UniEval on coherence).

- **vs LNC:** ties on every dimension.
- **vs UniEval:** wins on consistency (+.111).
- **vs G-Eval:** wins on fluency (+.279), ties elsewhere.

**Feature-group ablation** (ridge refit on dev, mean over 100 random splits):

- The perturbation features (order margin, negation margin, z-margin) matter most. Without them partial ρ̄ falls from .331 to .278. The perturbation idea failed as a metric on its own (CCM) but is the most valuable ingredient here.
- NLI adds about .019.
- SPL adds about .003.

**Pareto frontier on one machine** (`cmd/confound -cost-dir gb10`, full set; `paper/lncf_confound_gb10.gen.tex`):

- The frontier is Lead-k, METEOR, GPTScore, LGS, BERTScore, LNC-fast and UniEval on the partial axis. On the raw axis it is Lead-k, LNC-fast and UniEval.
- LNC-fast is the only reference-free metric on the frontier above .2 partial.
- It sits between BERTScore (210 ms, .241) and UniEval (336 ms, .424) with .339 on the full set.

Caveats:

- SMART-Model and EmbedScorer (Ollama) were not re-timed. Both are dominated on quality regardless.
- The lexical metrics keep their CPU times.

## Cost on one machine (GB10)

The per-sample costs in `output/*.json` for the published baselines are the original Apple-CPU runs, while CCM, CCM-D, PSC, NLIG, SPL and LNC were measured on an ASUS Ascent GX10 (NVIDIA GB10). To compare like with like, the baselines were re-timed on the GB10 (`gb10/*.json`; scores in `output/` are untouched):

- **Warm-up.** Each metric ran after a warm-up call, so model loading is not charged.
- **Samples.** All 1600 samples, except G-Eval, which ran on 48 samples per dimension in the canonical configuration (3 runs, T=0.7).
- **Quality.** Partial ρ̄ is from `cmd/confound` on the full set.

| metric | GB10 ms/sample | Apple-CPU ms/sample (old) | partial ρ̄ | reference-free |
|---|---|---|---|---|
| GPTScore | 31 | 91 | .165 | no |
| LGS | 93 | 57 | .198 | yes |
| BERTScore | 210 | 568 | .241 | no |
| BARTScore | 302 | 523 | .136 | no |
| CCM-D | 335 | — | .290 | yes |
| UniEval (4 dims) | 336 | 430 | .424 | no (relevance uses the reference) |
| MoverScore | 404 | 1277 | .091 | no |
| LNC-fast | 227 | — | .339 | yes |
| LNC | 854 | — | .344 | yes |
| G-Eval (4 dims × 3 runs) | 18 219 | 13 594 | .318 | yes |

**What this does to the claims.** On one machine UniEval is both cheaper than full LNC and better. LNC-fast (227 ms) is cheaper than UniEval and sits on the cost–quality frontier on both axes. Among reference-free metrics it is ~80× cheaper than G-Eval and ties or beats it on every dimension of the held-out half. Wherever a reference exists and quality matters more than a third of the cost, UniEval remains the better choice.

## Reproducing the paper

Every number in `paper/*.gen.tex` regenerates from Make targets. Steps below assume Ollama on `localhost:11434` with `nomic-embed-text` pulled and the model server (`cmd/modelsrv`) running on port 9200 for the reference-based baselines.

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

# 5. (Optional) Statistics-only targets. These read the per-sample
#    scores already stored in output/ and ablation/, so they need
#    neither Ollama nor the model server:
make paper-lambdaci                 # λ selection uncertainty (~3 min)
make paper-comparisons              # paired bootstrap + BH-FDR (~2 min)
make paper-confound                 # extractiveness confound (~2 min)
make benchmark-lead                 # lead-window controls (~1 min)
make paper-friedman                 # Friedman + Wilcoxon over the full pool (~1 min)
make paper-splitrobust              # λ* random-split robustness (~1 min)
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
| CCM (ours)   | LM contrast  | model server  | source           | `cmd/ccm`          |

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
make benchmark-modelsrv   # BERTScore, MoverScore, BARTScore, GPTScore, UniEval (4 dims), CCM
make benchmark-ollama     # EmbedScorer, SMART-Model, G-Eval (4 dims)
```

Each cmd writes a JSON report to `output/<metric>.json` containing per-sample scores plus summary-level and system-level Pearson, Spearman, and Kendall correlations against the SummEval human annotations (with bootstrap 95% CI by default).

## Generate paper tables

```sh
make paper
```

Writes the correlation tables from the JSON reports in `output/`. Each correlation table carries a bootstrap CI in every cell, so Spearman and Kendall are emitted as **separate** tables — side by side they are about twice as wide as the `elsarticle` text block:

| File | Contents | LaTeX label |
|------|----------|-------------|
| `paper/summary.gen.tex`         | summary-level Spearman ρ | `tab:correlations_summary` |
| `paper/summary_kendall.gen.tex` | summary-level Kendall τ  | `tab:correlations_summary_kendall` |
| `paper/system.gen.tex`          | system-level Spearman ρ  | `tab:correlations_system` |
| `paper/system_kendall.gen.tex`  | system-level Kendall τ   | `tab:correlations_system_kendall` |

`cmd/paper` takes `-coeffs` (`spearman`, `kendall`, `pearson`; comma-separated) and `-label` to override the default label.

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

### CCM

Source-conditioned and unconditional sequence log-likelihoods under a small causal LM (`CCM_MODEL`, default `Qwen/Qwen2.5-1.5B`). The source is encoded once per request and its KV cache reused across all candidates.

```sh
curl -X POST http://localhost:9200/ccm \
     -H "Content-Type: application/json" \
     -d '{"source": "The mayor announced on Monday that the city will build 3 new parks.", "candidates": ["The city will build 3 new parks.", "The city will build 5 new parks."]}'
```

### GPTScore

GPT-2 log-probability scoring normalized to [0, 1].

```sh
curl -X POST http://localhost:9200/gptscore \
     -H "Content-Type: application/json" \
     -d '{"reference": "A Pod is the smallest deployable unit in Kubernetes.", "candidate": "A Pod is the smallest unit of deployment in Kubernetes."}'
```

[MIT](LICENSE)
