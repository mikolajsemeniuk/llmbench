"""
Model Server — BERTScore, MoverScore, UniEval, GPTScore, BARTScore, CCM.

All models are loaded EAGERLY at startup. The server only starts accepting
requests after all models are in memory. Set EAGER_LOAD=0 to revert to
lazy loading (load on first request) for faster development startup.

Usage:
    python3 app.py

Endpoints:
    POST /bertscore      — canonical token-level BERTScore (F1)
    POST /moverscore     — Word Mover's Distance with contextual embeddings
    POST /unieval        — T5-based Boolean QA evaluator (canonical-style prompts)
    POST /gptscore       — generative log-probability scoring (GPT-2)
    POST /bartscore      — canonical BARTScore via facebook/bart-large-cnn
    POST /ccm            — source-conditioned sequence log-likelihoods (small causal LM)
    POST /tokenlogprobs  — per-token log-likelihoods + char offsets (SPL)
    POST /nli            — premise x hypothesis entailment matrix (DeBERTa-v3 NLI)
    POST /alignscore     — AlignScore-large, nli_sp mode (weights from alignscore_convert.py)
    GET  /health         — health check + loaded models
"""

import logging
import math
import os
import time

import numpy as np
import torch
from flask import Flask, jsonify, request

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Suppress Flask's per-request access logs to keep startup output clean.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

_cache = {}

BERTSCORE_MODEL = os.environ.get("BERTSCORE_MODEL", "roberta-large")


def _detect_device():
    override = os.environ.get("METRICS_DEVICE", "")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _detect_device()


# ── BERTScore ──────────────────────────────────────────────────────────


def get_bertscore_scorer():
    if "bertscore" not in _cache:
        logger.info(f"Loading BERTScore with {BERTSCORE_MODEL}...")
        from bert_score import BERTScorer

        _cache["bertscore"] = BERTScorer(
            model_type=BERTSCORE_MODEL,
            lang="en",
            device=DEVICE,
        )
        logger.info("BERTScore loaded.")
    return _cache["bertscore"]


@app.route("/bertscore", methods=["POST"])
def bertscore():
    data = request.json
    ref = data.get("reference", "")
    cand = data.get("candidate", "")
    if not ref or not cand:
        return jsonify({"error": "reference and candidate required"}), 400

    scorer = get_bertscore_scorer()
    P, R, F1 = scorer.score([cand], [ref])
    return jsonify(
        {
            "score": round(F1[0].item(), 6),
            "precision": round(P[0].item(), 6),
            "recall": round(R[0].item(), 6),
        }
    )


# ── MoverScore ─────────────────────────────────────────────────────────


def get_mover_model():
    if "mover_model" not in _cache:
        logger.info("Loading model for MoverScore...")
        from transformers import AutoModel, AutoTokenizer

        name = BERTSCORE_MODEL
        _cache["mover_tokenizer"] = AutoTokenizer.from_pretrained(name)
        _cache["mover_model"] = AutoModel.from_pretrained(
            name, use_safetensors=True
        ).to(DEVICE)
        _cache["mover_model"].eval()
        logger.info("MoverScore model loaded.")
    return _cache["mover_tokenizer"], _cache["mover_model"]


def _get_token_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        DEVICE
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeds = outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()
    return embeds


def _moverscore(ref_embeds, cand_embeds):
    import ot

    n_ref = ref_embeds.shape[0]
    n_cand = cand_embeds.shape[0]

    if n_ref == 0 or n_cand == 0:
        return 0.0

    w_ref = np.ones(n_ref, dtype=np.float64) / n_ref
    w_cand = np.ones(n_cand, dtype=np.float64) / n_cand

    ref_norm = ref_embeds / (np.linalg.norm(ref_embeds, axis=1, keepdims=True) + 1e-9)
    cand_norm = cand_embeds / (
        np.linalg.norm(cand_embeds, axis=1, keepdims=True) + 1e-9
    )
    cost = 1.0 - np.dot(ref_norm, cand_norm.T)
    cost = np.maximum(cost, 0.0).astype(np.float64)

    emd = ot.emd2(w_ref, w_cand, cost)
    return float(max(0.0, 1.0 - emd))


@app.route("/moverscore", methods=["POST"])
def moverscore():
    data = request.json
    ref = data.get("reference", "")
    cand = data.get("candidate", "")
    if not ref or not cand:
        return jsonify({"error": "reference and candidate required"}), 400

    tokenizer, model = get_mover_model()
    ref_emb = _get_token_embeddings(ref, tokenizer, model)
    cand_emb = _get_token_embeddings(cand, tokenizer, model)
    score = _moverscore(ref_emb, cand_emb)
    return jsonify({"score": round(score, 6)})


# ── UniEval ────────────────────────────────────────────────────────────


def get_unieval_model():
    if "unieval_model" not in _cache:
        logger.info("Loading UniEval model...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = "MingZhong/unieval-sum"
        _cache["unieval_tokenizer"] = AutoTokenizer.from_pretrained(name)
        _cache["unieval_model"] = AutoModelForSeq2SeqLM.from_pretrained(
            name, use_safetensors=True
        ).to(DEVICE)
        _cache["unieval_model"].eval()
        logger.info("UniEval loaded.")
    return _cache["unieval_tokenizer"], _cache["unieval_model"]


def _build_unieval_prompt(dimension, candidate, reference, source):
    """
    Builds canonical-style UniEval prompts following Zhong et al. 2022.

    Coherence/consistency: condition on source document (the news article).
    Fluency: candidate only (grammar/style judgment).
    Relevance: condition on reference summary (alignment with gold summary).
    """
    if dimension == "coherence":
        return (
            f"question: Is this a coherent summary to the document? </s> "
            f"summary: {candidate} </s> "
            f"document: {source}"
        )
    if dimension == "consistency":
        return (
            f"question: Is this claim consistent with the document? </s> "
            f"claim: {candidate} </s> "
            f"document: {source}"
        )
    if dimension == "fluency":
        return f"question: Is this a fluent paragraph? </s> paragraph: {candidate}"
    if dimension == "relevance":
        return (
            f"question: Is this summary relevant to the reference? </s> "
            f"summary: {candidate} </s> "
            f"reference: {reference}"
        )
    # Fallback: same as coherence with source.
    return (
        f"question: Is this a good summary of the document? </s> "
        f"summary: {candidate} </s> "
        f"document: {source}"
    )


def _unieval_score(prompt, tokenizer, model):
    """Returns P(Yes) given the prompt — one model forward pass."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(DEVICE)

    with torch.no_grad():
        yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
        no_id = tokenizer("No", add_special_tokens=False).input_ids[0]

        decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(DEVICE)
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[0, 0, :]

        # Softmax over yes/no only — closer to canonical Boolean QA setup.
        probs = torch.softmax(logits[[yes_id, no_id]], dim=0)
        return probs[0].item()


@app.route("/unieval", methods=["POST"])
def unieval():
    data = request.json
    ref = data.get("reference", "")
    cand = data.get("candidate", "")
    source = data.get("source", "")
    dimension = data.get("dimension", "overall")

    if not cand:
        return jsonify({"error": "candidate required"}), 400

    tokenizer, model = get_unieval_model()

    if dimension == "all":
        scores = {}
        for dim in ["coherence", "consistency", "fluency", "relevance"]:
            prompt = _build_unieval_prompt(dim, cand, ref, source)
            scores[dim] = round(_unieval_score(prompt, tokenizer, model), 6)
        scores["score"] = round(
            float(np.mean([v for k, v in scores.items() if k != "score"])), 6
        )
        return jsonify(scores)

    prompt = _build_unieval_prompt(dimension, cand, ref, source)
    score = _unieval_score(prompt, tokenizer, model)
    return jsonify({"score": round(score, 6), "dimension": dimension})


# ── GPTScore ───────────────────────────────────────────────────────────


def get_gptscore_model():
    if "gptscore_model" not in _cache:
        logger.info("Loading GPT-2 for GPTScore...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        _cache["gptscore_tokenizer"] = GPT2Tokenizer.from_pretrained("gpt2")
        _cache["gptscore_model"] = GPT2LMHeadModel.from_pretrained(
            "gpt2", use_safetensors=True
        ).to(DEVICE)
        _cache["gptscore_model"].eval()
        logger.info("GPTScore model loaded.")
    return _cache["gptscore_tokenizer"], _cache["gptscore_model"]


def _gptscore(reference, candidate, tokenizer, model):
    """
    GPTScore: average conditional log-probability of candidate tokens
    given reference as context.

    Score = sigmoid( (1/m) * Σ log P(cand_i | cand_<i, reference) + 3 )
    Returns a score in [0, 1].
    """
    prompt = f"Reference: {reference}\nResponse: {candidate}"

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(DEVICE)
    input_ids = inputs["input_ids"]

    ref_prompt = f"Reference: {reference}\nResponse: "
    ref_ids = tokenizer(
        ref_prompt, return_tensors="pt", truncation=True, max_length=1024
    )["input_ids"]
    cand_start = ref_ids.shape[1]

    if cand_start >= input_ids.shape[1]:
        return 0.0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        total_log_prob = 0.0
        n_tokens = 0

        for i in range(cand_start, input_ids.shape[1]):
            token_id = input_ids[0, i].item()
            if i > 0:
                token_log_prob = log_probs[0, i - 1, token_id].item()
                total_log_prob += token_log_prob
                n_tokens += 1

    if n_tokens == 0:
        return 0.0

    avg_log_prob = total_log_prob / n_tokens
    score = 1.0 / (1.0 + math.exp(-(avg_log_prob + 3)))
    return score


@app.route("/gptscore", methods=["POST"])
def gptscore():
    data = request.json
    ref = data.get("reference", "")
    cand = data.get("candidate", "")
    if not ref or not cand:
        return jsonify({"error": "reference and candidate required"}), 400

    tokenizer, model = get_gptscore_model()
    score = _gptscore(ref, cand, tokenizer, model)
    return jsonify({"score": round(score, 6)})


# ── BARTScore ──────────────────────────────────────────────────────────


def get_bartscore_model():
    if "bartscore_model" not in _cache:
        logger.info("Loading BART-large-cnn for BARTScore...")
        from transformers import BartForConditionalGeneration, BartTokenizer

        name = "facebook/bart-large-cnn"
        _cache["bartscore_tokenizer"] = BartTokenizer.from_pretrained(name)
        _cache["bartscore_model"] = BartForConditionalGeneration.from_pretrained(
            name, use_safetensors=True
        ).to(DEVICE)
        _cache["bartscore_model"].eval()
        logger.info("BARTScore model loaded.")
    return _cache["bartscore_tokenizer"], _cache["bartscore_model"]


def _bartscore(reference, candidate, tokenizer, model):
    """
    Canonical BARTScore (Yuan et al. 2021):
    log P(reference | candidate) averaged per token.

    Higher (less negative) = candidate better predicts the reference.
    Typical range: [-10, 0].
    """
    src_inputs = tokenizer(
        candidate, return_tensors="pt", truncation=True, max_length=1024
    ).to(DEVICE)
    tgt_inputs = tokenizer(
        reference, return_tensors="pt", truncation=True, max_length=1024
    ).to(DEVICE)

    src_ids = src_inputs["input_ids"]
    src_mask = src_inputs["attention_mask"]
    tgt_ids = tgt_inputs["input_ids"]

    if tgt_ids.shape[1] <= 1:
        return 0.0

    decoder_input_ids = tgt_ids[:, :-1]
    labels = tgt_ids[:, 1:]

    with torch.no_grad():
        outputs = model(
            input_ids=src_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits  # (1, tgt_len-1, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

        gathered = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        avg_log_prob = gathered.mean().item()

    return float(avg_log_prob)


@app.route("/bartscore", methods=["POST"])
def bartscore():
    data = request.json
    ref = data.get("reference", "")
    cand = data.get("candidate", "")
    if not ref or not cand:
        return jsonify({"error": "reference and candidate required"}), 400

    tokenizer, model = get_bartscore_model()
    score = _bartscore(ref, cand, tokenizer, model)
    return jsonify({"score": round(score, 6)})


# ── CCM (copy-invariant contrastive margin) ────────────────────────────
#
# CCM needs only raw sequence log-likelihoods from a small causal LM;
# the perturbations and the margin itself are computed in Go
# (pkg/metrics/ccm.go). The source is encoded ONCE per request and its
# KV cache is reused for every continuation, so a request carrying all
# 16 candidates x (K+1) variants of one article costs one source pass.

CCM_MODEL = os.environ.get("CCM_MODEL", "Qwen/Qwen2.5-1.5B")
CCM_BATCH = int(os.environ.get("CCM_BATCH", "48"))
CCM_HEAD_ROWS = int(os.environ.get("CCM_HEAD_ROWS", "4"))
CCM_MAX_SOURCE_TOKENS = int(os.environ.get("CCM_MAX_SOURCE_TOKENS", "3072"))
# float32 weights by default: in bf16 the batched, padded continuation
# pass drifts ~0.1 nats from a plain full forward, which is the same
# order as small margins. Matmuls run in TF32 inside the CCM call only
# (drift ~0.003 nats, ~3x faster on Ampere+). CCM_DTYPE=bfloat16 trades
# precision for another ~2x.
CCM_DTYPE = os.environ.get("CCM_DTYPE", "float32")
CCM_PREFIX = "Article:\n{source}\n\nSummary:\n"
CCM_UNCOND_PREFIX = "Summary:\n"


def get_ccm_model():
    if "ccm_model" not in _cache:
        logger.info(f"Loading {CCM_MODEL} for CCM...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _cache["ccm_tokenizer"] = AutoTokenizer.from_pretrained(CCM_MODEL)
        _cache["ccm_model"] = AutoModelForCausalLM.from_pretrained(
            CCM_MODEL, dtype=getattr(torch, CCM_DTYPE), use_safetensors=True
        ).to(DEVICE)
        _cache["ccm_model"].eval()
        logger.info("CCM model loaded.")
    return _cache["ccm_tokenizer"], _cache["ccm_model"]


def _ccm_continuation_logprobs(prefix, continuations, tokenizer, model, per_token=False):
    """
    Sum of log P(continuation tokens | prefix) for every continuation.
    The prefix is run once; its KV cache is repeated across each batch
    of right-padded continuations. Returns (sums, token_counts), or
    (per-token log-prob lists, token_counts) when per_token is set.
    """
    from transformers import DynamicCache

    pre_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ]
    if pre_ids.shape[1] > CCM_MAX_SOURCE_TOKENS:
        pre_ids = pre_ids[:, -CCM_MAX_SOURCE_TOKENS:]
    pre_ids = pre_ids.to(DEVICE)
    P = pre_ids.shape[1]

    with torch.no_grad():
        pre = model(pre_ids, use_cache=True, logits_to_keep=1)
    first_logp = torch.log_softmax(pre.logits[:, -1].float(), dim=-1)  # (1, V)
    legacy = pre.past_key_values
    if hasattr(legacy, "to_legacy_cache"):
        legacy = legacy.to_legacy_cache()

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    enc = [
        tokenizer(c, add_special_tokens=False)["input_ids"] or [pad_id]
        for c in continuations
    ]
    sums, counts = [], []
    for b in range(0, len(enc), CCM_BATCH):
        chunk = enc[b : b + CCM_BATCH]
        B, L = len(chunk), max(len(x) for x in chunk)
        ids = torch.full((B, L), pad_id, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i, x in enumerate(chunk):
            ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)

        cache = DynamicCache.from_legacy_cache(
            tuple((k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for k, v in legacy)
        )
        full_mask = torch.cat(
            [torch.ones((B, P), dtype=torch.long, device=DEVICE), mask], dim=1
        )
        pos = (torch.arange(L, device=DEVICE) + P).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            hidden = model.model(
                ids,
                attention_mask=full_mask,
                past_key_values=cache,
                position_ids=pos,
                use_cache=True,
            ).last_hidden_state[:, :-1]
            # log-softmax only at the target tokens, projecting through
            # the LM head a few rows at a time: a full (B, L, V) logits
            # tensor over the ~150k-token vocabulary runs to several GB.
            tgt_ids = ids[:, 1:]
            parts = []
            for r in range(0, B, CCM_HEAD_ROWS):
                logits = model.lm_head(hidden[r : r + CCM_HEAD_ROWS]).float()
                tgt = logits.gather(2, tgt_ids[r : r + CCM_HEAD_ROWS].unsqueeze(-1))
                parts.append(tgt.squeeze(-1) - torch.logsumexp(logits, dim=-1))
                del logits
            rest = torch.cat(parts) * mask[:, 1:]
            del hidden, cache

        # token 0 is predicted by the prefix's last position, token t>0
        # by continuation position t-1.
        tok0 = first_logp[0].gather(0, ids[:, 0])  # (B,)
        total = tok0 + rest.sum(dim=1)
        for i in range(B):
            n_i = int(mask[i].sum().item())
            if per_token:
                row = [float(tok0[i].item())] + rest[i, : n_i - 1].tolist()
                sums.append(row)
            else:
                sums.append(float(total[i].item()))
            counts.append(n_i)
    return sums, counts


def _ccm_tf32(fn, *args):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return fn(*args)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@app.route("/ccm", methods=["POST"])
def ccm():
    """
    Request:  {"source": str, "candidates": [str, ...]}
    Response: {"cond": [float], "uncond": [float], "tokens": [int]}

    cond[i]   = log P(candidates[i] | "Article: source  Summary:")
    uncond[i] = log P(candidates[i] | "Summary:")   (no source)
    """
    data = request.json
    src = data.get("source", "")
    cands = data.get("candidates") or []
    if not src or not cands:
        return jsonify({"error": "source and candidates required"}), 400

    tokenizer, model = get_ccm_model()
    cond, tokens = _ccm_tf32(
        _ccm_continuation_logprobs,
        CCM_PREFIX.format(source=src),
        cands,
        tokenizer,
        model,
    )
    uncond, _ = _ccm_tf32(
        _ccm_continuation_logprobs, CCM_UNCOND_PREFIX, cands, tokenizer, model
    )
    return jsonify({"cond": cond, "uncond": uncond, "tokens": tokens, "model": CCM_MODEL})


@app.route("/tokenlogprobs", methods=["POST"])
def tokenlogprobs():
    """
    Per-token log-likelihoods of each candidate under the CCM LM, with
    and without the source, plus each token's character span in the
    candidate string (used by SPL to locate copy splice points).

    Request:  {"source": str, "candidates": [str]}
    Response: {"cond": [[float]], "uncond": [[float]], "offsets": [[[s, e]]]}
    """
    data = request.json
    src = data.get("source", "")
    cands = data.get("candidates") or []
    if not src or not cands:
        return jsonify({"error": "source and candidates required"}), 400

    tokenizer, model = get_ccm_model()
    cond, _ = _ccm_tf32(
        _ccm_continuation_logprobs,
        CCM_PREFIX.format(source=src),
        cands,
        tokenizer,
        model,
        True,
    )
    uncond, _ = _ccm_tf32(
        _ccm_continuation_logprobs, CCM_UNCOND_PREFIX, cands, tokenizer, model, True
    )
    offsets = [
        tokenizer(c, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        for c in cands
    ]
    return jsonify({"cond": cond, "uncond": uncond, "offsets": offsets})


# ── NLI grounding ──────────────────────────────────────────────────────
#
# Premise x hypothesis entailment matrix for one article: every source
# window against every summary sentence of all its candidates, in one
# request. Returns P(entailment), P(neutral), P(contradiction) per pair.

NLI_MODEL = os.environ.get(
    "NLI_MODEL", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)
NLI_BATCH = int(os.environ.get("NLI_BATCH", "128"))


def get_nli_model():
    if "nli_model" not in _cache:
        logger.info(f"Loading {NLI_MODEL} for NLI grounding...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        _cache["nli_tokenizer"] = AutoTokenizer.from_pretrained(NLI_MODEL)
        _cache["nli_model"] = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL, dtype=dtype, use_safetensors=True
        ).to(DEVICE)
        _cache["nli_model"].eval()
        labels = {v.lower(): int(k) for k, v in _cache["nli_model"].config.id2label.items()}
        _cache["nli_labels"] = [labels["entailment"], labels["neutral"], labels["contradiction"]]
        logger.info("NLI model loaded.")
    return _cache["nli_tokenizer"], _cache["nli_model"], _cache["nli_labels"]


@app.route("/nli", methods=["POST"])
def nli():
    """
    Request:  {"premises": [str], "hypotheses": [str]}
    Response: {"probs": [[[e, n, c] for each premise] for each hypothesis]}

    or, sparse: {"pairs": [[premise, hypothesis], ...]} → {"flat": [[e, n, c], ...]}
    """
    data = request.json
    tokenizer, model, order = get_nli_model()

    # Sparse mode: explicit (premise, hypothesis) pairs, flat response.
    if data.get("pairs"):
        pairs = [(p, h) for p, h in data["pairs"]]
        return jsonify({"flat": _nli_probs(pairs, tokenizer, model, order)})

    prem = data.get("premises") or []
    hyp = data.get("hypotheses") or []
    if not prem or not hyp:
        return jsonify({"error": "premises and hypotheses (or pairs) required"}), 400

    pairs = [(p, h) for h in hyp for p in prem]
    out = _nli_probs(pairs, tokenizer, model, order)
    P = len(prem)
    return jsonify({"probs": [out[i * P : (i + 1) * P] for i in range(len(hyp))]})


def _nli_probs(pairs, tokenizer, model, order):
    out = []
    for b in range(0, len(pairs), NLI_BATCH):
        chunk = pairs[b : b + NLI_BATCH]
        enc = tokenizer(
            [p for p, _ in chunk],
            [h for _, h in chunk],
            return_tensors="pt",
            padding=True,
            truncation="only_first",
            max_length=320,
        ).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits.float(), dim=-1)[:, order]
        out.extend(probs.cpu().tolist())
    return out


# ── AlignScore ─────────────────────────────────────────────────────────
#
# AlignScore-large (Zha et al., ACL 2023), "nli_sp" mode as in the
# official inference code: the source is cut into chunks of ~350 words
# (whole sentences), every summary sentence is scored against every
# chunk with the 3-way head, P(aligned) is maxed over chunks and averaged
# over summary sentences. Sentence splitting is done by the caller
# (pkg/metrics SplitSentences) instead of NLTK. Weights come from
# alignscore_convert.py (safetensors, never unpickled).

ALIGNSCORE_WEIGHTS = os.environ.get(
    "ALIGNSCORE_WEIGHTS",
    os.path.expanduser("~/.cache/llmbench/alignscore-large.safetensors"),
)
ALIGNSCORE_BATCH = int(os.environ.get("ALIGNSCORE_BATCH", "32"))


def get_alignscore_model():
    if "alignscore_model" not in _cache:
        logger.info("Loading AlignScore-large...")
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, RobertaConfig, RobertaModel

        state = load_file(ALIGNSCORE_WEIGHTS)
        enc = RobertaModel(RobertaConfig.from_pretrained("roberta-large"))
        missing, unexpected = enc.load_state_dict(
            {k[len("base_model.") :]: v for k, v in state.items() if k.startswith("base_model.")},
            strict=False,
        )
        # position_ids is a non-persistent buffer in current transformers.
        unexpected = [k for k in unexpected if not k.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"AlignScore weights: missing {missing}, unexpected {unexpected}")
        head = torch.nn.Linear(enc.config.hidden_size, 3)
        head.load_state_dict({"weight": state["tri_layer.weight"], "bias": state["tri_layer.bias"]})
        _cache["alignscore_tokenizer"] = AutoTokenizer.from_pretrained("roberta-large")
        _cache["alignscore_model"] = (enc.to(DEVICE).eval(), head.to(DEVICE).eval())
        logger.info("AlignScore loaded.")
    return _cache["alignscore_tokenizer"], _cache["alignscore_model"]


@app.route("/alignscore", methods=["POST"])
def alignscore():
    """
    Request:  {"source_sents": [str], "candidates": [[str, ...], ...]}
              (source and every candidate already split into sentences)
    Response: {"scores": [float]}   one AlignScore per candidate
    """
    data = request.json
    src = data.get("source_sents") or []
    cands = data.get("candidates") or []
    if not src or not cands:
        return jsonify({"error": "source_sents and candidates required"}), 400

    tokenizer, (enc, head) = get_alignscore_model()
    n_chunk = len(" ".join(src).split()) // 350 + 1
    per = max(len(src) // n_chunk, 1)
    chunks = [" ".join(src[i : i + per]) for i in range(0, len(src), per)]

    pairs = [(c, h) for sents in cands for h in (sents or [""]) for c in chunks]
    probs = []
    with torch.no_grad():
        for b in range(0, len(pairs), ALIGNSCORE_BATCH):
            chunk = pairs[b : b + ALIGNSCORE_BATCH]
            x = tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                return_tensors="pt",
                padding=True,
                truncation="only_first",
                max_length=512,
            ).to(DEVICE)
            logits = head(enc(**x).pooler_output)
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 0].cpu().tolist())

    scores, j = [], 0
    for sents in cands:
        n = max(len(sents), 1)
        m = np.array(probs[j : j + n * len(chunks)]).reshape(n, len(chunks))
        scores.append(float(m.max(axis=1).mean()))
        j += n * len(chunks)
    return jsonify({"scores": scores})


# ── Health ─────────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    loaded = sorted(
        set(k.replace("_tokenizer", "").replace("_model", "") for k in _cache.keys())
    )
    return jsonify(
        {
            "status": "ok",
            "device": DEVICE,
            "loaded_models": loaded,
            "available": [
                "bertscore",
                "moverscore",
                "unieval",
                "gptscore",
                "bartscore",
                "ccm",
                "nli",
                "tokenlogprobs",
            ],
        }
    )


# ── Eager loading ──────────────────────────────────────────────────────


def warmup():
    if os.environ.get("EAGER_LOAD", "1") != "1":
        logger.info(
            "Eager loading disabled (EAGER_LOAD=0); models will load on first request."
        )
        return

    logger.info("Eager loading all models — this may take a few minutes...")
    start = time.time()

    loaders = [
        ("BERTScore", get_bertscore_scorer),
        ("MoverScore", get_mover_model),
        ("UniEval", get_unieval_model),
        ("GPTScore", get_gptscore_model),
        ("BARTScore", get_bartscore_model),
        ("CCM", get_ccm_model),
        ("NLI", get_nli_model),
    ]

    for name, fn in loaders:
        t0 = time.time()
        try:
            fn()
            logger.info(f"  {name} ready ({time.time() - t0:.1f}s)")
        except Exception as e:
            logger.error(f"  {name} FAILED to load: {e}")
            logger.error(f"  /{name.lower()} endpoint will return 500 until fixed.")

    logger.info(f"All models loaded in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9200))
    logger.info(f"Model server starting on :{port} (device: {DEVICE})")

    warmup()

    logger.info("=" * 60)
    logger.info(f"READY — accepting requests on :{port}")
    logger.info("=" * 60)

    app.run(host="0.0.0.0", port=port, use_reloader=False)
