"""
Model Server — BERTScore, MoverScore, UniEval, GPTScore, BARTScore.

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
