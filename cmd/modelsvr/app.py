"""
Model Server — BERTScore, MoverScore, UniEval, GPTScore.

All models are PyTorch-based. Loaded lazily on first request per metric.
Shares RoBERTa weights between BERTScore and MoverScore.

Usage:
    docker build -t modelserver .
    docker run -p 9200:9200 modelserver

Endpoints:
    POST /bertscore      — canonical token-level BERTScore (F1)
    POST /moverscore     — Word Mover's Distance with contextual embeddings
    POST /unieval        — T5-based multi-dimensional Boolean QA evaluator
    POST /gptscore       — generative log-probability scoring (GPT-2)
    GET  /health         — health check + loaded models
"""

import logging
import math
import os

import numpy as np
import torch
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
        _cache["mover_model"] = AutoModel.from_pretrained(name).to(DEVICE)
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
    if "unieval" not in _cache:
        logger.info("Loading UniEval model...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = "MingZhong/unieval-sum"
        _cache["unieval_tokenizer"] = AutoTokenizer.from_pretrained(name)
        _cache["unieval_model"] = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEVICE)
        _cache["unieval_model"].eval()
        logger.info("UniEval loaded.")
    return _cache["unieval_tokenizer"], _cache["unieval_model"]


def _unieval_score(question, source, tokenizer, model):
    input_text = f"question: {question} </s> premise: {source}"
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
        no_id = tokenizer("No", add_special_tokens=False).input_ids[0]

        decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(DEVICE)
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[0, 0, :]

        probs = torch.softmax(logits[[yes_id, no_id]], dim=0)
        return probs[0].item()


@app.route("/unieval", methods=["POST"])
def unieval():
    data = request.json
    ref = data.get("reference", "")
    cand = data.get("candidate", "")
    dimension = data.get("dimension", "overall")

    if not ref or not cand:
        return jsonify({"error": "reference and candidate required"}), 400

    questions = {
        "coherence": f"Is this a coherent piece of text? {cand}",
        "consistency": f"Is this text consistent with the reference? "
        f"Reference: {ref} Text: {cand}",
        "fluency": f"Is this a fluent piece of text? {cand}",
        "relevance": f"Is this text relevant to the reference? "
        f"Reference: {ref} Text: {cand}",
        "overall": f"Is this a good response given the reference? "
        f"Reference: {ref} Response: {cand}",
    }

    tokenizer, model = get_unieval_model()

    if dimension == "all":
        scores = {}
        for dim, question in questions.items():
            scores[dim] = round(_unieval_score(question, cand, tokenizer, model), 6)
        scores["score"] = round(
            float(np.mean([v for k, v in scores.items() if k != "score"])), 6
        )
        return jsonify(scores)
    else:
        question = questions.get(dimension, questions["overall"])
        score = _unieval_score(question, cand, tokenizer, model)
        return jsonify({"score": round(score, 6), "dimension": dimension})


# ── GPTScore ───────────────────────────────────────────────────────────


def get_gptscore_model():
    if "gptscore" not in _cache:
        logger.info("Loading GPT-2 for GPTScore...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        _cache["gptscore_tokenizer"] = GPT2Tokenizer.from_pretrained("gpt2")
        _cache["gptscore_model"] = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
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


# ── Health ─────────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    loaded = list(
        set(k.replace("_tokenizer", "").replace("_model", "") for k in _cache.keys())
    )
    return jsonify(
        {
            "status": "ok",
            "device": DEVICE,
            "loaded_models": loaded,
            "available": ["bertscore", "moverscore", "unieval", "gptscore"],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9200))
    logger.info(f"Model server starting on :{port} (device: {DEVICE})")
    app.run(host="0.0.0.0", port=port)
