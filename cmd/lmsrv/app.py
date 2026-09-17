"""
lmsrv — minimal scoring server for counterfactual-margin metrics.

Two endpoints, both batched, both teacher-forced (no generation):

    POST /lmscore   mean token log-probability of each target, optionally
                    conditioned on a shared context. The context is
                    encoded once and its KV cache is reused for every
                    target in the batch, so conditioning a summary on a
                    700-token news article costs one article pass per
                    document rather than one per (candidate, perturbation).
    POST /nli       entailment probabilities for (premise, hypothesis)
                    pairs. Used only for the SummaC-style baseline the
                    reviewers asked for, not by the margin metric itself.
    GET  /health

Env:
    LM_MODEL     causal LM for /lmscore   (default Qwen/Qwen3-0.6B-Base)
    NLI_MODEL    sequence classifier      (default MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
    LM_DEVICE    cuda|cpu|mps             (default: autodetect)
    LM_PORT      default 9300
"""

import logging
import os
import time

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__)

LM_MODEL = os.environ.get("LM_MODEL", "Qwen/Qwen3-0.6B-Base")
NLI_MODEL = os.environ.get("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
MAX_CTX = int(os.environ.get("LM_MAX_CTX", "1536"))
BATCH = int(os.environ.get("LM_BATCH", "16"))


def _device():
    d = os.environ.get("LM_DEVICE", "")
    if d:
        return d
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _device()
_cache = {}


def get_lm():
    if "lm" not in _cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"loading causal LM {LM_MODEL} on {DEVICE} ...")
        tok = AutoTokenizer.from_pretrained(LM_MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(LM_MODEL, dtype=dtype).to(DEVICE)
        model.eval()
        _cache["lm"] = (tok, model)
        logger.info("causal LM loaded.")
    return _cache["lm"]


def get_nli():
    if "nli" not in _cache:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"loading NLI model {NLI_MODEL} on {DEVICE} ...")
        tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
        model.eval()
        _cache["nli"] = (tok, model)
        logger.info("NLI model loaded.")
    return _cache["nli"]


# ── /lmscore ───────────────────────────────────────────────────────────


@torch.inference_mode()
def _score_pairs(pairs):
    """
    Scores a batch of (context, target) pairs whose contexts differ.

    Used by /lmpairs. A judge-style prompt contains the candidate
    summary, so every row has its own prefix and the shared-prefix trick
    of /lmscore does not apply; batching over rows is what is left.
    Rows are left-padded so that targets end at the same position and
    the LM head can be restricted to the last `keep` positions.
    """
    tok, model = get_lm()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    out_sum, out_cnt, ctx_total = [], [], 0

    for start in range(0, len(pairs), BATCH):
        chunk = pairs[start : start + BATCH]
        prefixes, targets = [], []
        for p in chunk:
            ctx = p.get("context", "") or ""
            ids = tok(ctx, add_special_tokens=False)["input_ids"][-MAX_CTX:] if ctx else [tok.eos_token_id]
            prefixes.append(ids)
            targets.append(tok(p.get("target", ""), add_special_tokens=False)["input_ids"])
        ctx_total += sum(len(p) for p in prefixes)

        seqs = [pre + t for pre, t in zip(prefixes, targets)]
        width = max(len(s) for s in seqs)
        keep = max(1, max(len(t) for t in targets)) + 1

        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s) :] = torch.tensor(s)
            mask[i, width - len(s) :] = 1
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)

        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits
        for i, t in enumerate(targets):
            n = len(t)
            if n == 0:
                out_sum.append(0.0)
                out_cnt.append(0)
                continue
            win = logits[i, -(n + 1) : -1].float()
            gold = ids[i, width - n :]
            lp = torch.log_softmax(win, dim=-1).gather(-1, gold.unsqueeze(-1))
            out_sum.append(float(lp.sum().item()))
            out_cnt.append(n)
        del logits

    return out_sum, out_cnt, ctx_total


@app.route("/lmpairs", methods=["POST"])
def lmpairs():
    data = request.json or {}
    pairs = data.get("pairs", [])
    if not isinstance(pairs, list) or not pairs:
        return jsonify({"error": "pairs (non-empty list) required"}), 400
    t0 = time.time()
    sums, cnts, ctx_tokens = _score_pairs(pairs)
    mean = [(s / c if c else 0.0) for s, c in zip(sums, cnts)]
    return jsonify(
        {
            "mean_logprob": mean,
            "sum_logprob": sums,
            "tokens": cnts,
            "context_tokens": ctx_tokens,
            "model": LM_MODEL,
            "elapsed_sec": round(time.time() - t0, 4),
        }
    )


@torch.inference_mode()
def _score_targets(context, targets):
    """
    Returns, for every target, the sum and the count of token
    log-probabilities under the LM, with `context` (possibly empty)
    prepended as conditioning. Context tokens are never scored.

    Memory matters here: a batch of 16 summaries conditioned on a
    700-token article is a (16, 1500, 151k) logits tensor if the head is
    applied to every position, which is several GB for nothing -- only
    the positions that predict target tokens are ever read. Sequences
    are therefore LEFT-padded, so every target sits at the end of its
    row, and the LM head is restricted to the last `keep` positions.
    Log-softmax is then taken one row at a time in float32.
    """
    tok, model = get_lm()

    ctx_ids = []
    if context:
        ctx_ids = tok(context, add_special_tokens=False)["input_ids"][-MAX_CTX:]
    prefix = ctx_ids if ctx_ids else [tok.eos_token_id]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    out_sum, out_cnt = [], []

    for start in range(0, len(targets), BATCH):
        chunk = targets[start : start + BATCH]
        enc = [tok(t, add_special_tokens=False)["input_ids"] for t in chunk]
        seqs = [prefix + e for e in enc]
        width = max(len(s) for s in seqs)
        keep = max(len(e) for e in enc) + 1

        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s) :] = torch.tensor(s)   # left padding
            mask[i, width - len(s) :] = 1
        ids, mask = ids.to(DEVICE), mask.to(DEVICE)

        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits
        # logits[:, -keep:] are the predictions at the last `keep`
        # positions; position width-1-k predicts token width-k.
        for i, e in enumerate(enc):
            n = len(e)
            if n == 0:
                out_sum.append(0.0)
                out_cnt.append(0)
                continue
            win = logits[i, -(n + 1) : -1].float()            # (n, V)
            gold = ids[i, width - n :]                         # (n,)
            lp = torch.log_softmax(win, dim=-1).gather(-1, gold.unsqueeze(-1))
            out_sum.append(float(lp.sum().item()))
            out_cnt.append(n)
        del logits

    return out_sum, out_cnt, len(ctx_ids)


@app.route("/lmscore", methods=["POST"])
def lmscore():
    data = request.json or {}
    context = data.get("context", "") or ""
    targets = data.get("targets", [])
    if not isinstance(targets, list) or not targets:
        return jsonify({"error": "targets (non-empty list) required"}), 400
    t0 = time.time()
    sums, cnts, ctx_tokens = _score_targets(context, targets)
    mean = [(s / c if c else 0.0) for s, c in zip(sums, cnts)]
    return jsonify(
        {
            "mean_logprob": mean,
            "sum_logprob": sums,
            "tokens": cnts,
            "context_tokens": ctx_tokens,
            "model": LM_MODEL,
            "elapsed_sec": round(time.time() - t0, 4),
        }
    )


# ── /nli ───────────────────────────────────────────────────────────────


@app.route("/nli", methods=["POST"])
@torch.inference_mode()
def nli():
    data = request.json or {}
    pairs = data.get("pairs", [])
    if not pairs:
        return jsonify({"error": "pairs required"}), 400
    tok, model = get_nli()
    labels = [l.lower() for l in model.config.id2label.values()]
    ent_idx = next(i for i, l in enumerate(labels) if l.startswith("entail"))
    con_idx = next(i for i, l in enumerate(labels) if l.startswith("contra"))

    ent, con = [], []
    for start in range(0, len(pairs), BATCH):
        chunk = pairs[start : start + BATCH]
        enc = tok(
            [p["premise"] for p in chunk],
            [p["hypothesis"] for p in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)
        ent += probs[:, ent_idx].tolist()
        con += probs[:, con_idx].tolist()
    return jsonify({"entailment": ent, "contradiction": con, "model": NLI_MODEL})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": DEVICE, "loaded": sorted(_cache.keys())})


if __name__ == "__main__":
    if os.environ.get("EAGER_LOAD", "1") == "1":
        get_lm()
    app.run(host="0.0.0.0", port=int(os.environ.get("LM_PORT", "9300")), threaded=False)
