# ══════════════════════════════════════════════════════════════════════
# ADD TO app.py — paste BEFORE the # ── Health ── section
# ══════════════════════════════════════════════════════════════════════

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
    Compute GPTScore: average conditional log-probability of candidate
    tokens given reference as context.

    GPTScore = (1/m) * Σ log P(cand_i | cand_<i, reference)

    Returns a score in [0, 1] via sigmoid normalization.
    """
    # Build prompt: reference as context, candidate as continuation
    prompt = f"Reference: {reference}\nResponse: {candidate}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(DEVICE)
    input_ids = inputs["input_ids"]

    # Find where candidate tokens start
    ref_prompt = f"Reference: {reference}\nResponse: "
    ref_ids = tokenizer(ref_prompt, return_tensors="pt",
                        truncation=True, max_length=1024)["input_ids"]
    cand_start = ref_ids.shape[1]

    if cand_start >= input_ids.shape[1]:
        return 0.0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # Get per-token log probabilities
        logits = outputs.logits  # (1, seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Gather log-prob of actual tokens for candidate portion
        total_log_prob = 0.0
        n_tokens = 0

        for i in range(cand_start, input_ids.shape[1]):
            token_id = input_ids[0, i].item()
            # Log-prob of token i is at position i-1 (shifted)
            if i > 0:
                token_log_prob = log_probs[0, i - 1, token_id].item()
                total_log_prob += token_log_prob
                n_tokens += 1

    if n_tokens == 0:
        return 0.0

    avg_log_prob = total_log_prob / n_tokens
    # avg_log_prob is negative (log of probability).
    # Typical range: -1 (high prob) to -10 (low prob).
    # Normalize to [0, 1] using sigmoid: sigmoid(avg_log_prob + 3)
    # shifts so that avg_log_prob ≈ -3 maps to 0.5
    import math
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


# ══════════════════════════════════════════════════════════════════════
# UPDATE models.py — add before "All models downloaded."
# ══════════════════════════════════════════════════════════════════════
#
# print("Downloading GPT-2 (GPTScore)...")
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# GPT2Tokenizer.from_pretrained("gpt2")
# GPT2LMHeadModel.from_pretrained("gpt2")
#
# ══════════════════════════════════════════════════════════════════════
# UPDATE health endpoint — add "gptscore" to available list
# ══════════════════════════════════════════════════════════════════════
#
# "available": ["bertscore", "moverscore", "unieval", "gptscore"],
