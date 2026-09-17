"""Pre-download all model weights during Docker build.

Forces safetensors format to avoid CVE-2025-32434 with old PyTorch versions.
"""

import os

os.environ["HF_HOME"] = "/models"

print("Downloading RoBERTa-large (BERTScore + MoverScore)...")
from transformers import AutoModel, AutoTokenizer

AutoTokenizer.from_pretrained("roberta-large")
AutoModel.from_pretrained("roberta-large", use_safetensors=True)

print("Downloading UniEval (T5-based)...")
from transformers import AutoModelForSeq2SeqLM

AutoTokenizer.from_pretrained("MingZhong/unieval-sum")
AutoModelForSeq2SeqLM.from_pretrained("MingZhong/unieval-sum", use_safetensors=True)

GPTSCORE_MODEL = os.environ.get("GPTSCORE_MODEL", "gpt2-large")
print(f"Downloading {GPTSCORE_MODEL} (GPTScore)...")
from transformers import GPT2LMHeadModel, GPT2Tokenizer

GPT2Tokenizer.from_pretrained(GPTSCORE_MODEL)
GPT2LMHeadModel.from_pretrained(GPTSCORE_MODEL, use_safetensors=True)

print("Downloading BART-large-cnn (BARTScore)...")
from transformers import BartForConditionalGeneration, BartTokenizer

BartTokenizer.from_pretrained("facebook/bart-large-cnn")
BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn", use_safetensors=True
)

print("All models downloaded.")
