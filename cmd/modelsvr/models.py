"""Pre-download all model weights during Docker build."""

import os

os.environ["HF_HOME"] = "/models"

print("Downloading RoBERTa-large (BERTScore + MoverScore)...")
from transformers import AutoModel, AutoTokenizer

AutoTokenizer.from_pretrained("roberta-large")
AutoModel.from_pretrained("roberta-large")

print("Downloading UniEval (T5-based)...")
from transformers import AutoModelForSeq2SeqLM

AutoTokenizer.from_pretrained("MingZhong/unieval-sum")
AutoModelForSeq2SeqLM.from_pretrained("MingZhong/unieval-sum")

print("All models downloaded.")
