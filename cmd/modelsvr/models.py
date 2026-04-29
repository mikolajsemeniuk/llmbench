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

print("Downloading GPT-2 (GPTScore)...")
from transformers import GPT2LMHeadModel, GPT2Tokenizer

GPT2Tokenizer.from_pretrained("gpt2")
GPT2LMHeadModel.from_pretrained("gpt2")

print("Downloading BART-large-cnn (BARTScore)...")
from transformers import BartForConditionalGeneration, BartTokenizer

BartTokenizer.from_pretrained("facebook/bart-large-cnn")
BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

print("All models downloaded.")
