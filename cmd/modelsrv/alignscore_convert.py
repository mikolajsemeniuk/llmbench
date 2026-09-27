"""
One-off: turn the official AlignScore-large checkpoint (a PyTorch
Lightning pickle) into a safetensors file of plain weights, so the model
server never unpickles it. The checkpoint is read with
torch.load(weights_only=True), which only rebuilds tensors and plain
containers and refuses anything else.

    python alignscore_convert.py   # writes $ALIGNSCORE_WEIGHTS
"""

import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

OUT = os.environ.get(
    "ALIGNSCORE_WEIGHTS",
    os.path.expanduser("~/.cache/llmbench/alignscore-large.safetensors"),
)

path = hf_hub_download("yzha/AlignScore", "AlignScore-large.ckpt")
ckpt = torch.load(path, map_location="cpu", weights_only=True)
# The MLM head (tied tensors that safetensors refuses) is only used in
# AlignScore's pre-training, not by inference.
state = {k: v.contiguous() for k, v in ckpt["state_dict"].items() if not k.startswith("mlm_head.")}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file(state, OUT)
print(f"{OUT}: {len(state)} tensors")
print("non-encoder tensors:", sorted(k for k in state if not k.startswith("base_model.")))
