"""
Project   : Mini GPT Wikipédia
File      : generate.py
Author    : Arthur PRIGENT <arthurprigent760@gmail.com>
Created   : 2025-10-08 11:00
Python    : version
Description: Script pour générer du texte avec un modèle TinyGPT entraîné.
"""

# -*- coding: utf-8 -*-



import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from tiny_gpt import TinyGPT


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, data_dir: str):
    device = get_device()
    meta = json.loads(Path(f"{data_dir}/meta.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = TinyGPT(**cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tok, device, cfg["block_size"]


def generate_text(prompt: str, ckpt_path: str, data_dir: str, max_new_tokens=150, temperature=0.9, top_k=50):
    model, tok, device, block = load_model(ckpt_path, data_dir)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    out_ids = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = tok.decode(out_ids[0].tolist())
    return text


if __name__ == "__main__":
    ckpt = "models/tinygpt/tinygpt_final.pt"
    data_dir = "data/tokens/gpt2_fr_ia"
    print(generate_text("Question: Qu'est-ce qu'un transformeur en IA? Réponse:", ckpt, data_dir))