"""
Project   : Mini GPT Wikipédia
File      : app_streamlit.py
Author    : Arthur PRIGENT <arthurprigent760@gmail.com>
Created   : 2025-10-08 11:01
Python    : version
Description: Streamlit app pour interagir avec le modèle TinyGPT entraîné sur des articles Wikipédia.
"""

# -*- coding: utf-8 -*-



import streamlit as st
import torch
from pathlib import Path
from transformers import AutoTokenizer
import json
import sys

# Permet d'importer depuis training/
BASE = Path(__file__).resolve().parents[1]
TRAINING = BASE / "training"
sys.path.append(str(TRAINING))
from tiny_gpt import TinyGPT  # noqa


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str, data_dir: str):
    device = get_device()
    meta = json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = TinyGPT(**cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tok, device, cfg["block_size"]


st.set_page_config(page_title="Mini GPT Wikipédia", page_icon="📚")
st.title("📚 Mini GPT Wikipédia")

ckpt_path = st.text_input("Chemin du checkpoint", value="models/tinygpt/tinygpt_final.pt")
data_dir = st.text_input("Dossier des tokens", value="data/tokens/gpt2_fr_ia")

if Path(ckpt_path).exists() and (Path(data_dir) / "meta.json").exists():
    model, tok, device, block = load_model(ckpt_path, data_dir)
    st.success(f"Modèle chargé sur {device}")

    prompt = st.text_area("Prompt", value="Question: Qu'est-ce qu'un transformeur en IA?\nRéponse:", height=150)
    max_new = st.slider("Max nouveaux tokens", 32, 512, 160, 32)
    temp = st.slider("Température", 0.1, 1.5, 0.9, 0.1)
    topk = st.slider("Top-k", 0, 200, 50, 5)

    if st.button("Générer"):
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        out = model.generate(ids, max_new_tokens=max_new, temperature=temp, top_k=(None if topk == 0 else topk))
        txt = tok.decode(out[0].tolist())
        st.write("---")
        st.markdown(f"**Réponse :**\n\n{txt}")
else:
    st.info("Renseigne un checkpoint entraîné et un dossier de tokens valides.")