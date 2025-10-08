"""
Project   : Mini GPT Wikipédia
File      : app_streamlit.py
Author    : Arthur PRIGENT <arthurprigent760@gmail.com>
Created   : 2025-10-08 11:01
Python    : version
Description: Streamlit app pour interagir avec le modèle TinyGPT entraîné sur des articles Wikipédia.
"""

# -*- coding: utf-8 -*-

from pathlib import Path
import json
import sys
import time

import streamlit as st
import torch
from transformers import AutoTokenizer

# Permet d'importer depuis training/
BASE = Path(__file__).resolve().parents[1]
TRAINING = BASE / "training"
sys.path.append(str(TRAINING))
from tiny_gpt import TinyGPT  # noqa


# -----------------------------
# Helpers
# -----------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_answer(generated: str, question: str) -> str:
    """Coupe le texte avant le prompt si le modèle le recopie, et nettoie un peu la sortie."""
    # Si le modèle ré-écrit le prompt, on tente de garder seulement la fin utile
    marker = f"Question: {question}\nRéponse:"
    if marker in generated:
        generated = generated.split(marker, 1)[-1]
    return generated.strip()


# -----------------------------
# Cache
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str, data_dir: str):
    device = get_device()
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json introuvable dans {data_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])  # ex: 'gpt2'

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")

    ckpt = torch.load(ckpt_file, map_location=device)
    cfg = ckpt["config"]
    model = TinyGPT(**cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    block_size = cfg.get("block_size", 256)
    return model, tok, device, block_size


# -----------------------------
# UI config
# -----------------------------
st.set_page_config(page_title="Mini GPT Wikipédia", page_icon="📚", layout="wide")

# Petite touche de style (centrer le contenu)
st.markdown(
    """
    <style>
    .main > div { padding-top: 1rem; }
    .small { font-size: 0.9rem; color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar — paramètres
# -----------------------------
with st.sidebar:
    st.header("⚙️ Paramètres modèle")
    ckpt_path = st.text_input(
        "Checkpoint",
        value="models/tinygpt/tinygpt_final.pt",
        help="Chemin vers le .pt sauvegardé",
    )
    data_dir = st.text_input(
        "Dossier tokens",
        value="data/tokens/gpt2_fr_ia",
        help="Dossier contenant meta.json du tokenizer",
    )

    st.divider()
    st.subheader("🧪 Génération")
    max_new = st.slider("Max nouveaux tokens", 16, 1024, 160, 16)
    temp = st.slider("Température", 0.1, 1.5, 0.9, 0.1)
    topk = st.slider("Top-k (0 = off)", 0, 200, 50, 5)
    seed = st.number_input("Seed (optionnel)", value=0, min_value=0, step=1)

    st.divider()
    if st.button("🔄 Recharger le modèle"):
        # Invalider le cache si besoin
        load_model.clear()
        st.experimental_rerun()


# -----------------------------
# Main — en-tête + question/réponse
# -----------------------------
st.title("📚 Mini GPT Wikipédia")
st.caption("Mini LLM entraîné sur un sous-ensemble de Wikipédia — scraper → entraînement → Web UI.")

left, right = st.columns([2, 1])
with right:
    # Informations système
    try:
        _dev = get_device()
        st.metric("Device", str(_dev))
        st.text(f"CUDA dispo: {torch.cuda.is_available()}")
        st.text(f"MPS dispo: {torch.backends.mps.is_available()}")
    except Exception:
        pass

with left:
    st.subheader("📝 Pose ta question")
    default_q = "Qui est Alan Turing ?"
    question = st.text_area("Question", value=default_q, height=120, placeholder="Écris ta question ici…")

    colA, colB = st.columns([1, 1])
    with colA:
        btn = st.button("🚀 Générer la réponse", use_container_width=True)
    with colB:
        clear = st.button("🧹 Effacer", use_container_width=True)

if clear:
    if "last_answer" in st.session_state:
        del st.session_state["last_answer"]
    st.experimental_rerun()

# Charger paresseusement le modèle (affiche un message si manquant)
model = tok = device = block = None
model_loaded = False
try:
    model, tok, device, block = load_model(ckpt_path, data_dir)
    model_loaded = True
    st.success(f"Modèle chargé sur {device}")
except Exception as e:
    st.warning(f"Modèle non chargé : {e}")

# Génération
if btn and model_loaded:
    if seed:
        torch.manual_seed(int(seed))
    prompt = f"Question: {question}\nRéponse:"

    t0 = time.time()
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids,
        max_new_tokens=int(max_new),
        temperature=float(temp),
        top_k=(None if int(topk) == 0 else int(topk)),
    )
    txt = tok.decode(out[0].tolist())
    answer = format_answer(txt, question)
    dt = time.time() - t0

    st.session_state["last_answer"] = answer
    st.info(f"Généré en {dt:.2f}s | bloc={block}")

# Affichage de la réponse
st.subheader("🧠 Réponse")
if "last_answer" in st.session_state:
    st.markdown(st.session_state["last_answer"]) 
else:
    st.markdown("*La réponse apparaîtra ici.*")
