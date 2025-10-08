"""
Project   : Mini GPT Wikipédia
File      : dataset.py
Author    : Arthur PRIGENT <arthurprigent760@gmail.com>
Created   : 2025-10-08 10:59
Python    : version
Description: Script pour construire un dataset de tokens à partir d'un fichier JSONL.
"""

# -*- coding: utf-8 -*-



from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def build_token_dataset(jsonl_path: str, out_dir: str, tokenizer_name: str = "gpt2", train_ratio: float = 0.9):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            txt = obj.get("text", "").strip()
            if txt:
                texts.append(txt)

    # Concat en un seul long flux avec séparateurs EOS
    big_text = ("\n\n" + tok.eos_token + "\n\n").join(texts)
    ids = tok.encode(big_text, add_special_tokens=False)
    ids = np.array(ids, dtype=np.uint32)

    n = len(ids)
    n_train = int(n * train_ratio)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    train_ids.tofile(out / "train.bin")
    val_ids.tofile(out / "val.bin")

    meta = {
        "tokenizer": tokenizer_name,
        "vocab_size": tok.vocab_size,
        "eos_token_id": tok.eos_token_id,
        "n_train_tokens": int(train_ids.size),
        "n_val_tokens": int(val_ids.size),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    # Exemple:
    build_token_dataset("data/raw/wiki_fr_ia.jsonl", "data/tokens/gpt2_fr_ia")