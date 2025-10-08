"""
Project   : Mini GPT Wikipédia
File      : train.py
Author    : Arthur PRIGENT <arthurprigent760@gmail.com>
Created   : 2025-10-08 11:01
Python    : version
Description: Script pour entraîner un modèle TinyGPT sur des données tokenisées.
"""

# -*- coding: utf-8 -*-



import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tiny_gpt import TinyGPT

torch.set_float32_matmul_precision('high')
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BinDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int):
        data = np.memmap(bin_path, dtype=np.uint32, mode='r')
        self.data = torch.from_numpy(data.astype(np.int64))
        self.block = block_size

    def __len__(self):
        return self.data.size(0) - self.block - 1

    def __getitem__(self, i):
        x = self.data[i:i+self.block].clone()
        y = self.data[i+1:i+1+self.block].clone()
        return x, y


def train(
    data_dir="data/tokens/gpt2_fr_ia",
    out_dir="models/tinygpt",
    vocab_size=50257,
    block_size=256,
    d_model=384,
    n_layer=6,
    n_head=6,
    dropout=0.1,
    batch_size=32,
    lr=3e-4,
    max_steps=5000,
    eval_interval=200,
    eval_iters=100,
    ckpt_interval=1000,
):
    device = get_device()
    print("Device:", device)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer meta
    meta = json.loads(Path(f"{data_dir}/meta.json").read_text(encoding="utf-8"))
    vocab_size = meta.get("vocab_size", vocab_size)

    # Datasets
    train_ds = BinDataset(f"{data_dir}/train.bin", block_size)
    val_ds = BinDataset(f"{data_dir}/val.bin", block_size)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # Model
    model = TinyGPT(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head, block_size=block_size, dropout=dropout)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def estimate_loss():
        model.eval()
        outs = []
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= eval_iters:
                    break
                x = x.to(device)
                y = y.to(device)
                _, loss = model(x, y)
                outs.append(loss.detach().float().cpu())
        model.train()
        return torch.stack(outs).mean().item() if outs else None

    step = 0
    while step < max_steps:
        for x, y in train_loader:
            step += 1
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 50 == 0:
                print(f"step {step} | train loss {loss.item():.4f}")

            if step % eval_interval == 0:
                val_loss = estimate_loss()
                if val_loss is not None:
                    print(f"step {step} | val loss {val_loss:.4f}")

            if step % ckpt_interval == 0:
                ckpt_path = Path(out_dir) / f"tinygpt_step{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "block_size": block_size,
                        "d_model": d_model,
                        "n_layer": n_layer,
                        "n_head": n_head,
                        "dropout": dropout,
                    }
                }, ckpt_path)
                print("Saved:", ckpt_path)

            if step >= max_steps:
                break

    # Save final
    ckpt_path = Path(out_dir) / "tinygpt_final.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "d_model": d_model,
            "n_layer": n_layer,
            "n_head": n_head,
            "dropout": dropout,
        }
    }, ckpt_path)
    print("Saved final:", ckpt_path)


if __name__ == "__main__":
    # Exemples de hyperparams raisonnables pour un POC sur Mac :
    train(max_steps=1500, batch_size=16, d_model=320, n_layer=4, n_head=4, block_size=256)