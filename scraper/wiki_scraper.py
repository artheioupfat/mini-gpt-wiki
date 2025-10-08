"""
Project   : Mini GPT Wikipédia
File      : wiki_scraper.py
Author    : Arthur PRIGENT <arthurprigent760@gmail.com>
Created   : 2025-10-08 10:58
Python    : version
Description: ...
"""

# -*- coding: utf-8 -*-


import argparse
import json
import re
import time
from pathlib import Path
from tqdm.auto import tqdm
import requests
import mwparserfromhell as mwp

API_URL = "https://{lang}.wikipedia.org/w/api.php"


def clean_text(wikitext: str) -> str:
    """Nettoyage du texte wiki brut."""
    code = mwp.parse(wikitext)
    for t in code.filter_templates():
        try:
            code.remove(t)
        except Exception:
            pass
    text = code.strip_code()
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_pages(titles, lang="fr", user_agent=None):
    """Récupère jusqu'à 50 pages en une requête."""
    if not titles:
        return []

    headers = {
        "User-Agent": user_agent or "MiniGPTWiki/1.0 (Prénom Nom; contact: prenom.nom@example.com)"
    }
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "formatversion": "2",
        "format": "json",
        "titles": "|".join(titles),
    }

    resp = requests.get(API_URL.format(lang=lang), params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pages = []
    for page in data.get("query", {}).get("pages", []):
        if "missing" in page:
            continue
        title = page.get("title")
        text = page.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")
        cleaned = clean_text(text)
        if len(cleaned.split()) < 100:
            continue
        pages.append({
            "title": title,
            "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "text": cleaned,
            "lang": lang
        })
    return pages


def get_links(title, lang="fr", user_agent=None):
    """Récupère tous les liens sortants de la page donnée."""
    headers = {
        "User-Agent": user_agent or "MiniGPTWiki/1.0 (Arthur Prigent; contact: arthur.prigent@example.com)"
    }
    links = []
    plcontinue = ""
    while True:
        params = {
            "action": "query",
            "titles": title,
            "prop": "links",
            "plnamespace": 0,
            "pllimit": "max",
            "format": "json",
        }
        if plcontinue:
            params["plcontinue"] = plcontinue
        resp = requests.get(API_URL.format(lang=lang), params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            for link in page.get("links", []):
                links.append(link["title"])
        if "continue" in data:
            plcontinue = data["continue"].get("plcontinue", "")
            time.sleep(0.05)
        else:
            break
    return links


def crawl_from_root(root_title, lang="fr", depth=1, max_pages=500, sleep_s=0.1, user_agent=None, out_path="data/raw/wiki.jsonl"):
    """Crawl BFS depuis la page mère, par lots de 50 pages."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Charger les titres déjà connus
    known_titles = set()
    if Path(out_path).exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    known_titles.add(obj.get("title"))
                except Exception:
                    pass

    visited = set(known_titles)
    collected = []
    frontier = [(root_title, 0)]
    pbar = tqdm(total=max_pages, desc="Scraping pages (batches of 50)", unit="pg")

    while frontier and len(visited) < max_pages:
        # Prépare un batch de 50 titres
        batch_titles = []
        while frontier and len(batch_titles) < 50:
            title, d = frontier.pop(0)
            if title in visited:
                continue
            batch_titles.append(title)
            visited.add(title)

        if not batch_titles:
            break

        # Téléchargement du batch
        try:
            pages = fetch_pages(batch_titles, lang=lang, user_agent=user_agent)
        except Exception as e:
            print(f"Erreur sur batch {batch_titles[:2]}...: {e}")
            time.sleep(1)
            continue

        # Ajout au résultat global
        collected.extend(pages)
        pbar.update(len(pages))
        pbar.set_postfix({"depth": depth, "known": len(visited)})

        # Exploration des liens des pages du batch
        if depth > 0:
            for page in pages:
                try:
                    for link in get_links(page["title"], lang=lang, user_agent=user_agent):
                        if link not in visited:
                            frontier.append((link, depth - 1))
                except Exception:
                    continue

        # Écriture incrémentale
        if collected:
            with open(out_path, "a", encoding="utf-8") as f:
                for p in collected:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            collected = []

        # Pause courte entre lots
        time.sleep(sleep_s)

        if len(visited) >= max_pages:
            break

    pbar.close()
    print(f"✅ Terminé — {len(visited)} titres visités. Résultat dans {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Titre de la page racine, ex: 'Intelligence artificielle'")
    ap.add_argument("--lang", default="fr")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--max_pages", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--user_agent", default=None)
    ap.add_argument("--out", default="data/raw/wiki_fast.jsonl")
    args = ap.parse_args()

    crawl_from_root(
        args.root,
        lang=args.lang,
        depth=args.depth,
        max_pages=args.max_pages,
        sleep_s=args.sleep,
        user_agent=args.user_agent,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
