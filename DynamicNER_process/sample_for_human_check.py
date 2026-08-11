#!/usr/bin/env python3
r"""
sample_for_human_check.py

Estrae un campione stratificato di (frase, entita', categoria) dai file
BASE per la verifica manuale della correttezza delle etichette di terzo
livello, seguendo la formula di campionamento stratificato del paper
DynamicNER (Appendice E, Eq. 8):

    s_i = min(S / m, n_i)

dove S e' la dimensione totale del campione desiderato, m e' il numero di
categorie distinte presenti nei dati, e n_i e' il numero di istanze
disponibili per la categoria i (quindi non si campiona mai piu' di quanto
esiste realmente per una categoria rara).

Uso:
    python sample_for_human_check.py \
        ..\..\DynamicNER\base\it\BASE\train.json \
        ..\..\DynamicNER\base\it\BASE\dev.json \
        ..\..\DynamicNER\base\it\BASE\test.json \
        --sample-size 180 \
        --output human_check_it.csv
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_base_file(path: Path, split_name: str) -> List[Tuple[str, str, str, str]]:
    """Appiattisce un file BASE in record (split, sentence, entity, category)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for sentence_id, info in data.items():
        sentence = info["sentence"]
        for entity, category in zip(info["entity"], info["category"]):
            records.append((split_name, sentence, entity, category))
    return records


def stratified_sample(records: List[Tuple[str, str, str, str]],
                       sample_size: int, seed: int = 42) -> List[Tuple[str, str, str, str]]:
    """Applica la formula s_i = min(S/m, n_i) del paper (Appendice E)."""
    random.seed(seed)

    by_category: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
    for rec in records:
        by_category[rec[3]].append(rec)

    m = len(by_category)
    if m == 0:
        return []

    target_per_category = sample_size / m

    sampled = []
    for category, cat_records in by_category.items():
        n_i = len(cat_records)
        s_i = min(round(target_per_category), n_i)
        sampled.extend(random.sample(cat_records, s_i))

    random.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_files", nargs="+", type=Path,
                         help="Uno o piu' file BASE (train.json, dev.json, test.json). "
                              "Il nome dello split viene dedotto dallo stem del file.")
    parser.add_argument("--sample-size", type=int, default=180,
                         help="Dimensione totale del campione desiderato S "
                              "(il paper usa 200 per l'IAA; 150-200 come "
                              "richiesto). Default: 180.")
    parser.add_argument("--output", type=Path, default=Path("human_check_it.csv"),
                         help="File CSV di output da compilare a mano.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed per la riproducibilita' del campionamento.")
    args = parser.parse_args()

    all_records = []
    for path in args.base_files:
        split_name = path.stem  # 'train', 'dev', o 'test'
        recs = load_base_file(path, split_name)
        print(f"[INFO] {path}: {len(recs)} entita' totali (tutte le frasi)")
        all_records.extend(recs)

    total_categories = len(set(r[3] for r in all_records))
    print(f"[INFO] Totale entita' disponibili: {len(all_records)}, "
          f"categorie distinte: {total_categories}")

    sampled = stratified_sample(all_records, args.sample_size, seed=args.seed)
    print(f"[INFO] Campione stratificato estratto: {len(sampled)} istanze "
          f"(target: {args.sample_size})")

    # Riepilogo per macro-controllo: quante categorie sono rappresentate
    sampled_categories = set(r[3] for r in sampled)
    print(f"[INFO] Categorie rappresentate nel campione: "
          f"{len(sampled_categories)}/{total_categories}")
    if len(sampled_categories) < total_categories:
        missing = set(r[3] for r in all_records) - sampled_categories
        print(f"[WARN] Categorie ASSENTI dal campione (dimensione target "
              f"troppo piccola rispetto al numero di categorie): {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "split", "sentence", "entity", "category_llm",
                          "corretto (S/N)", "note"])
        for i, (split, sentence, entity, category) in enumerate(sampled, start=1):
            writer.writerow([i, split, sentence, entity, category, "", ""])

    print(f"\n[OK] Campione scritto in: {args.output}")
    print("Compila la colonna 'corretto (S/N)' verificando se l'entita' "
          "e' correttamente classificata nella categoria indicata, guardando "
          "la frase come contesto. Usa 'note' per casi ambigui o errori "
          "specifici (es. categoria plausibile ma non ottimale).")


if __name__ == "__main__":
    main()