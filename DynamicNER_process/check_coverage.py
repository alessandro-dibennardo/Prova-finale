#!/usr/bin/env python3
"""
check_coverage.py

Verifica quante delle 155 categorie di terzo livello di DynamicNER.json
sono effettivamente popolate da uno o piu' file JSONL (formato prodotto
da multiconer_to_jsonl.py: {"text": ..., "entities": [{"category": ...}]})
oppure da file BASE gia' convertiti (formato {"sentenceN": {"category": [...]}}).

Uso:
    python3 check_coverage.py DynamicNER.json corpus_it_train.jsonl corpus_it_dev.jsonl ...

Accetta un numero arbitrario di file di input, in JSONL o BASE-JSON:
rileva automaticamente il formato guardando la prima riga/struttura.
"""

import json
import sys
from pathlib import Path
from collections import Counter


def load_hierarchy(hierarchy_path: Path):
    with open(hierarchy_path, encoding="utf-8") as f:
        hierarchy = json.load(f)

    leaf_to_parent = {}
    for parent, csv_string in hierarchy["third-level"].items():
        for leaf in csv_string.split(","):
            leaf_to_parent[leaf.strip()] = parent

    all_leaves = set(leaf_to_parent.keys())
    return all_leaves, leaf_to_parent


def detect_format_and_extract_categories(file_path: Path) -> Counter:
    """
    Rileva se il file e' JSONL (multiconer_to_jsonl.py) o BASE-JSON
    (build_base_it.py / example.json) e restituisce un Counter delle
    categorie trovate.
    """
    counter = Counter()
    suffix = file_path.suffix.lower()

    if suffix == ".jsonl":
        with open(file_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] {file_path}:{line_no} riga non valida, saltata",
                          file=sys.stderr)
                    continue
                for ent in rec.get("entities", []):
                    counter[ent["category"]] += 1

    elif suffix == ".json":
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        # Formato BASE: {"sentence1": {"category": [...]}, ...}
        for sentence_info in data.values():
            for cat in sentence_info.get("category", []):
                counter[cat] += 1

    else:
        print(f"[WARN] estensione non riconosciuta per {file_path}, saltato",
              file=sys.stderr)

    return counter


def main():
    if len(sys.argv) < 3:
        print("Uso: python3 check_coverage.py <DynamicNER.json> <file1.jsonl> "
              "[<file2.json> ...]", file=sys.stderr)
        sys.exit(1)

    hierarchy_path = Path(sys.argv[1])
    input_files = [Path(p) for p in sys.argv[2:]]

    all_leaves, leaf_to_parent = load_hierarchy(hierarchy_path)

    total_counter = Counter()
    for file_path in input_files:
        if not file_path.exists():
            print(f"[WARN] file non trovato: {file_path}", file=sys.stderr)
            continue
        file_counter = detect_format_and_extract_categories(file_path)
        total_counter.update(file_counter)
        print(f"  {file_path}: {sum(file_counter.values())} entita', "
              f"{len(file_counter)} categorie distinte", file=sys.stderr)

    covered = set(total_counter.keys())

    # Categorie usate ma NON presenti nella gerarchia ufficiale: segnale di
    # errore (typo o mappatura sbagliata a monte), va sempre indagato.
    unknown = covered - all_leaves
    if unknown:
        print("\n⚠️  Categorie trovate nei dati ma ASSENTI da DynamicNER.json "
              "(probabile errore di mappatura):", file=sys.stderr)
        for cat in sorted(unknown):
            print(f"  {cat}: {total_counter[cat]} occorrenze", file=sys.stderr)

    missing = all_leaves - covered
    present = all_leaves & covered

    print(f"\n=== RIEPILOGO COPERTURA ===")
    print(f"Categorie totali (DynamicNER.json): {len(all_leaves)}")
    print(f"Categorie coperte:                  {len(present)} "
          f"({100 * len(present) / len(all_leaves):.1f}%)")
    print(f"Categorie MANCANTI:                 {len(missing)} "
          f"({100 * len(missing) / len(all_leaves):.1f}%)")

    if missing:
        print(f"\nElenco categorie mancanti, raggruppate per macro-categoria "
              f"di secondo livello (utile per organizzare la generazione LLM "
              f"a blocchi tematici):\n")

        missing_by_parent = {}
        for leaf in missing:
            parent = leaf_to_parent[leaf]
            missing_by_parent.setdefault(parent, []).append(leaf)

        for parent in sorted(missing_by_parent.keys()):
            leaves = sorted(missing_by_parent[parent])
            print(f"  [{parent}]")
            for leaf in leaves:
                print(f"    - {leaf}")

    # Bonus: categorie con pochissime occorrenze (probabile sotto-copertura
    # anche se tecnicamente "presenti") - utile per decidere se la
    # generazione LLM serve anche li', non solo per le assenti del tutto.
    thin_threshold = 5
    thin = {cat: n for cat, n in total_counter.items()
            if cat in all_leaves and n < thin_threshold}
    if thin:
        print(f"\n=== Categorie presenti ma con MENO DI {thin_threshold} "
              f"esempi (copertura debole) ===")
        for cat, n in sorted(thin.items(), key=lambda x: x[1]):
            print(f"  {cat}: {n} occorrenze")


if __name__ == "__main__":
    main()