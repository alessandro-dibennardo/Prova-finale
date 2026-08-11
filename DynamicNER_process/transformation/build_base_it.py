#!/usr/bin/env python3
"""
build_base_it.py

Converte un corpus italiano annotato nel formato "BASE" di DynamicNER,
identico allo schema osservato in base/en/BASE/*.json e base/de/BASE/*.json.

Formato di OUTPUT (uno per sentenceN):
{
    "sentence1": {
        "sentence": "...",
        "entity":   ["...", "..."],
        "category": ["...", "..."]   # categorie di TERZO livello, vedi DynamicNER.json
    },
    ...
}

Formato di INPUT atteso (JSONL, un record per riga) — adatta la funzione
`load_input_records` se il tuo corpus di partenza ha un formato diverso
(es. CoNLL/BIO, WikiAnn, output di un LLM di annotazione, ecc.):

{"text": "L'Università di Bologna ha sede a Bologna.",
 "entities": [
    {"text": "Università di Bologna", "start": 2,  "end": 24, "category": "school"},
    {"text": "Bologna",               "start": 35, "end": 42, "category": "city"}
 ]}

Note importanti (vedi analisi dello schema fatta con l'utente):
- Le stringhe in "entity" devono comparire ESATTAMENTE come sottostringa in
  "sentence" (match testuale, non solo offset). Lo script verifica questo
  vincolo e scarta/segnala le frasi che lo violano.
- Le categorie devono essere etichette di TERZO livello prese da
  DynamicNER.json (es. "city", "school", "author"), non "location" o
  "geo-political entity".
- Per compatibilità immediata col codice esistente (dynamic1-4.py, main.py,
  stage1/2_trans.py) le categorie restano in INGLESE anche per il corpus
  italiano: cambia solo la lingua della frase, non la lingua della gerarchia.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Carica la gerarchia ufficiale per validare le categorie di terzo livello
# ---------------------------------------------------------------------------

def load_valid_leaf_categories(hierarchy_path: Path) -> set[str]:
    """Estrae l'insieme di tutte le categorie di terzo livello valide."""
    with open(hierarchy_path, encoding="utf-8") as f:
        hierarchy = json.load(f)

    valid = set()
    for csv_string in hierarchy["third-level"].values():
        for cat in csv_string.split(","):
            valid.add(cat.strip())
    return valid


# ---------------------------------------------------------------------------
# 2. Carica il tuo corpus grezzo — ADATTA QUESTA FUNZIONE al tuo formato reale
# ---------------------------------------------------------------------------

def load_input_records(input_path: Path):
    """
    Legge un file JSONL con record {text, entities:[{text,start,end,category}]}.
    Ogni riga del file deve essere un oggetto JSON valido.
    Modifica questa funzione se il tuo corpus di partenza ha un altro formato
    (es. CSV, BIO già tokenizzato, output strutturato di un LLM, ecc.).
    """
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] riga {line_no} non è JSON valido, saltata: {e}",
                      file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# 3. Validazione: l'entità deve comparire come sottostringa esatta nel testo
# ---------------------------------------------------------------------------

def normalize_apostrophes(text: str) -> str:
    """
    Uniforma le varianti di apostrofo (', ', `) a un unico carattere.
    Fondamentale per l'italiano (dell', l', un', ecc.): se il tuo corpus
    sorgente mischia apostrofo dritto (U+0027) e tipografico (U+2019),
    il match a sottostringa esatta richiesto dal formato BASE fallirebbe
    silenziosamente altrimenti.
    """
    return unicodedata.normalize("NFC", text).replace("\u2019", "'").replace("`", "'")


def build_base_record(text: str, entities: list[dict], valid_categories: set[str],
                       sentence_id: str) -> dict | None:
    text = normalize_apostrophes(text)

    entity_strings = []
    category_strings = []

    for ent in entities:
        ent_text = normalize_apostrophes(ent["text"])
        category = ent["category"].strip()

        if category not in valid_categories:
            print(f"[SKIP {sentence_id}] categoria non valida: '{category}' "
                  f"per entità '{ent_text}'", file=sys.stderr)
            return None

        if ent_text not in text:
            print(f"[SKIP {sentence_id}] entità '{ent_text}' non trovata come "
                  f"sottostringa esatta nella frase", file=sys.stderr)
            return None

        entity_strings.append(ent_text)
        category_strings.append(category)

    if not entity_strings:
        # Nessuna entità valida: la frase viene comunque scartata perché
        # DynamicNER non include frasi senza entità nel formato BASE.
        return None

    return {
        "sentence": text,
        "entity": entity_strings,
        "category": category_strings,
    }


# ---------------------------------------------------------------------------
# 4. Main: costruisce train/dev/test e scrive i file nella struttura corretta
# ---------------------------------------------------------------------------

def convert_split(records: list[dict], valid_categories: set[str]) -> dict:
    output = {}
    skipped = 0
    for i, rec in enumerate(records, start=1):
        sentence_id = f"sentence{i}"
        base_rec = build_base_record(
            rec["text"], rec["entities"], valid_categories, sentence_id
        )
        if base_rec is None:
            skipped += 1
            continue
        output[sentence_id] = base_rec

    print(f"Convertite {len(output)} frasi, scartate {skipped}.", file=sys.stderr)
    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Converte corpus italiano JSONL nel formato BASE di DynamicNER.")
    parser.add_argument("hierarchy", type=Path,
                         help="Path a DynamicNER.json (gerarchia categorie)")
    parser.add_argument("inputs", type=Path, nargs="+",
                         help="Uno o più file JSONL di input, es. "
                              "corpus_it_train.jsonl corpus_it_synthetic.jsonl "
                              "(vengono uniti nello stesso split di output)")
    parser.add_argument("output_dir", type=Path,
                         help="Cartella di output, es. DynamicNER/base/it/BASE")
    parser.add_argument("split_name", choices=["train", "dev", "test"],
                         help="Nome dello split di output (train.json / dev.json / test.json)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    valid_categories = load_valid_leaf_categories(args.hierarchy)

    records = []
    for input_path in args.inputs:
        file_records = load_input_records(input_path)
        print(f"[INFO] {input_path}: {len(file_records)} record letti", file=sys.stderr)
        records.extend(file_records)

    base_data = convert_split(records, valid_categories)

    output_path = args.output_dir / f"{args.split_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(base_data, f, ensure_ascii=False, indent=4)

    print(f"Scritto: {output_path} ({len(base_data)} frasi totali, "
          f"da {len(args.inputs)} file di input)", file=sys.stderr)


if __name__ == "__main__":
    main()