#!/usr/bin/env python3
"""
multiconer_to_jsonl.py

Converte il subset italiano di MultiCoNER v2 (Hugging Face) nel formato
JSONL richiesto da build_base_it.py, applicando una mappatura dalle 33
categorie fine-grained di MultiCoNER verso le 155 foglie di DynamicNER.

Uso:
    python3 multiconer_to_jsonl.py DynamicNER.json output_dir/

Richiede: pip install datasets
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Mappatura MultiCoNER (33 fine-grained) -> DynamicNER leaf (155 categorie)
# ---------------------------------------------------------------------------
# ATTENZIONE: questa è una mappatura DI PARTENZA, non definitiva.
# Diverse voci sono ambigue per natura (una categoria MultiCoNER può
# corrispondere a più foglie DynamicNER a seconda del contesto specifico
# della frase) — qui viene scelta l'approssimazione più plausibile in
# assenza di contesto aggiuntivo. Rivedi manualmente le voci commentate
# come "approssimato" prima di usare il corpus per training serio.
MULTICONER_TO_DYNAMICNER = {
    # --- PER ---
    "ARTIST":         "artist",
    "ATHLETE":        "athlete",
    "CLERIC":         "other person",       # approssimato: non esiste "cleric" in DynamicNER
    "POLITICIAN":     "politician",
    "SCIENTIST":      "scholar",
    "SPORTSMANAGER":  "other person",       # approssimato
    "OTHERPER":       "other person",

    # --- LOC ---
    "FACILITY":       "other facility",     # approssimato: da raffinare (public/commercial/transport/production)
    "HUMANSETTLEMENT":"city",               # approssimato: include anche paesi/villaggi, non solo città
    "STATION":        "other transportation facility",
    "OTHERLOC":       "other geographical entity",

    # --- GRP ---
    "AEROSPACEMANUFACTURER": "other commercial organization",
    "CARMANUFACTURER":       "other commercial organization",
    "MUSICALGRP":            "band",
    "ORG":                   "other non-commercial organization",  # approssimato: ORG è generico in MultiCoNER
    "PRIVATECORP":           "company",
    "PUBLICCORP":            "company",
    "OTHERCORP":             "company",       # categoria extra non documentata nel paper, presente nello schema reale HF
    "TECHCORP":              "company",       # categoria extra non documentata nel paper, presente nello schema reale HF
    "SPORTSGRP":             "sports team/league",

    # --- PROD ---
    "CLOTHING":       "clothes",
    "DRINK":          "beverages",
    "FOOD":           "other food",
    "VEHICLE":        "other vehicle",      # approssimato: da raffinare (air/car/water/rail/bike)
    "OTHERPROD":      "other product",

    # --- CW (Creative Work) ---
    "ARTWORK":        "other visual art",
    "MUSICALWORK":    "song",               # approssimato: potrebbe essere anche "album"
    "SOFTWARE":       "software",
    "VISUALWORK":     "film",               # approssimato: VisualWork copre film/TV/programmi
    "WRITTENWORK":    "fiction",            # approssimato: potrebbe essere "non-fiction" o "poem"
    "OTHERCW":        "other art",          # categoria extra non documentata nel paper, presente nello schema reale HF

    # --- MED ---
    "ANATOMICALSTRUCTURE": "other biological entity",
    "DISEASE":             "disease",
    "MEDICALPROCEDURE":    "other medical entity",
    "MEDICATION/VACCINE":  "medication",
    "SYMPTOM":             "symptom",
}


def load_valid_leaf_categories(hierarchy_path: Path) -> set:
    with open(hierarchy_path, encoding="utf-8") as f:
        hierarchy = json.load(f)
    valid = set()
    for csv_string in hierarchy["third-level"].values():
        for cat in csv_string.split(","):
            valid.add(cat.strip())
    return valid


def get_ner_tag_names(dataset):
    """Gestisce sia ner_tags come ClassLabel (interi) sia come stringhe dirette."""
    feature = dataset.features["ner_tags"].feature
    if hasattr(feature, "names"):
        return feature.names  # es. ['O', 'B-Facility', 'I-Facility', ...]
    return None  # i tag sono già stringhe nel dataset


def extract_entities_from_bio(tokens, tags):
    """
    Ricostruisce gli span di entità da una sequenza BIO.
    Restituisce lista di (entity_text, fine_grained_category).
    """
    entities = []
    current_tokens = []
    current_category = None

    def flush():
        if current_tokens and current_category:
            entities.append((" ".join(current_tokens), current_category))

    for token, tag in zip(tokens, tags):
        if tag == "O" or tag == "":
            flush()
            current_tokens, current_category = [], None
        elif tag.startswith("B-"):
            flush()
            current_tokens = [token]
            current_category = tag[2:]
        elif tag.startswith("I-"):
            if current_category == tag[2:]:
                current_tokens.append(token)
            else:
                # I- senza B- coerente: tratta come nuovo inizio (dato rumoroso)
                flush()
                current_tokens = [token]
                current_category = tag[2:]
    flush()
    return entities


def convert_split(dataset_split, tag_names, valid_categories: set):
    records = []
    unmapped_categories = {}
    skipped_no_entities = 0

    for example in dataset_split:
        tokens = example["tokens"]
        raw_tags = example["ner_tags"]

        if tag_names is not None:
            tags = [tag_names[t] for t in raw_tags]
        else:
            tags = raw_tags

        raw_entities = extract_entities_from_bio(tokens, tags)
        if not raw_entities:
            skipped_no_entities += 1
            continue

        text = " ".join(tokens)
        entities_out = []

        for entity_text, mc_category in raw_entities:
            mc_category_norm = mc_category.upper()
            dn_category = MULTICONER_TO_DYNAMICNER.get(mc_category_norm)

            if dn_category is None:
                unmapped_categories[mc_category_norm] = unmapped_categories.get(mc_category_norm, 0) + 1
                continue

            if dn_category not in valid_categories:
                # Difesa extra: se la tabella sopra viene modificata a mano
                # e si introduce un typo, non silenziamo l'errore.
                print(f"[ATTENZIONE] '{dn_category}' non è una foglia valida "
                      f"in DynamicNER.json — controlla la mappatura per "
                      f"'{mc_category_norm}'", file=sys.stderr)
                continue

            entities_out.append({
                "text": entity_text,
                "start": 0,  # non usato da build_base_it.py
                "end": 0,
                "category": dn_category,
            })

        if entities_out:
            records.append({"text": text, "entities": entities_out})
        else:
            skipped_no_entities += 1

    return records, unmapped_categories, skipped_no_entities


def write_jsonl(records, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    if len(sys.argv) != 3:
        print("Uso: python3 multiconer_to_jsonl.py <DynamicNER.json> <output_dir>",
              file=sys.stderr)
        sys.exit(1)

    hierarchy_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_categories = load_valid_leaf_categories(hierarchy_path)

    print("Scaricamento/caricamento subset italiano di MultiCoNER v2...", file=sys.stderr)
    dataset = load_dataset(
        "MultiCoNER/multiconer_v2",
        "Italian (IT)",
        trust_remote_code=True,
    )

    all_unmapped = {}

    # Il dataset HF di solito espone split 'train' / 'validation' / 'test'.
    # Mappiamo 'validation' -> 'dev' per coerenza con la struttura DynamicNER.
    split_name_map = {"train": "train", "validation": "dev", "test": "test"}

    for hf_split, dn_split in split_name_map.items():
        if hf_split not in dataset:
            print(f"[SKIP] split '{hf_split}' non presente nel dataset", file=sys.stderr)
            continue

        split_data = dataset[hf_split]
        tag_names = get_ner_tag_names(split_data)

        records, unmapped, skipped = convert_split(split_data, tag_names, valid_categories)

        for cat, count in unmapped.items():
            all_unmapped[cat] = all_unmapped.get(cat, 0) + count

        output_path = output_dir / f"corpus_it_{dn_split}.jsonl"
        write_jsonl(records, output_path)
        print(f"{hf_split} -> {output_path}: {len(records)} frasi con entità, "
              f"{skipped} scartate (nessuna entità mappabile)", file=sys.stderr)

    if all_unmapped:
        print("\nCategorie MultiCoNER incontrate ma NON mappate "
              "(controlla se sono typo o davvero mancanti nella tabella):",
              file=sys.stderr)
        for cat, count in sorted(all_unmapped.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count} occorrenze", file=sys.stderr)


if __name__ == "__main__":
    main()