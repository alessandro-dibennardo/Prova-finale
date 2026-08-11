#!/usr/bin/env python3
"""
generate_synonyms.py

Genera il file dynamic.txt richiesto da dynamic2.py: un sinonimo/variante
per ciascuna delle 155 foglie di terzo livello (e, opzionalmente, delle
categorie di primo/secondo livello) di DynamicNER.json.

Formato di output (una riga per categoria):
    categoria---sinonimo

Le categorie restano in INGLESE (coerente con build_base_it.py: il corpus
italiano usa comunque le etichette inglesi), quindi anche i sinonimi
generati qui sono in inglese: servono a dynamic2.py per variare la forma
lessicale delle opzioni mostrate nel prompt di classificazione, non a
tradurre in italiano.

Uso:
    export GEMINI_API_KEY="AIza..."
    python3 generate_synonyms.py DynamicNER.json dynamic.txt \
        --model gemini-2.0-flash

Richiede: pip install google-genai
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("[ERRORE] Manca il pacchetto 'google-genai'. Installa con: "
          "pip install google-genai", file=sys.stderr)
    sys.exit(1)

try:
    from dotenv import load_dotenv
    # Cerca un file .env risalendo dalla cartella dello script (utile se lo
    # script viene lanciato da una sottocartella diversa da quella del .env)
    load_dotenv()
except ImportError:
    print("[INFO] python-dotenv non installato: le variabili d'ambiente "
          "(es. GEMINI_API_KEY) devono essere già impostate nella sessione "
          "corrente, oppure installa con: pip install python-dotenv",
          file=sys.stderr)


def load_all_categories(hierarchy_path: Path) -> List[str]:
    """Raccoglie tutte le categorie di 1°, 2° e 3° livello (uniche),
    perché dynamic2.py può in teoria incontrare una categoria a qualunque
    livello come risposta/opzione."""
    with open(hierarchy_path, encoding="utf-8") as f:
        hierarchy = json.load(f)

    categories = set()
    for c in hierarchy["first-level"].split(","):
        categories.add(c.strip())
    for csv in hierarchy["second-level"].values():
        for c in csv.split(","):
            categories.add(c.strip())
    for csv in hierarchy["third-level"].values():
        for c in csv.split(","):
            categories.add(c.strip())

    return sorted(categories)


def build_prompt(categories_chunk: List[str]) -> str:
    lines = [
        "Per ciascuna delle seguenti categorie NER in inglese, fornisci UN SOLO "
        "sinonimo o variante lessicale equivalente in inglese (non una "
        "traduzione in altra lingua, non una spiegazione: solo un termine o "
        "breve espressione alternativa che un umano userebbe per riferirsi "
        "alla stessa categoria).",
        "",
        "Regole:",
        "- Il sinonimo deve avere significato IDENTICO o quasi identico alla "
        "categoria originale (non un iperonimo/iponimo, non una categoria "
        "diversa della stessa gerarchia).",
        "- Se la categoria è già molto generica o non ha un sinonimo naturale "
        "(es. 'other person', 'miscellaneous'), fornisci comunque una "
        "variante lessicale plausibile (es. 'other person' -> 'unspecified "
        "person'), non lasciare vuoto.",
        "- Mantieni lo stesso registro (minuscolo, senza articoli).",
        "",
        "Categorie:",
    ]
    for cat in categories_chunk:
        lines.append(f"- {cat}")

    lines += [
        "",
        "Rispondi SOLO con un array JSON di oggetti, nessun altro testo, "
        "in questo formato esatto:",
        '[{"category": "<categoria esatta come sopra>", "synonym": "<sinonimo>"}]',
    ]
    return "\n".join(lines)


def call_llm(client, model: str, prompt: str, retries: int = 4):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=4096,
                    temperature=0.7,
                ),
            )
            raw = (response.text or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)

        except json.JSONDecodeError as e:
            last_err = e
            print(f"[WARN] JSON non valido (tentativo {attempt}/{retries}): {e}",
                  file=sys.stderr)
            time.sleep(2 * attempt)
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 20 * attempt
                print(f"[WARN] Rate limit (tentativo {attempt}/{retries}), "
                      f"attendo {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"[WARN] Errore API (tentativo {attempt}/{retries}): {e}",
                      file=sys.stderr)
                time.sleep(5 * attempt)

    print(f"[ERRORE] Chunk fallito dopo {retries} tentativi: {last_err}",
          file=sys.stderr)
    return None


def normalize_synonym(syn: str) -> str:
    """Pulizia minima: minuscolo, senza punteggiatura finale, spazi singoli."""
    syn = syn.strip().lower()
    syn = re.sub(r'\s+', ' ', syn)
    syn = syn.strip('.').strip()
    return syn


def chunked(lst: List[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hierarchy", type=Path, help="Path a DynamicNER.json")
    parser.add_argument("output", type=Path, help="Path del dynamic.txt da creare")
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--chunk-size", type=int, default=25,
                         help="Numero di categorie per chiamata API")
    parser.add_argument("--dry-run", action="store_true",
                         help="Mostra i chunk/prompt senza chiamare l'API")
    args = parser.parse_args()

    categories = load_all_categories(args.hierarchy)
    print(f"Categorie totali da coprire: {len(categories)}", file=sys.stderr)

    chunks = list(chunked(categories, args.chunk_size))
    print(f"Divise in {len(chunks)} batch da {args.chunk_size}", file=sys.stderr)

    client = None if args.dry_run else genai.Client()

    all_pairs: Dict[str, str] = {}
    failed_categories: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        prompt = build_prompt(chunk)
        print(f"\n[Batch {i}/{len(chunks)}] {len(chunk)} categorie", file=sys.stderr)

        if args.dry_run:
            print(prompt, file=sys.stderr)
            continue

        result = call_llm(client, args.model, prompt)
        if result is None:
            failed_categories.extend(chunk)
            continue

        found_in_chunk = set()
        for entry in result:
            cat = entry.get("category", "").strip()
            syn = normalize_synonym(entry.get("synonym", ""))
            if not cat or not syn:
                continue
            if cat not in chunk:
                # Difesa: il modello a volte "corregge" leggermente il nome
                # della categoria. Se non matcha esattamente, scartiamo
                # piuttosto che inserire una chiave sbagliata nel dizionario.
                print(f"[WARN] categoria restituita non corrisponde a nessuna "
                      f"richiesta nel batch: '{cat}', scartata", file=sys.stderr)
                continue
            if syn == cat.lower():
                # Sinonimo identico all'originale: inutile, meglio segnalarlo
                print(f"[WARN] sinonimo identico alla categoria per '{cat}', "
                      f"scartato", file=sys.stderr)
                continue
            all_pairs[cat] = syn
            found_in_chunk.add(cat)

        missing_in_chunk = set(chunk) - found_in_chunk
        if missing_in_chunk:
            failed_categories.extend(sorted(missing_in_chunk))

        time.sleep(1)

    if args.dry_run:
        print("\n(dry-run: nessun file scritto)", file=sys.stderr)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for cat in categories:
            if cat in all_pairs:
                f.write(f"{cat}---{all_pairs[cat]}\n")

    print(f"\n=== RIEPILOGO ===", file=sys.stderr)
    print(f"Sinonimi generati: {len(all_pairs)}/{len(categories)}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)

    if failed_categories:
        print(f"\nCategorie SENZA sinonimo generato ({len(failed_categories)}):",
              file=sys.stderr)
        for cat in sorted(set(failed_categories)):
            print(f"  - {cat}", file=sys.stderr)
        print("\nPuoi rilanciare lo script solo su queste (aggiungendo "
              "manualmente le righe mancanti al file, o rilanciando con "
              "un DynamicNER.json temporaneo contenente solo queste), "
              "oppure completarle a mano nel file dynamic.txt.",
              file=sys.stderr)


if __name__ == "__main__":
    main()