#!/usr/bin/env python3
"""
generate_synthetic.py

Genera frasi sintetiche in italiano per popolare le foglie di terzo livello
di DynamicNER.json che risultano scoperte (o "thin") nei corpus reali,
usando l'API Anthropic (Claude) con output JSON strutturato.

Si integra con:
  - check_coverage.py       -> stessa logica di lettura gerarchia/leaf
  - multiconer_to_jsonl.py  -> stesso formato di output {"text","entities":[{"category":...}]}

Uso:
    export GEMINI_API_KEY="AIza..."
    python3 generate_synthetic.py DynamicNER.json output_it/ \
        --per-category 15 \
        --model gemini-2.5-flash \
        --only-missing \
        --existing output_it/corpus_it_train.jsonl output_it/corpus_it_dev.jsonl output_it/corpus_it_test.jsonl

Note:
  - Richiede: pip install google-genai
  - Il file di output NON viene mai mischiato ai corpus reali: viene scritto
    separatamente come corpus_it_synthetic.jsonl, con un campo "source":"synthetic"
    per ogni record, cosi' resta sempre distinguibile a valle.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[ERRORE] Manca il pacchetto 'google-genai'. Installa con: "
          "pip install google-genai", file=sys.stderr)
    sys.exit(1)

try:
    from dotenv import load_dotenv
    # Cerca un file .env nella stessa cartella dello script (o nelle
    # cartelle superiori) e carica le variabili al suo interno, es.
    # GEMINI_API_KEY=... Se il file non esiste, non fa nulla.
    load_dotenv()
except ImportError:
    # python-dotenv è opzionale: se manca, la GEMINI_API_KEY deve essere
    # gia' presente nell'ambiente (es. esportata a mano nel terminale).
    pass


# ---------------------------------------------------------------------------
# Definizioni/esempi brevi per foglia, usate per guidare l'LLM.
# Non serve coprire tutte le 155 (se manca una definizione, l'LLM se la
# ricava dal nome della categoria e dal nome del gruppo second-level),
# ma per le foglie piu' ambigue conviene darla esplicita.
# Aggiungi/correggi voci qui quando noti generazioni di scarsa qualita'.
# ---------------------------------------------------------------------------
LEAF_HINTS = {
    "firearms": "armi da fuoco (pistola, fucile, mitragliatrice)",
    "biological weapon": "armi biologiche (agenti patogeni usati come arma)",
    "chemical weapon": "armi chimiche (gas nervino, agenti chimici bellici)",
    "explosives": "ordigni esplosivi (bomba, mina, granata, tritolo)",
    "cold weapon": "armi bianche (spada, coltello, lancia, ascia)",
    "nuclear": "armi nucleari (testata nucleare, bomba atomica)",
    "other weapon": "armi che non rientrano nelle categorie precedenti",

    "address": "indirizzo completo (via, numero, citta')",
    "road": "nome di una strada o via specifica",
    "railway": "nome di una linea o tratta ferroviaria",
    "other address": "riferimenti di indirizzo non altrimenti classificabili",

    "literary award": "premio letterario (es. premio Strega, Nobel per la letteratura)",
    "sports award": "premio sportivo (es. Pallone d'Oro)",
    "artistic award": "premio artistico/cinematografico (es. Oscar, Grammy)",
    "other award": "premio non altrimenti classificabile",

    "protein": "nome di una proteina specifica (es. emoglobina, insulina)",
    "species": "nome di una specie biologica (es. Homo sapiens, Panthera leo)",
    "biological theory": "teoria biologica (es. teoria dell'evoluzione)",
    "other biological entity": "entita' biologica non altrimenti classificabile",

    "element": "elemento chimico (es. ossigeno, ferro, uranio)",
    "compound": "composto chimico (es. acqua, cloruro di sodio)",
    "reaction": "reazione chimica specifica (es. fotosintesi, combustione)",
    "chemical theory": "teoria chimica (es. teoria degli orbitali molecolari)",
    "other chemical entity": "entita' chimica non altrimenti classificabile",

    "hospital": "nome di un ospedale specifico",
    "library": "nome di una biblioteca specifica",
    "park": "nome di un parco pubblico specifico",
    "landmark": "monumento o luogo di interesse specifico",
    "school": "nome di una scuola specifica",
    "museum": "nome di un museo specifico",
    "sports facility": "impianto sportivo specifico (stadio, palazzetto)",
    "other public facility": "struttura pubblica non altrimenti classificabile",

    "bank": "nome di una banca specifica",
    "hotel": "nome di un hotel specifico",
    "restaurant": "nome di un ristorante specifico",
    "market/mall": "nome di un mercato o centro commerciale specifico",
    "theater/cinema": "nome di un teatro o cinema specifico",
    "other commercial facility": "struttura commerciale non altrimenti classificabile",

    "airport": "nome di un aeroporto specifico",
    "station": "nome di una stazione (treno, metro, bus) specifica",
    "port": "nome di un porto specifico",
    "other transportation facility": "struttura di trasporto non altrimenti classificabile",

    "factory": "nome/riferimento a una fabbrica specifica",
    "farm": "nome/riferimento a un'azienda agricola specifica",
    "mine": "nome/riferimento a una miniera specifica",
    "energy": "impianto energetico (centrale elettrica, nucleare, eolica)",
    "other production facility": "struttura produttiva non altrimenti classificabile",

    "residential": "edificio residenziale specifico (condominio, palazzo)",
    "government facility": "edificio governativo (ministero, comune, tribunale)",
    "other facility": "struttura non altrimenti classificabile",

    "painting": "nome di un quadro/dipinto specifico",
    "sculpture": "nome di una scultura specifica",
    "visual art genre": "corrente/genere artistico visivo (es. cubismo, impressionismo)",
    "other visual art": "opera d'arte visiva non altrimenti classificabile",

    "album": "titolo di un album musicale",
    "music genre": "genere musicale (es. jazz, rock, musica classica)",
    "other music": "opera musicale non altrimenti classificabile",

    "poem": "titolo di una poesia specifica",
    "non-fiction": "titolo di un'opera di saggistica/non-fiction",
    "literature genre": "genere letterario (es. romanzo giallo, fantascienza)",
    "other literature": "opera letteraria non altrimenti classificabile",

    "broadcast program": "programma televisivo o radiofonico specifico",
    "game": "titolo di un videogioco o gioco da tavolo",
    "play": "titolo di un'opera teatrale",
    "other art": "opera artistica non altrimenti classificabile",

    "ethnic group": "gruppo etnico (es. Berberi, Maori)",
    "religious group": "gruppo religioso (es. Cattolici, Sunniti)",
    "other social group": "gruppo sociale non altrimenti classificabile",

    "educational and research": "istituzione educativa/di ricerca (universita', centro di ricerca)",
    "political/military": "organizzazione politica o militare (partito, alleanza militare)",
    "community": "organizzazione comunitaria/associazione locale",
    "religious organization": "organizzazione religiosa (chiesa, ordine religioso)",
    "other non-commercial organization": "organizzazione non-profit non altrimenti classificabile",

    "media": "testata giornalistica, canale TV o media specifico",

    "business event": "evento economico/aziendale (fusione, IPO, conferenza aziendale)",
    "disaster": "disastro naturale o incidente (terremoto, alluvione, incidente aereo)",
    "political/military event": "evento politico o militare (elezione, battaglia, trattato)",
    "sporting event": "evento sportivo (olimpiadi, campionato, partita specifica)",
    "other event": "evento non altrimenti classificabile",

    "educational degree": "titolo di studio (laurea in ingegneria, dottorato)",
    "tradition": "tradizione o usanza culturale specifica",
    "god": "divinita' (es. Zeus, Odino, Ra)",
    "law": "legge o normativa specifica",
    "language": "lingua (es. italiano, mandarino, swahili)",
    "miscellaneous": "entita' varia non altrimenti classificabile",

    "algorithm": "algoritmo specifico (es. quicksort, PageRank)",
    "programlang": "linguaggio di programmazione (es. Python, Java)",
    "other computer science entity": "entita' informatica non altrimenti classificabile",

    "injury": "tipo di lesione/trauma (es. frattura, contusione)",
    "medical theory": "teoria medica specifica",

    "household": "oggetto per la casa (es. aspirapolvere, frullatore)",
    "musical instruments": "strumento musicale (es. chitarra, violino)",
    "personal care": "prodotto per la cura personale (es. shampoo, rasoio)",
    "toys": "giocattolo specifico",

    "packaged foods": "alimento confezionato/marchio specifico (es. Nutella, Kinder)",

    "academic journal": "rivista scientifica specifica (es. Nature, The Lancet)",
    "conference": "conferenza accademica specifica",
    "discipline": "disciplina scientifica (es. fisica quantistica, virologia)",
    "metrics": "metrica/indice scientifico (es. impact factor, indice di Gini)",
    "other scientific entity": "entita' scientifica non altrimenti classificabile",

    "astronomical object": "corpo celeste (es. Marte, Via Lattea, una stella specifica)",
    "physical phenomenon": "fenomeno fisico (es. gravita', effetto fotoelettrico)",
    "physical theory": "teoria fisica (es. relativita', meccanica quantistica)",
    "other physical entity": "entita' fisica non altrimenti classificabile",

    "continent": "continente (es. Europa, Asia)",
    "country": "nazione/stato",
    "state or province": "stato federato o provincia",
    "city": "citta'",
    "district": "distretto/quartiere",
    "region": "regione geografica/amministrativa",
    "other gpe": "entita' geo-politica non altrimenti classificabile",

    "water body": "corpo d'acqua (fiume, lago, mare, oceano)",
    "mountain": "montagna specifica",
    "island": "isola specifica",
    "desert": "deserto specifico",
    "other geographical entity": "entita' geografica non altrimenti classificabile",

    "mythological figure": "figura mitologica (es. Ercole, Thor come personaggio, non divinita' venerata)",
    "other figure": "figura fittizia non altrimenti classificabile",

    "actor": "attore/attrice",
    "author": "scrittore/scrittrice",
    "business executive": "dirigente/CEO di un'azienda",
    "director": "regista",
    "military": "figura militare (generale, comandante)",
    "musician": "musicista",

    "air": "veicolo aereo (aereo, elicottero)",
    "car": "automobile (marca/modello specifico)",
    "water": "veicolo acquatico (nave, barca, sottomarino)",
    "rail": "veicolo ferroviario (treno specifico)",
    "bike": "bicicletta o motocicletta",

    "electronics": "dispositivo elettronico (smartphone, TV, laptop)",
    "ai": "sistema/modello di intelligenza artificiale specifico",
    "other technology": "tecnologia non altrimenti classificabile",
    "website": "sito web specifico",
}


def load_hierarchy(hierarchy_path: Path):
    with open(hierarchy_path, encoding="utf-8") as f:
        hierarchy = json.load(f)

    leaf_to_parent = {}
    parent_to_leaves = {}
    for parent, csv_string in hierarchy["third-level"].items():
        leaves = [leaf.strip() for leaf in csv_string.split(",")]
        parent_to_leaves[parent] = leaves
        for leaf in leaves:
            leaf_to_parent[leaf] = parent

    all_leaves = set(leaf_to_parent.keys())
    return all_leaves, leaf_to_parent, parent_to_leaves


def load_existing_categories(paths) -> Counter:
    """Stessa logica di detect_format_and_extract_categories in check_coverage.py,
    ridotta al solo formato .jsonl (quello prodotto dalla pipeline reale)."""
    counter = Counter()
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"[WARN] file non trovato, ignorato: {p}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for ent in rec.get("entities", []):
                    counter[ent["category"]] += 1
    return counter


def build_batches(parent_to_leaves: dict, leaves_to_cover: set, max_leaves_per_batch: int = 8):
    """
    Raggruppa le foglie da generare per macro-categoria (second-level),
    come suggerito nel workflow di check_coverage.py. Se un gruppo ha
    troppe foglie, lo spezza in piu' batch per non sovraccaricare il prompt.
    """
    batches = []
    for parent, leaves in parent_to_leaves.items():
        leaves_needed = [l for l in leaves if l in leaves_to_cover]
        if not leaves_needed:
            continue
        for i in range(0, len(leaves_needed), max_leaves_per_batch):
            chunk = leaves_needed[i:i + max_leaves_per_batch]
            batches.append((parent, chunk))
    return batches


def build_prompt(parent: str, leaves: list, per_category: int) -> str:
    lines = [
        f"Genera frasi in italiano naturale (stile notizie, enciclopedico o "
        f"narrativo, varia il registro) che contengano entita' della "
        f"macro-categoria '{parent}'.",
        "",
        "Categorie specifiche da coprire in questo batch:",
    ]
    for leaf in leaves:
        hint = LEAF_HINTS.get(leaf, "")
        if hint:
            lines.append(f'- "{leaf}": {hint}')
        else:
            lines.append(f'- "{leaf}"')

    lines += [
        "",
        f"Genera esattamente {per_category} frasi per ciascuna categoria elencata "
        f"(quindi {per_category * len(leaves)} frasi totali in questo batch).",
        "Regole:",
        "- Varia soggetto, tempo verbale, lunghezza e struttura sintattica delle frasi.",
        "- Ogni frase deve contenere ESATTAMENTE UNA entita' della categoria target "
        "(evita di introdurre per sbaglio entita' di altre categorie nella stessa frase, "
        "a meno che sia naturale e in tal caso etichettale comunque tutte).",
        "- Il campo \"text\" dell'entita' deve comparire ESATTAMENTE (stessa forma, "
        "stesse maiuscole/minuscole) all'interno del campo \"text\" della frase.",
        "- Non ripetere la stessa entita' piu' di 2-3 volte nell'intero batch.",
        "- Non inventare markdown, commenti o testo fuori dal JSON.",
        "",
        "Rispondi SOLO con un array JSON valido, nessun altro testo, in questo formato:",
        '[{"text": "<frase completa>", "entities": [{"text": "<entita\'>", '
        '"category": "<nome categoria esatto come sopra>"}]}]',
    ]
    return "\n".join(lines)


def call_llm(client, model: str, prompt: str, max_tokens: int = 32768, retries: int = 3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    # Chiediamo direttamente output JSON: Gemini forza la
                    # risposta a essere JSON valido, eliminando il problema
                    # dei fence markdown che si aveva con altri provider.
                    response_mime_type="application/json",
                    safety_settings=[
                        types.SafetySetting(
                            category=cat,
                            threshold="BLOCK_ONLY_HIGH",
                        )
                        for cat in (
                            "HARM_CATEGORY_HARASSMENT",
                            "HARM_CATEGORY_HATE_SPEECH",
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "HARM_CATEGORY_DANGEROUS_CONTENT",
                        )
                    ],
                ),
            )

            # Diagnostica: se la risposta e' vuota, capiamo se e' un blocco
            # di sicurezza (frequente per categorie come "weapon") o un
            # troncamento per limite di token, invece di un generico
            # errore JSON poco informativo.
            if not response.candidates:
                feedback = getattr(response, "prompt_feedback", None)
                raise RuntimeError(
                    f"Nessuna risposta generata (prompt_feedback: {feedback})"
                )

            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            raw = (response.text or "").strip()

            if not raw:
                safety_ratings = getattr(candidate, "safety_ratings", None)
                raise RuntimeError(
                    f"Risposta vuota (finish_reason: {finish_reason}, "
                    f"safety_ratings: {safety_ratings})"
                )

            if str(finish_reason) not in ("STOP", "FinishReason.STOP", "1"):
                print(f"[WARN] finish_reason inatteso: {finish_reason} "
                      f"(possibile troncamento o blocco parziale)", file=sys.stderr)

            # Difesa extra, nel caso in cui il modello aggiunga comunque
            # fence markdown nonostante response_mime_type.
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
        except Exception as e:  # errori di rete/rate limit/safety block
            last_err = e
            print(f"[WARN] Errore chiamata API (tentativo {attempt}/{retries}): {e}",
                  file=sys.stderr)
            time.sleep(5 * attempt)

    print(f"[ERRORE] Batch fallito dopo {retries} tentativi: {last_err}",
          file=sys.stderr)
    return None


def validate_and_filter(records: list, expected_leaves: set, all_valid_leaves: set):
    """
    Valida ogni record generato:
      - la categoria deve essere una foglia valida di DynamicNER (difesa
        contro invenzioni del modello, stessa logica di multiconer_to_jsonl.py)
      - il testo dell'entita' deve comparire nel testo della frase
    Restituisce (records_validi, stats) dove stats conta scarti per motivo.
    """
    clean_records = []
    stats = Counter()

    for rec in records:
        text = rec.get("text", "")
        entities = rec.get("entities", [])
        if not text or not entities:
            stats["record_vuoto"] += 1
            continue

        good_entities = []
        for ent in entities:
            ent_text = ent.get("text", "")
            category = ent.get("category", "")

            if category not in all_valid_leaves:
                stats["categoria_non_valida"] += 1
                print(f"[ATTENZIONE] categoria inventata/non valida: '{category}' "
                      f"(frase: '{text[:60]}...')", file=sys.stderr)
                continue

            if ent_text not in text:
                stats["span_non_trovato"] += 1
                print(f"[ATTENZIONE] entita' '{ent_text}' non presente "
                      f"letteralmente nel testo: '{text[:60]}...'", file=sys.stderr)
                continue

            good_entities.append({
                "text": ent_text,
                "start": 0,
                "end": 0,
                "category": category,
            })

        if good_entities:
            clean_records.append({
                "text": text,
                "entities": good_entities,
                "source": "synthetic",
            })
        else:
            stats["nessuna_entita_valida"] += 1

    return clean_records, stats


def write_jsonl(records: list, output_path: Path, mode="w"):
    with open(output_path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hierarchy", type=Path, help="Path a DynamicNER.json")
    parser.add_argument("output_dir", type=Path, help="Cartella di output")
    parser.add_argument("--existing", nargs="*", default=[],
                         help="File .jsonl reali già esistenti, per capire cosa è "
                              "già coperto (es. corpus_it_train.jsonl corpus_it_dev.jsonl)")
    parser.add_argument("--per-category", type=int, default=15,
                         help="Numero di frasi da generare per ciascuna foglia mancante")
    parser.add_argument("--thin-threshold", type=int, default=0,
                         help="Se >0, genera anche per le foglie presenti ma con meno "
                              "occorrenze di questa soglia (stesso concetto di "
                              "check_coverage.py). Default 0 = solo foglie del tutto assenti.")
    parser.add_argument("--model", default="gemini-3.6-flash",
                         help="Modello Gemini da usare (es. gemini-3.6-flash, "
                              "gemini-3.5-flash-lite, gemini-2.5-pro). "
                              "Nota: gemini-2.5-flash non e' piu' disponibile "
                              "per i nuovi account/chiavi API.")
    parser.add_argument("--max-leaves-per-batch", type=int, default=8,
                         help="Numero massimo di foglie per chiamata API")
    parser.add_argument("--max-output-tokens", type=int, default=32768,
                         help="Tetto massimo di token in output per chiamata "
                              "(default 32768). Alzalo se vedi 'FinishReason.MAX_TOKENS' "
                              "e JSON troncati/non validi con batch grandi; oppure "
                              "riduci --max-leaves-per-batch per rimpicciolire i batch.")
    parser.add_argument("--only-groups", nargs="*", default=None,
                         help="Se specificato, limita la generazione solo a questi "
                              "gruppi second-level (es. --only-groups weapon award)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Costruisce i batch e i prompt ma non chiama l'API "
                              "(utile per controllare cosa verrebbe generato)")
    parser.add_argument("--resume", action="store_true",
                         help="Non sovrascrive corpus_it_synthetic.jsonl: legge le "
                              "foglie gia' generate in run precedenti (nello stesso "
                              "file synthetic) e continua ad accodare da li'. Utile "
                              "per cambiare modello/API key a meta' quando si esaurisce "
                              "la quota, senza perdere il lavoro gia' fatto.")
    args = parser.parse_args()

    all_leaves, leaf_to_parent, parent_to_leaves = load_hierarchy(args.hierarchy)
    existing_counts = load_existing_categories(args.existing)

    output_path = args.output_dir / "corpus_it_synthetic.jsonl"
    if args.resume and output_path.exists():
        synthetic_counts = load_existing_categories([output_path])
        print(f"[RESUME] Trovate {sum(synthetic_counts.values())} entita' gia' "
              f"generate in {output_path} su {len(synthetic_counts)} foglie diverse: "
              f"non verranno rigenerate.", file=sys.stderr)
        existing_counts.update(synthetic_counts)

    if args.thin_threshold > 0:
        leaves_to_cover = {
            leaf for leaf in all_leaves
            if existing_counts.get(leaf, 0) < args.thin_threshold
        }
    else:
        leaves_to_cover = all_leaves - set(existing_counts.keys())

    if args.only_groups:
        allowed_parents = set(args.only_groups)
        leaves_to_cover = {
            leaf for leaf in leaves_to_cover
            if leaf_to_parent[leaf] in allowed_parents
        }

    if not leaves_to_cover:
        print("Nessuna foglia da generare: copertura già completa per i criteri scelti.",
              file=sys.stderr)
        return

    batches = build_batches(parent_to_leaves, leaves_to_cover, args.max_leaves_per_batch)
    print(f"Foglie da generare: {len(leaves_to_cover)} in {len(batches)} batch",
          file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        # Sovrascrive a inizio run, poi appende batch per batch (così se il
        # processo si interrompe a metà non perdi il lavoro già fatto).
        write_jsonl([], output_path, mode="w")
    elif not output_path.exists():
        # --resume ma il file non esiste ancora: e' la prima run, si crea vuoto.
        write_jsonl([], output_path, mode="w")
    # altrimenti (--resume + file esistente): non tocchiamo il file, si continua
    # ad accodare sotto con mode="a" nel ciclo dei batch.

    client = None
    if not args.dry_run:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[ERRORE] Variabile d'ambiente GEMINI_API_KEY non impostata.",
                  file=sys.stderr)
            sys.exit(1)
        client = genai.Client(api_key=api_key)

    total_written = 0
    total_stats = Counter()

    for batch_num, (parent, leaves) in enumerate(batches, start=1):
        prompt = build_prompt(parent, leaves, args.per_category)
        print(f"\n[Batch {batch_num}/{len(batches)}] gruppo='{parent}' "
              f"foglie={leaves}", file=sys.stderr)

        if args.dry_run:
            print(prompt, file=sys.stderr)
            print("--- (dry-run: nessuna chiamata API effettuata) ---",
                  file=sys.stderr)
            continue

        raw_records = call_llm(client, args.model, prompt,
                                max_tokens=args.max_output_tokens)
        if raw_records is None:
            continue

        clean_records, stats = validate_and_filter(raw_records, set(leaves), all_leaves)
        total_stats.update(stats)

        write_jsonl(clean_records, output_path, mode="a")
        total_written += len(clean_records)

        print(f"  -> {len(clean_records)}/{len(raw_records)} record validi scritti "
              f"(scarti: {dict(stats)})", file=sys.stderr)

        time.sleep(1)  # piccolo cuscinetto anti rate-limit

    print(f"\n=== RIEPILOGO GENERAZIONE ===", file=sys.stderr)
    print(f"Record sintetici scritti: {total_written}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)
    if total_stats:
        print(f"Scarti totali per motivo: {dict(total_stats)}", file=sys.stderr)
    print(f"\nRilancia check_coverage.py includendo anche questo file per "
          f"verificare la nuova copertura, es.:", file=sys.stderr)
    print(f"  python3 check_coverage.py {args.hierarchy} "
          f"{' '.join(args.existing)} {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()