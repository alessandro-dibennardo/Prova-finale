import json
import os
from pathlib import Path


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def mark_entities_by_position(sentence, entities):
    """Marca le entita' nella frase inserendo ##...## per POSIZIONE, non con
    .replace() sequenziale.

    Perche' non usare .replace(): se un'entita' e' sottostringa di un'altra
    (es. "Guangzhou" dentro "restaurant Guangzhou Taotaoju"), .replace()
    ri-cerca il testo anche DENTRO ai marcatori ## appena inseriti per
    l'entita' piu' lunga, producendo marcature annidate rotte tipo
    "##restaurant ##Guangzhou## Taotaoju##". Calcolando prima le posizioni
    non sovrapposte nel testo originale ed inserendo i marcatori per indice
    (dal fondo verso l'inizio, per non invalidare gli indici successivi),
    il problema non si presenta.
    """
    # Entita' piu' lunghe prima: se due entita' si sovrappongono, vince quella
    # piu' lunga/specifica.
    sorted_entities = sorted(set(entities), key=len, reverse=True)

    occupied = []  # lista di (start, end) gia' assegnati, per controllare overlap
    spans = []     # (start, end) da marcare

    for entity in sorted_entities:
        search_from = 0
        while True:
            idx = sentence.find(entity, search_from)
            if idx == -1:
                break
            end = idx + len(entity)
            overlaps = any(idx < o_end and end > o_start for o_start, o_end in occupied)
            if not overlaps:
                occupied.append((idx, end))
                spans.append((idx, end))
            search_from = idx + 1

    # Inserisci i marcatori dal fondo della frase verso l'inizio, cosi' gli
    # indici delle marcature precedenti non vengono invalidati.
    spans.sort(key=lambda s: s[0], reverse=True)
    marked = sentence
    for start, end in spans:
        marked = f"{marked[:start]}##{marked[start:end]}##{marked[end:]}"
    return marked


def generate_conversations(json_data, max_sentences=None):
    """Generatore: produce una conversazione alla volta invece di accumulare
    tutto in una lista in RAM."""
    n = 0
    for sentence_id, sentence_info in json_data.items():
        if max_sentences is not None and n >= max_sentences:
            break
        n += 1

        sentence = sentence_info['sentence']
        entities = sentence_info['entity']
        marked_sentence = mark_entities_by_position(sentence, entities)

        yield {
            "conversations": [
                {"from": "user", "value": sentence},
                {"from": "assistant", "value": marked_sentence},
            ]
        }


def write_jsonl_streaming(conversations_iter, output_file):
    """Scrive una conversazione per riga (JSONL), senza tenere l'intero
    dataset in memoria."""
    count = 0
    with open(output_file, 'w', encoding='utf-8') as file:
        for conv in conversations_iter:
            file.write(json.dumps(conv, ensure_ascii=False))
            file.write('\n')
            count += 1
    return count


def process_file(input_file, output_file, max_sentences=None):
    json_data = load_json_file(input_file)
    n_input = len(json_data)
    conversations_iter = generate_conversations(json_data, max_sentences=max_sentences)
    n_written = write_jsonl_streaming(conversations_iter, output_file)

    limited_note = ""
    if max_sentences is not None and n_input > max_sentences:
        limited_note = f" [LIMITATO: processate solo {max_sentences}/{n_input} frasi in input]"
    print(f"Conversion complete: {n_written} conversazioni scritte in {output_file}.{limited_note}")


def main(input_directory, output_directory, max_sentences=None):
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            input_file = os.path.join(input_directory, filename)
            # Output in .jsonl (streaming), non piu' .json (accumulo in RAM)
            output_file = os.path.join(output_directory, f"{Path(filename).stem}.jsonl")
            process_file(input_file, output_file, max_sentences=max_sentences)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Converte i file BASE/*.json in formato conversazionale "
                     "(stage1: estrazione entita' marcate con ##...##). "
                     "Output in formato JSONL, scritto in streaming per non "
                     "saturare la RAM su file grandi.")
    parser.add_argument("input_directory", help="Cartella con i file BASE (es. base/it/BASE)")
    parser.add_argument("output_directory", help="Cartella di output (es. base/it/SWIFT/extract)")
    parser.add_argument("--max-sentences", type=int, default=None,
                         help="Limita il numero di frasi processate PER FILE "
                              "(utile per file enormi come test.json). Se non "
                              "specificato, processa tutte le frasi.")
    args = parser.parse_args()
    main(args.input_directory, args.output_directory, args.max_sentences)