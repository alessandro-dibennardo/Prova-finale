import json
import os
from pathlib import Path

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_category_structure(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def find_categories(category, category_structure):
    for second, third_list in category_structure['third-level'].items():
        if category in third_list.split(', '):
            third_level = category
            second_level = second
            for first, second_list in category_structure['second-level'].items():
                if second in second_list.split(', '):
                    first_level = first
                    return first_level, second_level, third_level
    
    for first, second_list in category_structure['second-level'].items():
        if category in second_list.split(', '):
            first_level = first
            second_level = category
            return first_level, second_level, category
    
    if category in category_structure['first-level'].split(', '):
        return category, category, category
    
    return 'miscellaneous', 'miscellaneous', 'miscellaneous'

def get_entity_list(level, category, category_structure):
    if level == 'first-level':
        return category_structure['first-level']
    elif level == 'second-level':
        for first, second_list in category_structure['second-level'].items():
            if category in second_list.split(', '):
                return second_list
        return category_structure['second-level'].get(category, '')
    elif level == 'third-level':
        for second, third_list in category_structure['third-level'].items():
            if category in third_list.split(', '):
                return third_list
    return ''

def generate_conversations(json_data, category_structure, max_sentences=None):
    """Generatore: produce una conversazione alla volta invece di accumulare
    tutto in una lista in RAM (fondamentale per file grandi come test.json
    con centinaia di migliaia di frasi, dato che qui ogni entita' produce
    3 conversazioni, una per livello gerarchico)."""
    n_sentences = 0
    for sentence_id, sentence_info in json_data.items():
        if max_sentences is not None and n_sentences >= max_sentences:
            break
        n_sentences += 1

        sentence = sentence_info['sentence']
        entities = sentence_info['entity']
        categories = sentence_info['category']

        for i, (entity, category) in enumerate(zip(entities, categories)):
            first_level, second_level, third_level = find_categories(category, category_structure)

            for level, category_value in [('first-level', first_level), ('second-level', second_level), ('third-level', third_level)]:
                entity_list = get_entity_list(level, category_value, category_structure)

                # Create a copy of the sentence and highlight the current entity
                highlighted_sentence = sentence
                start_index = 0
                entity_positions = []
                for j, e in enumerate(entities):
                    try:
                        index = highlighted_sentence.index(e, start_index)
                        entity_positions.append((index, index + len(e), j))
                        start_index = index + len(e)
                    except ValueError:
                        continue

                entity_positions.sort(key=lambda x: x[0])

                for start, end, j in reversed(entity_positions):
                    if j == i:
                        highlighted_sentence = f"{highlighted_sentence[:start]}##{highlighted_sentence[start:end]}##{highlighted_sentence[end:]}"

                user_message = f"The ##{entity}## in the sentence: \"{highlighted_sentence}\" belongs to which entity in the list: {entity_list}?"

                yield {
                    "conversations": [
                        {"from": "user", "value": user_message},
                        {"from": "assistant", "value": category_value}
                    ]
                }


def write_jsonl_streaming(conversations_iter, output_file):
    """Scrive una conversazione per riga (JSONL), senza mai tenere l'intero
    dataset in memoria."""
    count = 0
    with open(output_file, 'w', encoding='utf-8') as file:
        for conv in conversations_iter:
            file.write(json.dumps(conv, ensure_ascii=False))
            file.write('\n')
            count += 1
    return count


def process_directory(input_dir, category_structure_file, output_dir, max_sentences=None, limit_file=None):
    category_structure = load_category_structure(category_structure_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            # Applica il limite SOLO al file indicato da --limit-file (se
            # specificato); se --limit-file non e' specificato, il limite si
            # applica a tutti i file (comportamento precedente).
            effective_max = max_sentences if (limit_file is None or filename == limit_file) else None
            # Output in .jsonl (streaming), non piu' .json (accumulo in RAM)
            output_file = os.path.join(output_dir, f"{Path(filename).stem}.jsonl")

            json_data = load_json_file(input_file)
            n_input_sentences = len(json_data)
            conversations_iter = generate_conversations(
                json_data, category_structure, max_sentences=effective_max)
            n_written = write_jsonl_streaming(conversations_iter, output_file)

            limited_note = ""
            if effective_max is not None and n_input_sentences > effective_max:
                limited_note = (f" [LIMITATO: processate solo {effective_max}/"
                                 f"{n_input_sentences} frasi in input]")
            print(f"Processed {filename}: {n_written} conversazioni scritte "
                  f"in {output_file}.{limited_note}")


def main(input_dir, category_structure_file, output_dir, max_sentences=None, limit_file=None):
    process_directory(input_dir, category_structure_file, output_dir, max_sentences, limit_file)
    print(f"Conversion complete. All processed files are in {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Converte i file BASE/*.json in formato conversazionale "
                     "gerarchico (stage2: classificazione su 3 livelli). "
                     "Output in formato JSONL (una conversazione per riga), "
                     "scritto in streaming per non saturare la RAM su file grandi.")
    parser.add_argument("input_dir", help="Cartella con i file BASE (es. base/it/BASE)")
    parser.add_argument("category_structure_file", help="Path a DynamicNER.json")
    parser.add_argument("output_dir", help="Cartella di output (es. base/it/SWIFT/classify)")
    parser.add_argument("--max-sentences", type=int, default=None,
                         help="Limita il numero di frasi processate (utile per "
                              "file enormi come test.json, dato che ogni entita' "
                              "genera 3 conversazioni). Se non specificato, "
                              "processa tutte le frasi.")
    parser.add_argument("--limit-file", default=None,
                         help="Se specificato insieme a --max-sentences, il "
                              "limite si applica SOLO a questo file (es. "
                              "--limit-file test.json), lasciando gli altri "
                              "file della cartella interi. Se omesso, il "
                              "limite si applica a tutti i file.")
    args = parser.parse_args()
    main(args.input_dir, args.category_structure_file, args.output_dir,
         args.max_sentences, args.limit_file)