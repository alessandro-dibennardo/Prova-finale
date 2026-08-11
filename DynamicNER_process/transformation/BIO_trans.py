import json
import re
import os
from pathlib import Path

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def tokenize_with_punctuation(text):
    return re.findall(r'\w+|[^\w\s]', text)

def format_category(category):
    return category.replace(' ', '_')

def convert_to_bio(json_data):
    bio_data = []
    
    for sentence_id, sentence_info in json_data.items():
        sentence = sentence_info['sentence']
        entities = sentence_info['entity']
        categories = sentence_info['category']
        
        tokens = tokenize_with_punctuation(sentence)
        bio_tokens = ['O'] * len(tokens)
        
        for entity, category in zip(entities, categories):
            formatted_category = format_category(category)
            entity_tokens = tokenize_with_punctuation(entity)
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if tokens[i:i+len(entity_tokens)] == entity_tokens:
                    bio_tokens[i] = f'B-{formatted_category}'
                    for j in range(1, len(entity_tokens)):
                        bio_tokens[i+j] = f'I-{formatted_category}'
                    break
        
        bio_data.extend([(token, bio) for token, bio in zip(tokens, bio_tokens)])
        bio_data.append(('', ''))  # Empty line to separate sentences
    
    return bio_data

def write_bio_file(bio_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for token, bio in bio_data:
            if token == '' and bio == '':
                file.write('\n')
            else:
                file.write(f'{token} {bio}\n')

def process_file(input_file, output_file):
    json_data = load_json_file(input_file)
    bio_data = convert_to_bio(json_data)
    write_bio_file(bio_data, output_file)
    print(f"Conversion complete. BIO format data written to {output_file}")

def process_language(language_dir: Path):
    input_directory = language_dir / "BASE"
    if not input_directory.exists():
        print(f"Skipping {language_dir.name}: BASE directory not found.")
        return

    output_directory = language_dir / "BIO"
    output_directory.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            input_file = input_directory / filename
            output_file = output_directory / f"{Path(filename).stem}.txt"
            process_file(input_file, output_file)

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Converte base/<lang>/BASE/*.json in formato BIO dentro base/<lang>/BIO/.")
    parser.add_argument("--lang", default=None,
                         help="Se specificato, processa solo questa lingua "
                              "(es. --lang it) invece di tutte le sottocartelle "
                              "di base/.")
    args = parser.parse_args()

    # Struttura: .../Prova finale/DynamicNER/DynamicNER_process/transformation/BIO_trans.py
    #            .../Prova finale/DynamicNER/DynamicNER/base/<lang>/BASE
    # DynamicNER_process e DynamicNER (quella con base/) sono cartelle sorelle,
    # entrambe dentro "Prova finale/DynamicNER/".
    repo_root = Path(__file__).resolve().parent.parent.parent
    base_dir = repo_root / "DynamicNER" / "base"

    if not base_dir.exists():
        print(f"Base directory not found at {base_dir}")
        return

    if args.lang:
        language_dir = base_dir / args.lang
        if not language_dir.is_dir():
            print(f"Lingua '{args.lang}' non trovata in {base_dir}")
            return
        print(f"Generating BIO format for language: {language_dir.name}")
        process_language(language_dir)
        return

    for language_dir in sorted(base_dir.iterdir()):
        if language_dir.is_dir():
            print(f"Generating BIO format for language: {language_dir.name}")
            process_language(language_dir)

if __name__ == "__main__":
    main()

# Comando per eseguire lo script solo per la lingua italiana:
# python BIO_trans.py --lang it 