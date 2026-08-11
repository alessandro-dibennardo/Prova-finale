import json
import random
import re
from pathlib import Path
import argparse

def load_hierarchy(path):
    """Mappa ogni foglia (terzo livello) al suo genitore (secondo livello)"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    leaf_to_parent = {}
    for parent, leaves in data.get("third-level", {}).items():
        if isinstance(leaves, str):
            leaves_list = [l.strip().lower() for l in leaves.split(',') if l.strip()]
        else:
            leaves_list = [l.strip().lower() for l in leaves]
        for leaf in leaves_list:
            leaf_to_parent[leaf] = parent.strip().lower()
    return leaf_to_parent

def load_synonyms(path):
    """Carica i sinonimi da dynamic.txt"""
    syn_map = {}
    if not Path(path).exists():
        return syn_map
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # Supporta vari delimitatori usati di solito per i sinonimi
            parts = re.split(r'[,=:>]+|\s*->\s*', line)
            if len(parts) >= 2:
                syn_map[parts[0].strip().lower()] = parts[1].strip().lower()
    return syn_map

def process_true_random(input_path, output_path, leaf_to_parent, syn_map):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip(): continue
            data = json.loads(line)
            
            # Estrazione flessibile (ms-swift o base)
            query, response = None, None
            if "query" in data:
                query, response = data["query"], data["response"]
            elif "messages" in data:
                for msg in data["messages"]:
                    if msg.get("role") == "user": query = msg.get("content")
                    if msg.get("role") == "assistant": response = msg.get("content")
            elif "conversations" in data:
                for msg in data["conversations"]:
                    if msg.get("from") == "user": query = msg.get("value")
                    if msg.get("from") == "assistant": response = msg.get("value")
                    
            if not query or not response:
                continue
                
            orig_response = response.strip().lower()
            new_response = orig_response
            
            # Scelta completamente casuale della strategia (25% ciascuna)
            strategy = random.choice([1, 2, 3, 4])
            
            if strategy == 1:
                # Metodo 1: Mix Granularity (Sostituisci con il genitore)
                new_response = leaf_to_parent.get(orig_response, orig_response)
            elif strategy == 2:
                # Metodo 2: Sinonimi
                new_response = syn_map.get(orig_response, orig_response)
            elif strategy == 3:
                # Metodo 3: Rimozione opzioni (non cambia il target, cambia solo il prompt)
                pass
            elif strategy == 4:
                # Metodo 4: Merge in miscellaneous
                new_response = random.choice(["miscellaneous", "other"])
            
            # Aggiorna la query con la nuova risposta corretta e (se strategia 3) riduce opzioni
            match = re.search(r'in the list:\s*(.*?)(?:\?|If none)', query)
            if match:
                opts_str = match.group(1)
                options = [o.strip() for o in opts_str.split(',')]
                
                new_options = []
                for o in options:
                    if o.lower() == orig_response:
                        new_options.append(new_response)
                    else:
                        new_options.append(o)
                        
                if strategy == 3 and len(new_options) > 3:
                    try:
                        new_options.remove(new_response)
                        num_keep = random.randint(2, len(new_options))
                        kept_negatives = random.sample(new_options, num_keep - 1)
                        new_options = [new_response] + kept_negatives
                        random.shuffle(new_options)
                    except ValueError:
                        pass
                
                new_opts_str = ", ".join(new_options)
                new_query = query.replace(opts_str, new_opts_str)
                
                # Riassegna i valori
                if "query" in data:
                    data["query"] = new_query
                    data["response"] = new_response
                elif "messages" in data:
                    for msg in data["messages"]:
                        if msg.get("role") == "user": msg["content"] = new_query
                        if msg.get("role") == "assistant": msg["content"] = new_response
                elif "conversations" in data:
                    for msg in data["conversations"]:
                        if msg.get("from") == "user": msg["value"] = new_query
                        if msg.get("from") == "assistant": msg["value"] = new_response

            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="../../DynamicNER/base/it/SWIFT/classify/")
    parser.add_argument("--out_dir", type=str, default="../../DynamicNER/random/classify/it/")
    parser.add_argument("--cat_file", type=str, default="../DynamicNER.json")
    parser.add_argument("--syn_file", type=str, default="dynamic.txt")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    leaf_to_parent = load_hierarchy(args.cat_file)
    syn_map = load_synonyms(args.syn_file)

    for split in ["train", "dev", "test"]:
        in_file = Path(args.base_dir) / f"{split}.jsonl"
        out_file = out_path / f"{split}.jsonl"
        if in_file.exists():
            process_true_random(in_file, out_file, leaf_to_parent, syn_map)
            print(f"Generato vero random per {split} in {out_file}")