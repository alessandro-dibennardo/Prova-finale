import glob
import json
import os

DATASET_DIR = "test_reali/KIND_test/dataset"
OUTPUT_DIR = "test_reali/KIND_test"

def process_kind_file(file_path):
    sentences = []
    ground_truth = {}
    
    current_tokens = []
    current_entities = []
    current_entity_words = []
    current_tag = "O"
    
    sentence_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    if current_tokens:
                        sentence_count += 1
                        if current_entity_words:
                            current_entities.append(" ".join(current_entity_words))
                            current_entity_words = []
                            current_tag = "O"
                            
                        testo_frase = " ".join(current_tokens)
                        sentences.append(testo_frase)
                        
                        ground_truth[f"frase_{sentence_count}"] = {
                            "testo": testo_frase,
                            "entita_reali": list(set(current_entities))
                        }
                        
                        current_tokens = []
                        current_entities = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        tag = parts[-1]
                        
                        current_tokens.append(token)
                        
                        if tag != "O":
                            if tag == current_tag:
                                # Stessa entità che continua (es. "Napoleone" -> "III")
                                current_entity_words.append(token)
                            else:
                                # Nuova entità trovata
                                if current_entity_words:
                                    current_entities.append(" ".join(current_entity_words))
                                current_entity_words = [token]
                                current_tag = tag
                        else:
                            # Parola normale (O)
                            if current_entity_words:
                                current_entities.append(" ".join(current_entity_words))
                                current_entity_words = []
                            current_tag = "O"
                                
    except FileNotFoundError:
        print(f"Errore: Il file {file_path} non è stato trovato.")
        return [], {}

    return sentences, ground_truth

if __name__ == "__main__":
    test_files = glob.glob(os.path.join(DATASET_DIR, "*_test.tsv"))
    print(f"Trovati {len(test_files)} file di test da processare.")
    
    for file_path in test_files:
        file_name = os.path.basename(file_path).split('.')[0]
        
        sentences, ground_truth = process_kind_file(file_path)
        
        if sentences:
            # Salvataggio nelle sottocartelle specifiche create da te
            target_dir = os.path.join(OUTPUT_DIR, f"{file_name}")
            os.makedirs(target_dir, exist_ok=True)
            
            gt_path = os.path.join(target_dir, f"{file_name}_ground_truth.json")
            infer_path = os.path.join(target_dir, f"{file_name}_infer.jsonl")
            
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(ground_truth, f, ensure_ascii=False, indent=4)
                
            with open(infer_path, "w", encoding="utf-8") as f:
                for s in sentences:
                    record = {"query": s, "response": ""}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    
            print(f"Elaborato {file_name}: rigenerato Ground Truth con {len(sentences)} frasi.")