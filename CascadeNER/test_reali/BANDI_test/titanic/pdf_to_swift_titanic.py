import json
import re

import fitz  # PyMuPDF


def extract_and_clean_pdf(pdf_path):
    """Estrae il testo da tutte le pagine del PDF."""
    doc = fitz.open(pdf_path)
    raw_blocks = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        
        for b in blocks:
            text = b[4].strip()
            # Filtra blocchi troppo corti (es. numeri di pagina isolati)
            if len(text) > 20:
                raw_blocks.append(text)
                
    return raw_blocks

def text_to_sentences(blocks):
    """Pulisce i ritorni a capo e segmenta il testo in frasi."""
    clean_sentences = []
    
    for text in blocks:
        # Ricongiunge le parole spezzate per andare a capo
        text = re.sub(r'-\n\s*', '', text)
        # Sostituisce i ritorni a capo rimanenti con uno spazio
        text = re.sub(r'\n', ' ', text)
        # Rimuove spazi multipli
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Divide in frasi cercando punteggiatura forte (. ! ?) seguita da maiuscola
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        for s in sentences:
            if len(s) > 20:
                clean_sentences.append(s)
                
    return clean_sentences

def write_swift_jsonl(sentences, output_path):
    """Salva le frasi con la chiave fittizia 'response' per ms-swift."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in sentences:
            record = {"query": s, "response": ""}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_pdf = "test_reali/titanic.pdf"  # Assicurati che il PDF si trovi qui
    output_jsonl = "test_reali/test_titanic_infer.jsonl"
    
    blocks = extract_and_clean_pdf(input_pdf)
    sentences = text_to_sentences(blocks)
    write_swift_jsonl(sentences, output_jsonl)
    print(f"Finito! Salvate {len(sentences)} frasi in {output_jsonl}.")