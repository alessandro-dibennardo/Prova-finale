import json
import re

import fitz  # PyMuPDF


def extract_and_clean_pdf(pdf_path, start_page=6):
    """Estrae il testo dal PDF, pulendolo dalle imperfezioni tipografiche."""
    doc = fitz.open(pdf_path)
    raw_blocks = []
    
    # Iniziamo dalla pagina 7 (indice 6), saltando il sommario iniziale
    for page_num in range(start_page, min(start_page + 10, doc.page_count)):
        page = doc.load_page(page_num)
        # Estraiamo i blocchi di testo mantenendo l'ordine di lettura
        blocks = page.get_text("blocks")
        
        for b in blocks:
            text = b[4].strip()
            # Ignoriamo le intestazioni di pagina, i numeri e i blocchi minuscoli
            if len(text) > 40 and "GAZZETTA UFFICIALE DELLA REPUBBLICA ITALIANA" not in text:
                raw_blocks.append(text)
                
    return raw_blocks

def text_to_sentences(blocks):
    """Pulisce la formattazione a colonne e divide in frasi di senso compiuto."""
    clean_sentences = []
    
    for text in blocks:
        # 1. Ricongiunge le parole spezzate per andare a capo (es. "amministra-\nzione")
        text = re.sub(r'-\n\s*', '', text)
        
        # 2. Sostituisce i ritorni a capo rimanenti con uno spazio
        text = re.sub(r'\n', ' ', text)
        
        # 3. Rimuove eventuali spazi multipli
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 4. Divide in frasi cercando il punto fermo seguito da spazio e lettera maiuscola
        sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
        
        for s in sentences:
            if len(s) > 40:  # Evita frasi monche o titoli brevi
                clean_sentences.append(s)
                
    return clean_sentences

def write_swift_jsonl(sentences, output_path):
    """Salva le frasi nel formato JSONL atteso da ms-swift per l'inferenza."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in sentences:
            # Formato standard richiesto per l'inferenza: query e un response fittizio vuoto
            record = {"query": s, "response": ""}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_pdf = "20260724_056.pdf"
    output_jsonl = "test_gazzetta_infer.jsonl"
    
    print(f"Leggendo il PDF {input_pdf}...")
    blocks = extract_and_clean_pdf(input_pdf, start_page=6) # Pagina 7 del PDF
    
    print("Pulizia del testo e segmentazione in frasi...")
    sentences = text_to_sentences(blocks)
    
    print(f"Trovate {len(sentences)} frasi valide. Salvataggio in corso...")
    write_swift_jsonl(sentences, output_jsonl)
    
    print(f"Finito! Il file {output_jsonl} è pronto per essere dato in pasto all'Extractor.")