import json
import os

import matplotlib.pyplot as plt

# Parametri di configurazione
BASE_DIR = "test_reali/KIND_test"
NOME_TEST = "moro"  

# Percorsi
TEST_DIR = os.path.join(BASE_DIR, f"{NOME_TEST}_test")
GT_PATH = os.path.join(TEST_DIR, f"{NOME_TEST}_test_ground_truth.json")
PRED_PATH = os.path.join(TEST_DIR, f"cascade_{NOME_TEST}_pred.json")
OUTPUT_PLOT = os.path.join(TEST_DIR, f"grafico_confronto_{NOME_TEST}_migliorato.png")

def genera_grafico_confronto():
    try:
        with open(GT_PATH, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        with open(PRED_PATH, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    except FileNotFoundError as e:
        print(f"Errore: file non trovato. {e}")
        return
        
    y_gt = []
    y_pred = []
    
    num_frasi = min(len(ground_truth), len(predictions))
    x_labels = list(range(1, num_frasi + 1))
    
    totale_gt = 0
    totale_pred = 0
    
    # Allineamento e Conteggio
    for i in range(1, num_frasi + 1):
        chiave_gt = f"frase_{i}"
        chiave_pred = f"sentence{i}"
        
        num_entita_gt = len(ground_truth[chiave_gt].get("entita_reali", [])) if chiave_gt in ground_truth else 0
        num_entita_pred = len(predictions[chiave_pred].get("entity", [])) if chiave_pred in predictions else 0
            
        y_gt.append(num_entita_gt)
        y_pred.append(num_entita_pred)
        
        totale_gt += num_entita_gt
        totale_pred += num_entita_pred

    print(f"\n--- REPORT CONFRONTO ({NOME_TEST.upper()}) ---")
    print(f"Totale Entità Reali (Dataset KIND): {totale_gt}")
    print(f"Totale Entità Estratte (CascadeNER): {totale_pred}")
    print("-" * 30)
    
    if totale_gt == 0:
        print("\n[!] ATTENZIONE: Il totale del Ground Truth è 0. Controlla che il file JSON di KIND contenga le entità o che le chiavi si chiamino effettivamente 'frase_1', 'frase_2', ecc.")

    # Generazione grafico pulito
    plt.figure(figsize=(16, 6)) # Formato panorama più ampio
    
    # Usiamo solo linee (senza marker ingombranti) per un look più professionale
    plt.plot(x_labels, y_gt, label='Ground Truth (KIND)', color='#1f77b4', linewidth=1.8, alpha=0.9)
    plt.plot(x_labels, y_pred, label='Predizioni (CascadeNER)', color='#d62728', linewidth=1.5, alpha=0.7)
    
    plt.title(f'Confronto Entità Estratte per Frase - Dominio: {NOME_TEST.capitalize()}', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Indice della Frase', fontsize=12)
    plt.ylabel('Numero di Entità', fontsize=12)
    
    # Asse X Intelligente: stampa un'etichetta ogni N frasi
    step = max(1, num_frasi // 30) 
    plt.xticks(range(0, num_frasi + 1, step))
    
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"\nGrafico generato con successo e salvato in: {OUTPUT_PLOT}\n")

if __name__ == "__main__":
    genera_grafico_confronto()