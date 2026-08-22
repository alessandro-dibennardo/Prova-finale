## Riconoscimenti e Architettura
Questo progetto adatta per l'italiano il framework originale **DynamicNER** e la pipeline **CascadeNER**. I crediti per l'architettura originaria vanno ai rispettivi autori ([DynamicNER: A Dynamic, Multilingual, and Fine-Grained Dataset for LLM-based Named Entity Recognition](https://aclanthology.org/2025.emnlp-main.835/) (Luo et al., EMNLP 2025)), mentre i dati di validazione e fine-tuning derivano dal dataset **MultiCoNER**. La pipeline, basata sul modello `Qwen2.5-1.5B-Instruct`, si divide in due fasi sequenziali:
*   **Stage 1 (Extractor):** Individua i confini delle entità testuali racchiudendole tra speciali delimitatori (`##entità##`).
*   **Stage 2 (Classifier):** Naviga un'ontologia gerarchica a 155 categorie per assegnare a ciascuna estrazione l'etichetta semantica corretta.

## Esperimenti: Zero-Shot e Fine-Tuning
Il progetto confronta due diversi approcci di addestramento e inferenza:
*   **Baseline Zero-Shot (Cross-Lingual):** Inizialmente, il modello è stato addestrato esclusivamente su dati in lingua inglese e testato in modalità zero-shot sul dataset italiano, dimostrando le capacità native di transfer learning cross-linguale dell'architettura.
*   **Fine-Tuning Nativo (Italiano):** Il modello è stato successivamente ottimizzato (LoRA) con `ms-swift` direttamente su dati italiani, portando a un drastico incremento delle prestazioni. I checkpoint ottimali individuati tramite Early Stopping sono lo Step 2200 per l'Extractor e lo Step 9100 per il Classifier.

## Stress Test su Dati Reali
Per valutare il fenomeno del *Domain Shift*, la cartella `test_reali/` include test su documenti PDF originali:
*   **Out-of-Domain (Gazzetta Ufficiale):** Applicato a sintassi burocratiche complesse, l'Extractor ha mostrato difficoltà (allucinazioni testuali), mentre il Classifier ha generato risultati `unknown` per concetti amministrativi estranei all'ontologia originaria.
*   **In-Domain (Recensione "Titanic"):** Su testi narrativo-giornalistici, il modello ha ripreso a estrarre in modo impeccabile luoghi, attori e opere, confermando l'efficacia del fine-tuning italiano su strutture sintattiche standard.

## Comandi di Inferenza e Riproducibilità
Dopo aver generato il formato JSONL dai PDF tramite gli script Python dedicati, eseguire i seguenti comandi dalla radice `CascadeNER`:

*   **Stage 1 (Extractor):**
    `swift infer --ckpt_dir "model/extractor_it/qwen2_5-1_5b-instruct/v0-20260807-115116/checkpoint-2200" --custom_val_dataset_path "test_reali/test_titanic_infer.jsonl" --result_dir "test_reali/titanic_stage1" --save_result true --max_new_tokens 256 --repetition_penalty 1.3 --do_sample false`

*   **Stage 2 (Classifier):**
    `python model/infer.py --responses_dir "test_reali/titanic_stage1" --classifier_model "model/classifier_it/qwen2_5-1_5b-instruct/v0-20260808-111412/checkpoint-9100" --category_file "../DynamicNER_process/DynamicNER.json" --output_file "test_reali/cascade_titanic_pred.json" --device cuda`