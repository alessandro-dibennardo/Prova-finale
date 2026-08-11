import json
import math
import re
from collections import Counter, defaultdict
from itertools import accumulate
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# ---------------------------------------------------------------------------
# I/O condiviso: legge sia JSON-array (formato originale del progetto) sia
# JSONL (una riga = un oggetto JSON), senza bisogno di conversioni preliminari.
# Il formato viene rilevato guardando il primo carattere non-whitespace del
# file: '[' -> JSON array, altrimenti -> JSONL (un oggetto per riga).
# ---------------------------------------------------------------------------

def read_data_file(file_path: Union[str, Path]) -> List[Dict]:
    """Legge un file .json (array) o .jsonl (un oggetto per riga) e restituisce
    sempre una lista di dict, indipendentemente dal formato su disco."""
    file_path = Path(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = ''
        pos = f.tell()
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break
        f.seek(pos)

        if first_char == '[':
            return json.load(f)

        # JSONL: una riga = un oggetto JSON. Righe vuote vengono saltate.
        records = []
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] {file_path}:{line_no} riga JSONL non valida, "
                      f"saltata ({e})")
        return records


def write_json_file(file_path: Union[str, Path], data: List[Dict],
                     as_jsonl: bool = False) -> None:
    """Scrive la lista di dict. Se as_jsonl=True scrive in formato JSONL
    (consigliato per file grandi, coerente con stage2_trans.py); altrimenti
    scrive un JSON array (comportamento originale degli script dynamic*.py)."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if as_jsonl:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Alias per compatibilita' con il codice esistente in dynamic1-4.py, che
# importava read_json_file/write_json_file con questi nomi esatti.
def read_json_file(file_path: Union[str, Path]) -> List[Dict]:
    return read_data_file(file_path)


def resolve_input_path(path_without_ext: Union[str, Path]) -> Path:
    """
    Dato un path SENZA estensione (es. '.../train'), trova quale dei due
    file esiste realmente su disco: '.../train.json' o '.../train.jsonl'.
    """
    base = Path(path_without_ext)
    candidates = [base.with_suffix('.jsonl'), base.with_suffix('.json')]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ', '.join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Nessun file trovato per '{base}'. Provati: {tried}")


# ---------------------------------------------------------------------------
# Estrazione dell'entita' dal prompt di classificazione.
# Il prompt (generato da stage2_trans.py) ha sempre la forma:
#   The ##ENTITA'## in the sentence: "...##ENTITA'##..." belongs to which
#   entity in the list: opt1, opt2, ...?
# La prima occorrenza di ##...## e' sempre l'entita' isolata, quindi basta
# prendere il primo match.
# ---------------------------------------------------------------------------

_ENTITY_PATTERN = re.compile(r"##(.*?)##")


def extract_entity_from_query(query: str) -> Optional[str]:
    match = _ENTITY_PATTERN.search(query)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Embedding BERT per il calcolo della coesione, fedele alla formula del paper
# DynamicNER (Appendice B, Eq. 3-4): la coesione e' la similarita' coseno
# MEDIA tra gli embedding BERT di tutte le entita' della stessa categoria.
# Il paper usa "BERT-base"; qui usiamo bert-base-multilingual-cased per
# poter processare correttamente l'italiano (e restare confrontabili con
# le altre lingue del progetto), con mean-pooling sull'ultimo hidden state
# come rappresentazione della frase/entita' (approccio standard in assenza
# di un modello sentence-transformers dedicato).
# ---------------------------------------------------------------------------

class BertEntityEmbedder:
    _instances: Dict[Tuple[str, str], "BertEntityEmbedder"] = {}  # cache per (model_name, device)

    def __init__(self, model_name: str = "bert-base-multilingual-cased",
                 device: str = "cpu", batch_size: int = 32):
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "Il calcolo fedele della cohesion (Appendice B del paper) "
                "richiede 'transformers' e 'torch'. Installa con:\n"
                "    pip install transformers torch --break-system-packages\n"
                "In alternativa, usa MetricsEvaluator(..., cohesion_mode='proxy') "
                "per il vecchio proxy basato su gerarchia (piu' veloce ma NON "
                "confrontabile con i valori del paper).")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def get_shared(cls, model_name: str = "bert-base-multilingual-cased",
                    device: str = "cpu", **kwargs) -> "BertEntityEmbedder":
        """Riusa la stessa istanza per una data coppia (model_name, device),
        cosi' modelli diversi richiesti nella stessa sessione Python (es.
        multilingue vs italiano) non si sovrascrivono a vicenda in cache."""
        key = (model_name, device)
        if key not in cls._instances:
            cls._instances[key] = cls(model_name=model_name, device=device, **kwargs)
        return cls._instances[key]

    def embed(self, texts: List[str]) -> "np.ndarray":
        if np is None:
            raise ImportError("numpy e' richiesto per il calcolo degli embedding.")

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                      max_length=64, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
            # Mean pooling sui token non-padding (attention_mask), pratica
            # standard per ottenere un embedding di frase/entita' da BERT
            # senza una testa di pooling dedicata.
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts
            all_embeddings.append(mean_pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def _cosine_similarity_matrix_mean(vectors: "np.ndarray") -> float:
    """Calcola la similarita' coseno media su tutte le coppie (i<j) di un
    insieme di vettori, esattamente come la Cohesion di Eq. 3 del paper:
        Cohesion = 1/(n(n-1)) * sum_{i} sum_{j != i} cos(v_i, v_j)
    Nota: la formula del paper somma su TUTTE le coppie ordinate i != j
    (non solo i<j) ma cos(vi,vj)=cos(vj,vi), quindi e' equivalente a usare
    coppie non ordinate i<j con normalizzazione 2/(n(n-1)) -> risultato
    identico a fare la media su tutte le coppie i<j.
    """
    n = vectors.shape[0]
    if n < 2:
        return None  # coesione non definita per categorie con <2 entita'

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normalized = vectors / norms
    sim_matrix = normalized @ normalized.T  # cos(vi, vj) per ogni coppia

    # Somma dei soli elementi sopra la diagonale (coppie i<j), poi media
    # su n(n-1)/2 coppie non ordinate (equivalente a n(n-1) coppie ordinate).
    iu = np.triu_indices(n, k=1)
    pair_sims = sim_matrix[iu]
    return float(np.mean(pair_sims))


class MetricsEvaluator:
    def __init__(self, hierarchy_path: str, cohesion_mode: str = "proxy",
                 bert_model_name: str = "bert-base-multilingual-cased",
                 device: str = "cpu"):
        """
        Args:
            hierarchy_path: percorso al file di gerarchia (DynamicNER.json)
            cohesion_mode: 'proxy' (DEFAULT — verificato sul codice ufficiale
                rilasciato dagli autori del paper: co-occorrenza gerarchica,
                stesso ordine di grandezza di Figura 6) oppure 'bert'
                (similarita' coseno tra embedding BERT delle entita';
                inizialmente implementato seguendo la descrizione testuale
                dell'Appendice B, ma i valori assoluti NON sono confrontabili
                con Figura 6 — verificato con due modelli diversi,
                multilingue e italiano, entrambi fuori scala di un ordine
                di grandezza. Mantenuto solo come riferimento/nota
                metodologica, non usare per il confronto con Figura 6).
            bert_model_name: modello HuggingFace usato per gli embedding
                quando cohesion_mode='bert'.
            device: 'cpu' o 'cuda' per il calcolo degli embedding (rilevante
                solo se cohesion_mode='bert').
        """
        self.hierarchy = self._load_hierarchy(hierarchy_path)
        self.cohesion_mode = cohesion_mode
        self._category_to_branch = self._build_branch_index()
        self._embedder = None
        if cohesion_mode == "bert":
            self._embedder = BertEntityEmbedder.get_shared(
                model_name=bert_model_name, device=device)
        elif cohesion_mode != "proxy":
            raise ValueError("cohesion_mode deve essere 'bert' o 'proxy'")

    def _load_hierarchy(self, file_path: str) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_branch_index(self) -> Dict[str, str]:
        """Usato solo se cohesion_mode='proxy'. Assegna a ogni categoria di
        2°/3° livello l'id del suo genitore immediato."""
        mapping: Dict[str, str] = {}
        first_levels = [c.strip() for c in self.hierarchy['first-level'].split(',')]

        for first_level in first_levels:
            second_levels_str = self.hierarchy['second-level'].get(first_level, '')
            second_levels = [c.strip() for c in second_levels_str.split(',') if c.strip()]
            for second_level in second_levels:
                mapping[second_level] = f"L1::{first_level}"
                third_levels_str = self.hierarchy['third-level'].get(second_level, '')
                third_levels = [c.strip() for c in third_levels_str.split(',') if c.strip()]
                for third_level in third_levels:
                    mapping[third_level] = f"L2::{second_level}"

        return mapping

    def _extract_categories(self, data: List[Dict]) -> List[str]:
        categories = []
        for item in data:
            try:
                answer = item['conversations'][1]['value']
                categories.append(answer)
            except (KeyError, IndexError):
                continue
        return categories

    def _extract_category_entity_pairs(self, data: List[Dict]) -> List[Tuple[str, str]]:
        """Estrae coppie (categoria, testo_entita') da ogni conversazione,
        necessarie per il calcolo della cohesion fedele al paper (che opera
        sulle ENTITA', non sulle sole etichette di categoria)."""
        pairs = []
        for item in data:
            try:
                query = item['conversations'][0]['value']
                answer = item['conversations'][1]['value']
            except (KeyError, IndexError):
                continue
            entity = extract_entity_from_query(query)
            if entity:
                pairs.append((answer, entity))
        return pairs

    def calculate_cohesion_score_bert(self, category_entity_pairs: List[Tuple[str, str]]) -> float:
        """
        Cohesion fedele al paper (Appendice B, Eq. 3-4): per ogni categoria,
        calcola la similarita' coseno media tra gli embedding BERT di tutte
        le entita' etichettate con quella categoria; il valore finale e' la
        media (non pesata) delle cohesion di categoria, ignorando le
        categorie con meno di 2 istanze (per cui la coesione non e' definita).

        ATTENZIONE PERFORMANCE: richiede di calcolare un embedding BERT per
        ciascuna entita' distinta nel dataset. Su file con decine di migliaia
        di righe, valuta l'uso di --device cuda o di un campione rappresentativo
        piuttosto che l'intero file, specialmente in fase di sviluppo/debug.
        """
        if not category_entity_pairs:
            return 0.0

        entities_by_category: Dict[str, List[str]] = defaultdict(list)
        for category, entity in category_entity_pairs:
            entities_by_category[category].append(entity)

        all_entities = [e for _, e in category_entity_pairs]
        all_embeddings = self._embedder.embed(all_entities)

        # Rimappa ogni embedding alla propria categoria, nell'ordine originale
        idx = 0
        embeddings_by_category: Dict[str, List["np.ndarray"]] = defaultdict(list)
        for category, entity in category_entity_pairs:
            embeddings_by_category[category].append(all_embeddings[idx])
            idx += 1

        category_cohesions = []
        for category, emb_list in embeddings_by_category.items():
            if len(emb_list) < 2:
                continue  # coesione non definita con una sola entita'
            vectors = np.stack(emb_list, axis=0)
            cohesion = _cosine_similarity_matrix_mean(vectors)
            if cohesion is not None:
                category_cohesions.append(cohesion)

        if not category_cohesions:
            return 0.0
        return float(np.mean(category_cohesions))

    def calculate_cohesion_score_proxy(self, categories: List[str]) -> float:
        """Vecchio proxy O(n) basato su co-occorrenza gerarchica (NON e' la
        metrica del paper: mantenuto solo per retrocompatibilita' e per un
        controllo di sanita' rapido senza dover caricare BERT)."""
        if not categories:
            return 0.0

        total_n = len(categories)
        total_pairs = total_n * (total_n - 1) // 2
        if total_pairs == 0:
            return 0.0

        branch_counts: Counter = Counter()
        for cat in categories:
            branch_id = self._category_to_branch.get(cat)
            if branch_id is not None:
                branch_counts[branch_id] += 1

        cohesive_pairs = sum(
            count * (count - 1) // 2 for count in branch_counts.values()
        )

        return cohesive_pairs / total_pairs

    def calculate_normalized_entropy(self, categories: List[str]) -> float:
        if not categories:
            return 0.0

        counter = Counter(categories)
        total = len(categories)

        entropy = 0
        for count in counter.values():
            p = count / total
            entropy -= p * math.log2(p)

        max_entropy = math.log2(len(counter)) if len(counter) > 1 else 0.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def calculate_gini_coefficient(self, categories: List[str]) -> float:
        if not categories:
            return 0.0

        counter = Counter(categories)
        values = sorted(counter.values())
        total = sum(values)

        if total == 0:
            return 0.0

        n = len(values)
        if n == 0:
            return 0.0

        if np is not None:
            cumsum = np.cumsum(values)
            total_cumsum = float(np.sum(cumsum))
        else:
            cumsum_list = list(accumulate(values))
            total_cumsum = float(sum(cumsum_list))

        return (n + 1 - 2 * total_cumsum / total) / n

    def calculate_coefficient_of_variation(self, categories: List[str]) -> float:
        if not categories:
            return 0.0

        counter = Counter(categories)
        values = list(counter.values())

        if np is not None:
            mean = float(np.mean(values))
            std = float(np.std(values))
        else:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance)

        return std / mean if mean > 0 else 0.0

    def evaluate_file(self, file_path: str) -> Dict[str, float]:
        try:
            data = read_data_file(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return {}

        categories = self._extract_categories(data)

        if self.cohesion_mode == "bert":
            pairs = self._extract_category_entity_pairs(data)
            cohesion = self.calculate_cohesion_score_bert(pairs)
        else:
            cohesion = self.calculate_cohesion_score_proxy(categories)

        metrics = {
            'cohesion_score': cohesion,
            'normalized_entropy': self.calculate_normalized_entropy(categories),
            'gini_coefficient': self.calculate_gini_coefficient(categories),
            'coefficient_of_variation': self.calculate_coefficient_of_variation(categories)
        }

        return metrics


def evaluate_metrics(json_path: str, hierarchy_path: str,
                      cohesion_mode: str = "proxy", device: str = "cpu",
                      bert_model_name: str = "bert-base-multilingual-cased") -> Dict[str, float]:
    """Funzione di convenienza per valutare un singolo file."""
    evaluator = MetricsEvaluator(hierarchy_path, cohesion_mode=cohesion_mode,
                                  device=device, bert_model_name=bert_model_name)
    return evaluator.evaluate_file(json_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Valuta le metriche di qualita' di un file (fedeli al "
                    "paper DynamicNER, Appendice B)")
    parser.add_argument("json_path", nargs="?", default="dynamic/classify/it/train.json")
    parser.add_argument("--hierarchy", default="DynamicNER.json")
    parser.add_argument("--cohesion-mode", choices=["bert", "proxy"], default="proxy",
                         help="'proxy' (DEFAULT) = co-occorrenza gerarchica, verificato "
                              "sul codice ufficiale del paper: comparabile con Figura 6. "
                              "'bert' = similarita' embedding BERT (interpretazione del "
                              "testo dell'Appendice B, MA NON comparabile in valore "
                              "assoluto con Figura 6, verificato con 2 modelli diversi "
                              "— usare solo come nota metodologica separata).")
    parser.add_argument("--bert-model", default="bert-base-multilingual-cased",
                         help="Modello HuggingFace per gli embedding (usato solo se "
                              "--cohesion-mode=bert). Es. 'dbmdz/bert-base-italian-cased' "
                              "per un modello monolingue italiano, piu' coerente con "
                              "l'uso di modelli monolingue del paper originale "
                              "(bert-base-cased per l'inglese, bert-base-chinese per il cinese).")
    parser.add_argument("--device", default="cpu", help="cpu o cuda, per il calcolo degli embedding")
    args = parser.parse_args()

    metrics = evaluate_metrics(args.json_path, args.hierarchy,
                                cohesion_mode=args.cohesion_mode, device=args.device,
                                bert_model_name=args.bert_model)
    print("Evaluation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")