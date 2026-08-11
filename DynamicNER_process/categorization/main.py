import argparse
import logging
from pathlib import Path
from metrics import evaluate_metrics, resolve_input_path
from dynamic1 import main as dynamic1_main
from dynamic2 import process_json as dynamic2_process
from dynamic3 import process_json as dynamic3_process
from dynamic4 import process_json as dynamic4_process

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_directories(base_dir: Path, lang: str):
    """Assicura che le cartelle necessarie esistano.
    Corretto da 'balanced/de/SWIFT/classify' (originale) a 'base/<lang>/SWIFT/classify'
    (struttura reale usata nel resto del progetto per l'italiano)."""
    (base_dir / 'dynamic' / 'classify' / lang).mkdir(parents=True, exist_ok=True)
    (base_dir / 'base' / lang / 'SWIFT' / 'classify').mkdir(parents=True, exist_ok=True)


def calculate_final_metrics(json_path: str, hierarchy_path: str):
    try:
        metrics = evaluate_metrics(json_path, hierarchy_path)
        logger.info("Final Metrics:")
        logger.info("-" * 50)
        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")


def run_pipeline(lang: str, split: str, base_dir: Path, output_jsonl: bool,
                  synonyms_path: str = None, hierarchy_path: str = None):
    try:
        ensure_directories(base_dir, lang)

        # Input: base/<lang>/SWIFT/classify/<split>.json OPPURE .jsonl
        # (nella struttura reale del progetto alcuni split sono in .jsonl,
        # es. generati da stage2_trans.py: risolviamo automaticamente
        # quale dei due esiste, invece di assumere un'estensione fissa)
        initial_json_base = base_dir / 'base' / lang / 'SWIFT' / 'classify' / split
        initial_json = str(resolve_input_path(initial_json_base))

        # Stage intermedi: dynamic/classify/<lang>/<split>{1,2,3}.json
        # (corretto da 'dynamic/<lang>' del main.py originale, per allinearsi
        # alla struttura reale dynamic/classify/<lang>/ vista nello screenshot)
        stage_dir = base_dir / 'dynamic' / 'classify' / lang
        dev1_json = str(stage_dir / f'{split}1.json')
        dev2_json = str(stage_dir / f'{split}2.json')
        dev3_json = str(stage_dir / f'{split}3.json')

        ext = 'jsonl' if output_jsonl else 'json'
        final_json = str(stage_dir / f'{split}.{ext}')

        hierarchy_json = hierarchy_path or str(base_dir / 'DynamicNER.json')
        dynamic_txt = synonyms_path or str(base_dir / 'dynamic.txt')

        if not Path(dynamic_txt).exists():
            logger.warning(
                f"File sinonimi non trovato: {dynamic_txt}. Le categorie restano "
                f"in inglese anche per il corpus italiano (vedi build_base_it.py), "
                f"quindi puoi riusare un dynamic.txt già esistente da un'altra "
                f"lingua (es. de/ja) rinominandolo/copiandolo qui, invece di "
                f"crearne uno da zero.")

        logger.info(f"Running Stage 1: Merge Processing (lang={lang}, split={split})")
        dynamic1_main(initial_json, dev1_json, hierarchy_json)

        logger.info("Running Stage 2: Synonym Replacement")
        dynamic2_process(dev1_json, dev2_json, dynamic_txt, hierarchy_json)

        logger.info("Running Stage 3: Option Reduction")
        dynamic3_process(dev2_json, dev3_json, hierarchy_json)

        logger.info("Running Stage 4: Special Processing")
        dynamic4_process(dev3_json, final_json, hierarchy_json, output_as_jsonl=output_jsonl)

        calculate_final_metrics(final_json, hierarchy_json)

        logger.info("Pipeline completed successfully!")
        logger.info(f"Output finale: {final_json}")

    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        logger.error("Error details:", exc_info=True)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestratore della pipeline dynamic1-4 per una lingua/split")
    parser.add_argument("--lang", default="it",
                         help="Codice lingua (default: it, era hardcoded 'ja' nell'originale)")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"],
                         help="Split da processare (default: train)")
    parser.add_argument("--base-dir", default=None,
                         help="Cartella radice DynamicNER (default: cartella padre "
                              "di questo script, come nell'originale)")
    parser.add_argument("--synonyms", default=None,
                         help="Override del file dynamic.txt")
    parser.add_argument("--hierarchy", default=None,
                         help="Override esplicito di DynamicNER.json. Usa questo se "
                              "DynamicNER.json NON si trova nella stessa cartella "
                              "radice di base/ e dynamic/ (es. sta in "
                              "DynamicNER_process/ mentre base/ sta in DynamicNER/).")
    parser.add_argument("--output-jsonl", action="store_true",
                         help="Scrive l'output finale in JSONL invece di JSON array "
                              "(consigliato per split grandi come train, ~85k righe)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        # Comportamento originale: cartella padre della cartella di questo script
        current_dir = Path(__file__).parent.absolute()
        base_dir = current_dir.parent

    run_pipeline(
        lang=args.lang,
        split=args.split,
        base_dir=base_dir,
        output_jsonl=args.output_jsonl,
        synonyms_path=args.synonyms,
        hierarchy_path=args.hierarchy,
    )