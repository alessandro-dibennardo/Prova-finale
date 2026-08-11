import json
import random
import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple
from metrics import evaluate_metrics, read_data_file, write_json_file
from pathlib import Path

def get_category_frequencies(data: List[Dict]) -> Dict[str, int]:
    frequencies = {}
    for item in data:
        answer = item['conversations'][1]['value']
        frequencies[answer] = frequencies.get(answer, 0) + 1
    return frequencies

def calculate_health_score(metrics: Dict[str, float]) -> float:
    return (
        0.4 * metrics['cohesion_score'] +
        0.3 * metrics['normalized_entropy'] +
        0.2 * (1 - metrics['gini_coefficient']) +
        0.1 * (2.5 - min(metrics['coefficient_of_variation'], 2.5)) / 2.5
    )

def adjust_special_strategy(metrics: Dict[str, float],
                          frequencies: Dict[str, int]) -> Tuple[List[float], float, bool]:
    health_score = calculate_health_score(metrics)

    if health_score < 0.4:
        process_prob = 0.2
        removal_probs = [0.1, 0.2, 0.2, 0.5]
        aggressive = False
    else:
        process_prob = 0.3
        removal_probs = [0.2, 0.3, 0.3, 0.2]
        aggressive = True

    if metrics['cohesion_score'] < 0.03:
        process_prob *= 0.8
    if metrics['normalized_entropy'] < 0.75:
        aggressive = False

    return removal_probs, process_prob, aggressive

def extract_options(user_message: str) -> List[str]:
    options_start = user_message.rfind(':') + 1
    return [opt.strip() for opt in user_message[options_start:].strip('?').split(',')]

def rebuild_user_message(original_message: str, new_options: List[str]) -> str:
    options_start = original_message.rfind(':') + 1
    return original_message[:options_start] + ' ' + ', '.join(new_options) + '?'

def remove_duplicate_special_options(options: List[str]) -> List[str]:
    seen_special = False
    unique_options = []
    for opt in options:
        if opt.startswith(('miscellaneous', 'other')):
            if not seen_special:
                unique_options.append(opt)
                seen_special = True
        else:
            unique_options.append(opt)
    return unique_options

def select_special_answer(options: List[str], frequencies: Dict[str, int],
                         aggressive: bool) -> str:
    special_options = [opt for opt in options
                      if opt.startswith(('miscellaneous', 'other'))]

    if not special_options:
        return 'unknown'

    if aggressive:
        return min(special_options, key=lambda x: frequencies.get(x, 0))
    else:
        return random.choice(special_options)

def process_conversation(conversation: List[Dict], removal_probs: List[float],
                        frequencies: Dict[str, int], aggressive: bool) -> None:
    user_message = conversation[0]['value']
    options = extract_options(user_message)

    options = remove_duplicate_special_options(options)
    if len(options) < 3:
        return

    new_answer = select_special_answer(options, frequencies, aggressive)
    if new_answer == 'unknown' and 'unknown' not in options:
        options.append('unknown')

    max_removable = len(options) - 3
    num_to_remove = min(
        random.choices(range(4), weights=removal_probs, k=1)[0],
        max_removable
    )

    options_to_remove = random.sample(
        [opt for opt in options if opt != new_answer],
        num_to_remove
    )
    kept_options = [opt for opt in options if opt not in options_to_remove]

    if new_answer not in kept_options:
        kept_options.append(new_answer)

    random.shuffle(kept_options)
    conversation[0]['value'] = rebuild_user_message(user_message, kept_options)
    conversation[1]['value'] = new_answer

def process_json(input_file_path: str, output_file_path: str,
                hierarchy_file_path: str, output_as_jsonl: bool = False) -> None:
    try:
        data = read_data_file(input_file_path)

        metrics = evaluate_metrics(input_file_path, hierarchy_file_path)
        frequencies = get_category_frequencies(data)

        removal_probs, process_prob, aggressive = adjust_special_strategy(
            metrics, frequencies
        )

        for item in data:
            if random.random() < process_prob:
                try:
                    process_conversation(
                        item['conversations'],
                        removal_probs,
                        frequencies,
                        aggressive
                    )
                except Exception as e:
                    print(f"Error processing conversation: {str(e)}")
                    continue

        write_json_file(output_file_path, data, as_jsonl=output_as_jsonl)

        print(f"Initial metrics: {metrics}")
        print(f"Health score: {calculate_health_score(metrics):.4f}")
        print(f"Removal probabilities: {removal_probs}")
        print(f"Process probability: {process_prob}")
        print(f"Aggressive mode: {aggressive}")

    except Exception as e:
        print(f"Error processing file: {str(e)}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 4: gestione opzioni speciali miscellaneous/other (dynamic4)")
    parser.add_argument("--lang", default="it")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--base-dir", default="DynamicNER")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--hierarchy", default=None)
    parser.add_argument("--output-jsonl", action="store_true",
                         help="Scrive il file finale in formato JSONL invece di "
                              "JSON array (utile per split grandi come train, "
                              "coerente con stage2_trans.py)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    base_dir = Path(args.base_dir)

    input_file_path = args.input or str(
        base_dir / 'dynamic' / 'classify' / args.lang / f'{args.split}3.json')
    # Output finale: dynamic/classify/<lang>/<split>.json (o .jsonl),
    # coerente con la struttura reale mostrata nello screenshot
    # (dynamic/classify/<lang>/{dev,test,train}.json).
    ext = 'jsonl' if args.output_jsonl else 'json'
    output_file_path = args.output or str(
        base_dir / 'dynamic' / 'classify' / args.lang / f'{args.split}.{ext}')
    hierarchy_file_path = args.hierarchy or str(base_dir / 'DynamicNER.json')

    process_json(input_file_path, output_file_path, hierarchy_file_path,
                 output_as_jsonl=args.output_jsonl)