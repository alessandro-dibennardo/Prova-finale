import json
import random
import argparse
import logging
from typing import Dict, List, Tuple
from metrics import evaluate_metrics, read_data_file, write_json_file
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_category_frequencies(data: List[Dict]) -> Dict[str, int]:
    frequencies = {}
    for item in data:
        answer = item['conversations'][1]['value']
        frequencies[answer] = frequencies.get(answer, 0) + 1
    return frequencies

def extract_options(user_message: str) -> List[str]:
    options_start = user_message.rfind(':') + 1
    return [opt.strip() for opt in user_message[options_start:].strip('?').split(',')]

def rebuild_user_message(original_message: str, new_options: List[str]) -> str:
    options_start = original_message.rfind(':') + 1
    return original_message[:options_start] + ' ' + ', '.join(new_options) + '?'

def process_conversation(conversation: List[Dict], frequencies: Dict[str, int]) -> None:
    try:
        user_message = conversation[0]['value']
        assistant_answer = conversation[1]['value']
        options = extract_options(user_message)

        if len(options) <= 3:
            return

        if assistant_answer not in options:
            options.append(assistant_answer)

        current_size = len(options)
        min_keep = 3
        if current_size <= min_keep:
            return

        num_to_keep = random.randint(min_keep, current_size)

        kept_options = [assistant_answer]
        other_options = [opt for opt in options if opt != assistant_answer]

        additional_keeps = num_to_keep - 1
        if additional_keeps > 0 and other_options:
            kept_options.extend(random.sample(other_options, min(additional_keeps, len(other_options))))

        random.shuffle(kept_options)

        conversation[0]['value'] = rebuild_user_message(user_message, kept_options)

    except Exception as e:
        logger.error(f"Error in process_conversation: {str(e)}")
        raise

def process_json(input_file_path: str, output_file_path: str, hierarchy_file_path: str) -> None:
    try:
        data = read_data_file(input_file_path)

        metrics = evaluate_metrics(input_file_path, hierarchy_file_path)
        frequencies = get_category_frequencies(data)

        base_prob = 0.3
        if metrics['normalized_entropy'] < 0.8:
            process_prob = base_prob * 0.8
        else:
            process_prob = base_prob

        for item in data:
            if random.random() < process_prob:
                try:
                    process_conversation(item['conversations'], frequencies)
                except Exception as e:
                    logger.error(f"Error processing conversation: {str(e)}")
                    continue

        write_json_file(output_file_path, data)

        logger.info(f"Initial metrics: {metrics}")
        logger.info(f"Process probability: {process_prob}")

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: riduzione/mescolamento opzioni (dynamic3)")
    parser.add_argument("--lang", default="it")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--base-dir", default="DynamicNER")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--hierarchy", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    base_dir = Path(args.base_dir)

    input_file_path = args.input or str(
        base_dir / 'dynamic' / 'classify' / args.lang / f'{args.split}2.json')
    output_file_path = args.output or str(
        base_dir / 'dynamic' / 'classify' / args.lang / f'{args.split}3.json')
    hierarchy_file_path = args.hierarchy or str(base_dir / 'DynamicNER.json')

    process_json(input_file_path, output_file_path, hierarchy_file_path)