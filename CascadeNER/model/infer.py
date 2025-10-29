"""Stage-2 categorisation script for CascadeNER.

The script aggregates entity extraction responses from Stage-1 (Swift inference),
merges overlapped mentions, and queries a classification model to obtain the
final label for each entity. All paths are configurable through CLI flags to
avoid hard-coded assumptions about the project layout.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path: Path, device: str):
    """Load a causal LM and its tokenizer for classification."""

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def extract_entities_with_positions(sentence: str, response: str) -> List[Dict[str, int]]:
    """Locate entities wrapped by ##...## in the LLM response."""

    entities: List[Dict[str, int]] = []
    for match in re.finditer(r"##(.*?)##", response):
        entity_text = match.group(1)
        start_idx = sentence.find(entity_text)
        if start_idx != -1:
            entities.append({
                "text": entity_text,
                "start": start_idx,
                "end": start_idx + len(entity_text),
            })
            continue

        # Fallback: search without anchors
        fallback = list(re.finditer(re.escape(entity_text), sentence))
        if fallback:
            first_match = fallback[0]
            entities.append({
                "text": entity_text,
                "start": first_match.start(),
                "end": first_match.end(),
            })
    return entities


def merge_entities(entity_lists: Sequence[Sequence[Dict[str, int]]]) -> List[Dict[str, int]]:
    """Merge entities across multiple responses, keeping the widest span."""

    merged: List[Dict[str, int]] = []
    for entities in entity_lists:
        for entity in entities:
            for existing in merged:
                overlaps = not (entity["end"] <= existing["start"] or entity["start"] >= existing["end"])
                if overlaps:
                    if (entity["end"] - entity["start"]) > (existing["end"] - existing["start"]):
                        existing.update(entity)
                    break
            else:
                merged.append(dict(entity))
    return merged


def parse_category_list(raw: Iterable[str] | str) -> List[str]:
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [str(item).strip() for item in raw]


class CategoryStructure:
    def __init__(self, data: Dict):
        self.first_options = parse_category_list(data.get("first-level", []))
        self.first_lookup = {item.lower(): item for item in self.first_options}

        self.second_options = {
            key.strip().lower(): parse_category_list(value)
            for key, value in (data.get("second-level") or {}).items()
        }
        self.second_lookup = {
            key: {option.lower(): option for option in options}
            for key, options in self.second_options.items()
        }

        self.third_options = {
            key.strip().lower(): parse_category_list(value)
            for key, value in (data.get("third-level") or {}).items()
        }
        self.third_lookup = {
            key: {option.lower(): option for option in options}
            for key, options in self.third_options.items()
        }


def load_category_structure(category_file: Path) -> CategoryStructure:
    with category_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return CategoryStructure(data)


def build_question(entity: str, sentence: str, level_name: str, options: Sequence[str]) -> str:
    options_str = ", ".join(options)
    return (
        f'The ##{entity}## in the sentence: "{sentence}" '
        f'belongs to which entity in the {level_name} list: {options_str}?'
    )


def generate_response(
    model,
    tokenizer,
    query: str,
    device: str,
    max_new_tokens: int,
    temperature: Optional[float],
) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    if device != "cpu":
        model_inputs = model_inputs.to(device)

    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if temperature is not None:
        gen_kwargs.update({"do_sample": True, "temperature": temperature})

    with torch.inference_mode():
        generated = model.generate(**model_inputs, **gen_kwargs)

    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def categorize_entities(
    entities: Sequence[str],
    sentence: str,
    structure: CategoryStructure,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    temperature: Optional[float],
) -> List[str]:
    if not entities:
        return []

    first_predictions: List[str] = []
    for entity in entities:
        query = build_question(entity, sentence, "first", structure.first_options)
        response = generate_response(model, tokenizer, query, device, max_new_tokens, temperature).lower()
        canonical = structure.first_lookup.get(response)
        if canonical is None:
            return []
        first_predictions.append(canonical)

    if not structure.second_options:
        return first_predictions

    second_predictions: List[str] = []
    for entity, first_choice in zip(entities, first_predictions):
        options = structure.second_options.get(first_choice.lower())
        if not options:
            return []
        query = build_question(entity, sentence, "second", options)
        response = generate_response(model, tokenizer, query, device, max_new_tokens, temperature).lower()
        canonical = structure.second_lookup[first_choice.lower()].get(response)
        if canonical is None:
            return []
        second_predictions.append(canonical)

    if not structure.third_options:
        return second_predictions

    final_predictions: List[str] = []
    for entity, second_choice in zip(entities, second_predictions):
        options = structure.third_options.get(second_choice.lower())
        if not options:
            return []
        query = build_question(entity, sentence, "third", options)
        response = generate_response(model, tokenizer, query, device, max_new_tokens, temperature).lower()
        canonical = structure.third_lookup[second_choice.lower()].get(response)
        if canonical is None:
            return []
        final_predictions.append(canonical)

    return final_predictions


def load_stage1_responses(responses_dir: Path) -> Dict[str, List[str]]:
    responses: Dict[str, List[str]] = defaultdict(list)
    for path in sorted(responses_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query = data.get("query") or data.get("prompt") or data.get("input")
                response = data.get("response") or data.get("output") or data.get("text")
                if query and response:
                    responses[query].append(response)
    return responses


def process_responses(
    responses: Dict[str, List[str]],
    structure: CategoryStructure,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    temperature: Optional[float],
    limit: int,
) -> Dict[str, Dict[str, List[str]]]:
    results: Dict[str, Dict[str, List[str]]] = {}
    start = time.time()
    example_id = 1

    for idx, (query, response_list) in enumerate(sorted(responses.items()), start=1):
        if not response_list:
            continue

        entity_candidates = [extract_entities_with_positions(query, resp) for resp in response_list]
        merged_entities = merge_entities(entity_candidates)
        if not merged_entities:
            continue

        entity_texts = [entity["text"] for entity in merged_entities]
        categories = categorize_entities(
            entity_texts,
            query,
            structure,
            model,
            tokenizer,
            device,
            max_new_tokens,
            temperature,
        )
        if not categories:
            continue

        results[f"sentence{example_id}"] = {
            "sentence": query,
            "entity": entity_texts,
            "category": categories,
        }
        example_id += 1

        if limit != -1 and len(results) >= limit:
            break

        if idx % 50 == 0:
            elapsed = time.time() - start
            print(f"Processed {idx} prompts | accumulated results: {len(results)} | {elapsed:.1f}s elapsed")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CascadeNER stage-2 categorisation")
    parser.add_argument(
        "--responses_dir",
        type=Path,
        required=True,
        help="Directory containing Stage-1 JSONL response files.",
    )
    parser.add_argument(
        "--classifier_model",
        type=Path,
        required=True,
        help="Path to the classifier model (Stage-2).",
    )
    parser.add_argument(
        "--category_file",
        type=Path,
        required=True,
        help="Hierarchy file describing DynamicNER categories (e.g. DynamicNER_process/DynamicNER.json).",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("outputs/cascade_predictions.json"),
        help="Where to store the aggregated predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens generated per classification query.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature; leave unset for greedy decoding.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional limit on number of sentences to process (-1 means all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    responses_dir: Path = args.responses_dir
    if not responses_dir.exists():
        raise FileNotFoundError(f"Responses directory not found: {responses_dir}")

    responses = load_stage1_responses(responses_dir)
    if not responses:
        raise ValueError(f"No JSONL responses found under {responses_dir}")

    structure = load_category_structure(args.category_file)
    model, tokenizer = load_model_and_tokenizer(args.classifier_model, args.device)

    results = process_responses(
        responses,
        structure,
        model,
        tokenizer,
        args.device,
        args.max_new_tokens,
        args.temperature,
        args.limit,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} sentences to {args.output_file}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
