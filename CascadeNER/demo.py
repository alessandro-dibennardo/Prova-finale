"""Quick demo script for running a single CascadeNER classification query."""

import argparse
from pathlib import Path

from model.infer import (
    CategoryStructure,
    build_question,
    generate_response,
    load_category_structure,
    load_model_and_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single CascadeNER classification demo.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the classifier model.")
    parser.add_argument("--category_file", type=Path, required=True, help="Path to DynamicNER.json.")
    parser.add_argument("--sentence", type=str, required=True, help="Sentence containing the entity of interest.")
    parser.add_argument("--entity", type=str, required=True, help="Entity mention to classify.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (e.g. cuda or cpu).")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens when querying the model.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structure: CategoryStructure = load_category_structure(args.category_file)
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    question = build_question(args.entity, args.sentence, "first", structure.first_options)
    print("Prompt:", question)
    prediction = generate_response(
        model,
        tokenizer,
        question,
        args.device,
        args.max_new_tokens,
        args.temperature,
    ).lower()

    canonical = structure.first_lookup.get(prediction)
    if canonical is None:
        print(f"Model answered '{prediction}', which does not match the first-level options: {structure.first_options}")
    else:
        print("Prediction:", canonical)


if __name__ == "__main__":
    main()
