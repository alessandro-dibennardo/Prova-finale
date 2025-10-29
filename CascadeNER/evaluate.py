import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")


def extract_entities(entities: Iterable[str]) -> Set[str]:
    return {entity.lower() for entity in entities}


def calculate_entity_metrics(true_data: Dict, pred_data: Dict) -> Tuple[float, float, float]:
    true_entities: List[Set[str]] = []
    pred_entities: List[Set[str]] = []

    common_sentences = set(true_data.keys()).intersection(pred_data.keys())

    for sentence in common_sentences:
        true_entities.append(extract_entities(true_data[sentence]["entity"]))
        pred_entities.append(extract_entities(pred_data[sentence]["entity"]))

    if not common_sentences:
        return 0.0, 0.0, 0.0

    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(true_entities)
    y_pred = mlb.transform(pred_entities)

    precision_entity = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_entity = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_entity = f1_score(y_true, y_pred, average="micro", zero_division=0)

    return precision_entity, recall_entity, f1_entity


def calculate_metrics(true_data: Dict, pred_data: Dict) -> Tuple[float, float, float]:
    true_entities_with_categories: List[Set[Tuple[str, str]]] = []
    pred_entities_with_categories: List[Set[Tuple[str, str]]] = []

    common_sentences = set(true_data.keys()).intersection(pred_data.keys())

    for sentence in common_sentences:
        true_entities_with_categories.append(
            {
                (entity.lower(), category.lower())
                for entity, category in zip(
                    true_data[sentence]["entity"], true_data[sentence]["category"]
                )
            }
        )
        pred_entities_with_categories.append(
            {
                (entity.lower(), category.lower())
                for entity, category in zip(
                    pred_data[sentence]["entity"], pred_data[sentence]["category"]
                )
            }
        )

    if not common_sentences:
        return 0.0, 0.0, 0.0

    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(true_entities_with_categories)
    y_pred = mlb.transform(pred_entities_with_categories)

    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    return precision, recall, f1


def load_json(file_path: Path) -> Dict:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main(ground_truth_path: Path, predict_path: Path) -> None:
    ground_truth = load_json(ground_truth_path)
    predict = load_json(predict_path)

    precision, recall, f1 = calculate_metrics(ground_truth, predict)
    precision_entity, recall_entity, f1_entity = calculate_entity_metrics(ground_truth, predict)

    print(f"Overall Precision (Entity + Category): {precision:.4f}")
    print(f"Overall Recall (Entity + Category): {recall:.4f}")
    print(f"Overall F1 Score (Entity + Category): {f1:.4f}")
    print(f"Entity-level Precision: {precision_entity:.4f}")
    print(f"Entity-level Recall: {recall_entity:.4f}")
    print(f"Entity-level F1 Score: {f1_entity:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CascadeNER predictions.")
    parser.add_argument("ground_truth", type=Path, help="Path to the ground-truth JSON file.")
    parser.add_argument("predictions", type=Path, help="Path to the predicted JSON file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.ground_truth, args.predictions)
