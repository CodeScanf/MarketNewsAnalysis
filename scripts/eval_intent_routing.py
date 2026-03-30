#!/usr/bin/env python3
"""Evaluate intent-routing accuracy against a labeled dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from intanalysis.intent import IntentClassifier
from intanalysis.models import QueryIntent


INTENTS = [intent.value for intent in QueryIntent]


def load_examples(path: Path) -> list[dict]:
    """Load examples from JSONL."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            missing = {"id", "query", "expected_intent"} - set(item)
            if missing:
                raise ValueError(f"Line {line_no} missing fields: {sorted(missing)}")
            if item["expected_intent"] not in INTENTS:
                raise ValueError(
                    f"Line {line_no} has invalid expected_intent={item['expected_intent']!r}"
                )
            examples.append(item)
    return examples


def safe_div(numerator: int, denominator: int) -> float:
    """Safely divide two numbers."""
    return numerator / denominator if denominator else 0.0


def compute_metrics(rows: list[dict]) -> dict:
    """Compute overall and per-class metrics."""
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])

    confusion: dict[str, Counter] = {intent: Counter() for intent in INTENTS}
    for row in rows:
        confusion[row["expected"]][row["predicted"]] += 1

    per_class = {}
    f1_values = []
    for intent in INTENTS:
        tp = confusion[intent][intent]
        fp = sum(confusion[other][intent] for other in INTENTS if other != intent)
        fn = sum(confusion[intent][other] for other in INTENTS if other != intent)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        per_class[intent] = {
            "support": sum(confusion[intent].values()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)

    by_difficulty: dict[str, dict[str, int | float]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("difficulty", "unknown")].append(row)
    for difficulty, items in grouped.items():
        by_difficulty[difficulty] = {
            "count": len(items),
            "accuracy": safe_div(sum(1 for item in items if item["correct"]), len(items)),
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": safe_div(correct, total),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "by_difficulty": by_difficulty,
    }


def print_report(metrics: dict, rows: list[dict], show_errors: int) -> None:
    """Print a readable evaluation report."""
    print("Intent Routing Evaluation")
    print("=" * 80)
    print(f"Examples: {metrics['total']}")
    print(f"Correct:  {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro F1: {metrics['macro_f1']:.2%}")
    print()

    print("Per-Class Metrics")
    print("-" * 80)
    for intent in INTENTS:
        item = metrics["per_class"][intent]
        print(
            f"{intent:16} support={item['support']:>3}  "
            f"precision={item['precision']:.2%}  recall={item['recall']:.2%}  f1={item['f1']:.2%}"
        )
    print()

    print("Accuracy by Difficulty")
    print("-" * 80)
    for difficulty in sorted(metrics["by_difficulty"]):
        item = metrics["by_difficulty"][difficulty]
        print(f"{difficulty:8} count={item['count']:>3}  accuracy={item['accuracy']:.2%}")
    print()

    print("Confusion Matrix")
    print("-" * 80)
    header = "expected \\ predicted".ljust(22) + "".join(pred.ljust(18) for pred in INTENTS)
    print(header)
    for expected in INTENTS:
        row = expected.ljust(22)
        for predicted in INTENTS:
            row += str(metrics["confusion"][expected][predicted]).ljust(18)
        print(row)
    print()

    errors = [row for row in rows if not row["correct"]]
    if errors and show_errors:
        print(f"First {min(show_errors, len(errors))} Errors")
        print("-" * 80)
        for row in errors[:show_errors]:
            print(f"[{row['id']}] expected={row['expected']} predicted={row['predicted']} source={row['source']}")
            print(f"query: {row['query']}")
            if row.get("reason"):
                print(f"reason: {row['reason']}")
            if row.get("note"):
                print(f"note: {row['note']}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate intent-routing accuracy.")
    parser.add_argument(
        "--dataset",
        default="dataset/intent_routing_eval.jsonl",
        help="Path to a JSONL file with id/query/expected_intent fields.",
    )
    parser.add_argument(
        "--rule-only",
        action="store_true",
        help="Disable LLM fallback and evaluate rules only.",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=15,
        help="How many misclassified examples to print.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    examples = load_examples(dataset_path)
    classifier = IntentClassifier(use_llm=not args.rule_only)

    rows = []
    for item in examples:
        decision = classifier.classify(item["query"])
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "expected": item["expected_intent"],
                "predicted": decision.intent.value,
                "source": decision.source,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "difficulty": item.get("difficulty", "unknown"),
                "note": item.get("note", ""),
                "correct": item["expected_intent"] == decision.intent.value,
            }
        )

    metrics = compute_metrics(rows)
    print_report(metrics, rows, args.show_errors)


if __name__ == "__main__":
    main()
