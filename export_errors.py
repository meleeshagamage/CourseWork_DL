
import csv
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from factcheck import EntailmentModel, EntailmentFactChecker
from factchecking_main import read_passages, read_fact_examples


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
OUTPUT_DIR = "results"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading DeBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

ent_model = EntailmentModel(
    model,
    tokenizer,
    False
)

checker = EntailmentFactChecker(ent_model)
# Phase 2 pruning experiment
checker.pruning_threshold = 0.20
checker.entailment_threshold = 0.40

print("Model loaded successfully!")


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

passages = read_passages(PASSAGES_PATH)

examples = read_fact_examples(
    LABELS_PATH,
    passages
)

print(f"Total examples: {len(examples)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# GENERATE PREDICTIONS
# ==========================================

rows = []

for i, example in enumerate(examples, start=1):

    gold = "NS" if example.label == "IR" else example.label

    prediction = checker.predict(
        example.fact,
        example.passages
    )

    correct = gold == prediction

    if gold == "NS" and prediction == "S":
        error_type = "False Positive"

    elif gold == "S" and prediction == "NS":
        error_type = "False Negative"

    else:
        error_type = ""

    rows.append({
        "example_id": i,
        "fact": example.fact,
        "gold_label": gold,
        "predicted_label": prediction,
        "correct": correct,
        "error_type": error_type,
        "best_entailment_score": getattr(
            checker,
            "last_best_score",
            0.0
        ),
        "best_evidence": getattr(
            checker,
            "last_best_evidence",
            ""
        ),
        "num_passages": len(example.passages)
    })

    if i % 25 == 0 or i == len(examples):
        print(f"Processed {i}/{len(examples)} examples")


# ==========================================
# CALCULATE RESULTS
# ==========================================

total = len(rows)

correct_count = sum(
    1 for row in rows if row["correct"]
)

accuracy = correct_count / total if total else 0.0

false_positives = [
    row for row in rows
    if row["error_type"] == "False Positive"
]

false_negatives = [
    row for row in rows
    if row["error_type"] == "False Negative"
]

errors = false_positives + false_negatives


print("\n========== PHASE 2 RESULTS ==========")

print(f"Total Examples: {total}")
print(f"Correct Predictions: {correct_count}")
print(f"Incorrect Predictions: {len(errors)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"False Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")

print("=====================================\n")


# ==========================================
# EXPORT CSV FILES
# ==========================================

outputs = {
    "phase2_all_predictions.csv": rows,
    "phase2_errors.csv": [
        row for row in rows if row["error_type"]
    ],
    "phase2_false_positives.csv": false_positives,
    "phase2_false_negatives.csv": false_negatives
}

fields = [
    "example_id",
    "fact",
    "gold_label",
    "predicted_label",
    "correct",
    "error_type",
    "best_entailment_score",
    "best_evidence",
    "num_passages"
]

for filename, data in outputs.items():

    path = os.path.join(OUTPUT_DIR, filename)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(data)

    print(f"Saved {len(data)} rows: {path}")


print("\nCSV EXPORT COMPLETED SUCCESSFULLY!")
