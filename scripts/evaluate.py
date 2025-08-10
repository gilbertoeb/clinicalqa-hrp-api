# scripts/evaluate.py

# === Imports ===
import os
import json
import string
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
from src.eval_utils import f1_score, exact_match_score

# === Constants and Paths ===
MODEL_PATH = "models/clinicalbert-qa-synthea"
VAL_DATA_PATH = "data/raw/testing/real_qa_test.jsonl"
RESULTS_DIR = "results"
PREDS_PATH = os.path.join(RESULTS_DIR, "real_predictions.jsonl")
METRICS_PATH = os.path.join(RESULTS_DIR, "real_eval_results.json")

# === Ensure output directory exists ===
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Utility Functions ===
def normalize_text(s):
    """Lowercase, remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def simple_normalize(s):
    """Lowercase and strip whitespace."""
    return s.lower().strip()

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

# === Load validation data ===
val_dataset = load_dataset("json", data_files=VAL_DATA_PATH)["train"]

# === Create QA pipeline ===
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# === Generate predictions and compute metrics ===
preds = []
refs = []
f1s = []
ems = []

for ex in val_dataset:
    result = qa_pipeline({
        "context": ex["context"],
        "question": ex["question"]
    })
    pred_text = simple_normalize(result["answer"])
    ref_text = simple_normalize(ex["answer_text"])
    preds.append({
        "id": ex.get("id", ""),
        "prediction_text": pred_text
    })
    refs.append({
        "id": ex.get("id", ""),
        "answers": {"answer_start": [0], "text": [ref_text]}
    })
    f1s.append(f1_score(result["answer"], ex["answer_text"]))
    ems.append(exact_match_score(result["answer"], ex["answer_text"]))

# === Save predictions to file ===
with open(PREDS_PATH, "w") as f:
    for ex, pred in zip(val_dataset, preds):
        entry = {
            "context": ex["context"],
            "question": ex["question"],
            "prediction": pred["prediction_text"],
            "reference": simple_normalize(ex["answer_text"])
        }
        f.write(json.dumps(entry) + "\n")
print(f"Saved predictions to {PREDS_PATH}")

# === Save evaluation metrics ===
metrics = {
    "exact_match": round(100 * sum(ems) / len(ems), 2),
    "f1_score": round(100 * sum(f1s) / len(f1s), 2),
    "num_eval_examples": len(ems)
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved evaluation metrics to {METRICS_PATH}")

# === Print summary to console ===
print("\nEvaluation Results:")
print(f"Exact Match: {metrics['exact_match']}%")
print(f"F1 Score: {metrics['f1_score']}%")