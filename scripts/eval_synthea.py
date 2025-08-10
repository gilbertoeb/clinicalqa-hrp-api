# scripts/eval_synthea.py

# === Imports ===
import os
import json
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
from src.eval_utils import f1_score, exact_match_score

# === Constants and Paths ===
MODEL_PATH = "models/clinicalbert-qa-mixed-v3"  # Path to trained model
INPUT_PATH = "data/raw/synthea/synthea_val.jsonl"  # Synthea evaluation set
PREDS_PATH = "results/real_predictionsv3_synthea.jsonl"
METRICS_PATH = "results/real_eval_resultsv3_synthea.json"

# === Ensure output directory exists ===
os.makedirs("results", exist_ok=True)

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

# === Load evaluation dataset ===
eval_dataset = load_dataset("json", data_files={"eval": INPUT_PATH})["eval"]

# === Create QA pipeline ===
qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

# === Run predictions and compute metrics ===
preds = []
f1s = []
ems = []

for ex in eval_dataset:
    result = qa({
        "context": ex["context"],
        "question": ex["question"]
    })
    pred = result["answer"]
    ref = ex["answer_text"]

    preds.append({
        "context": ex["context"],
        "question": ex["question"],
        "prediction": pred,
        "reference": ref
    })

    f1s.append(f1_score(pred, ref))
    ems.append(exact_match_score(pred, ref))

# === Save predictions to file ===
with open(PREDS_PATH, "w") as f:
    for p in preds:
        f.write(json.dumps(p) + "\n")
print(f"Saved predictions to: {PREDS_PATH}")

# === Calculate and save evaluation metrics ===
metrics = {
    "exact_match": round(100 * sum(ems) / len(ems), 2),
    "f1_score": round(100 * sum(f1s) / len(f1s), 2),
    "num_eval_examples": len(ems)
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved eval results to: {METRICS_PATH}")

# === Print summary to console ===
print("\nEvaluation Results:")
print(f"Exact Match: {metrics['exact_match']}%")
print(f"F1 Score: {metrics['f1_score']}%")