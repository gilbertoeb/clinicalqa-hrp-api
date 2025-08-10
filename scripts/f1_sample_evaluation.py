# scripts/f1_sample_evaluation.py

# === Imports ===
import json

# === Functions ===
def compute_em_f1(samples):
    """
    Compute exact match (EM) and F1 score for a list of prediction-reference pairs.
    """
    em, f1 = 0, 0
    for s in samples:
        pred = s["prediction"].strip().lower()
        ref = s["reference"].strip().lower()
        em += int(pred == ref)

        pred_tokens = pred.split()
        ref_tokens = ref.split()
        common = set(pred_tokens) & set(ref_tokens)

        prec = len(common) / len(pred_tokens) if pred_tokens else 0
        rec = len(common) / len(ref_tokens) if ref_tokens else 0
        f1 += (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0

    n = len(samples)
    return (em / n * 100, f1 / n * 100) if n > 0 else (0, 0)

# === Main Logic ===

# Load predictions from file
with open("results/synthea_predictions.jsonl") as f:
    preds = [json.loads(line) for line in f]

# Filter entries where the reference contains brackets (e.g., medications)
bracketed_preds = [
    p for p in preds
    if "[" in p["reference"] and "]" in p["reference"]
]

print(f"Bracketed subset size: {len(bracketed_preds)}")
em, f1 = compute_em_f1(bracketed_preds)
print(f"Bracketed EM: {em:.2f}")
print(f"Bracketed F1: {f1:.2f}")