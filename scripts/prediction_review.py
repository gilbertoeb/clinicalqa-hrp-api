# scripts/prediction_review.py

# === Imports ===
import json
from src.eval_utils import f1_score, exact_match_score
from transformers import AutoTokenizer

# === Configuration ===
PREDICTIONS_PATH = "results/synthea_predictions.jsonl"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# === Functions ===

def load_predictions(path):
    """
    Load prediction examples from a JSONL file.
    """
    with open(path) as f:
        return [json.loads(line) for line in f]

def collect_bad_predictions(examples, f1_threshold=0.9):
    """
    Collect predictions with low F1 or non-exact matches.
    """
    bad_preds = []
    for ex in examples:
        pred = ex["prediction"]
        ref = ex["reference"]
        f1 = f1_score(pred, ref)
        em = exact_match_score(pred, ref)
        if em == 0 or f1 < f1_threshold:
            bad_preds.append({
                "f1": f1,
                "prediction": pred,
                "reference": ref,
                "question": ex["question"],
                "context": ex["context"]
            })
    return bad_preds

def print_top_bad_predictions(bad_preds, n=10):
    """
    Print the top N worst predictions sorted by F1 score.
    """
    bad_preds.sort(key=lambda x: x["f1"])
    for ex in bad_preds[:n]:
        print(f"F1: {ex['f1']:.2f} | Q: {ex['question']}\n→ Pred: {ex['prediction']}\n→ Ref:  {ex['reference']}\n")

def inspect_tokenization(model_name, text):
    """
    Print tokenization of a sample text using the specified model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tokenizer.tokenize(text))

# === Main Logic ===
if __name__ == "__main__":
    # Load predictions from file
    examples = load_predictions(PREDICTIONS_PATH)

    # Collect and display problematic predictions
    bad_preds = collect_bad_predictions(examples)
    print_top_bad_predictions(bad_preds, n=10)

    # Inspect tokenization for a sample medication string
    inspect_tokenization(MODEL_NAME, "1 ML Epoetin Alfa 4000 UNT/ML Injection [Epogen]")