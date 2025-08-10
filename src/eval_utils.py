# src/eval_utils.py

# === Imports ===
import evaluate
from collections import Counter
import re
import string

# === Text Normalization ===

def normalize_answer(s):
    """
    Lower text, remove punctuation, articles, and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# === Evaluation Metrics ===

def f1_score(prediction, ground_truth):
    """
    Compute the F1 score between prediction and ground truth answers.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction, ground_truth):
    """
    Check if the normalized prediction exactly matches the ground truth.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)