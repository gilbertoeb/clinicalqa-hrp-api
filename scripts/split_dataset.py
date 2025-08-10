# scripts/split_dataset.py

# === Imports ===
import json
import random

# === Configuration ===
DATA_PATH = "data/processed/synthea_qa.jsonl"
TRAIN_PATH = "data/processed/synthea_train.jsonl"
VAL_PATH = "data/processed/synthea_val.jsonl"
TRAIN_SPLIT = 0.9
SEED = 42

# === Main Logic ===

# Set random seed for reproducibility
random.seed(SEED)

# Load all examples from the JSONL file
with open(DATA_PATH, "r") as f:
    lines = [json.loads(l) for l in f]

# Shuffle and split the dataset
random.shuffle(lines)
split_idx = int(TRAIN_SPLIT * len(lines))
train = lines[:split_idx]
val = lines[split_idx:]

# Save the training set
with open(TRAIN_PATH, "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")

# Save the validation set
with open(VAL_PATH, "w") as f:
    for ex in val:
        f.write(json.dumps(ex) + "\n")

# Print summary of split sizes
print(f"Train: {len(train)} | Validation: {len(val)}")