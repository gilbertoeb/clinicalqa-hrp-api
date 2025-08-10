# scripts/preprocess_mimic.py

# === Imports ===
import json
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset
from src.data_utils import prepare_train_features

# === Configuration: Load YAML Config ===
with open("configs/train_config.yaml") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model_name"]
DATA_PATH = config["data_path_mimic_2"]
MAX_LENGTH = config["max_length"]
DOC_STRIDE = config["doc_stride"]
SAVE_PATH = Path("data/processed/mimic_train_dataset_v2.pt")

# === Initialize Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === Functions ===

def squeeze_batch_dims(example):
    """
    Squeeze batch dimension for tensor fields if present.
    """
    for k in ["input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions"]:
        example[k] = torch.tensor(example[k])
        if example[k].dim() == 2:
            example[k] = example[k].squeeze(0)
    return example

def normalize_dtype(example):
    """
    Normalize data types for all fields to ensure consistency.
    """
    return {
        "input_ids": [int(x) for x in example["input_ids"]],
        "token_type_ids": [int(x) for x in example["token_type_ids"]],
        "attention_mask": [int(x) for x in example["attention_mask"]],
        "start_positions": int(example["start_positions"]),
        "end_positions": int(example["end_positions"]),
    }

# === Main Logic ===

# Load JSONL QA samples
with open(DATA_PATH, "r") as f:
    data = [json.loads(line) for line in f]

# Wrap as Hugging Face Dataset
raw_dataset = Dataset.from_list(data)

# Tokenize and align answer spans
tokenized_dataset = raw_dataset.map(
    lambda x: prepare_train_features(x, tokenizer, config),
    batched=True,
    remove_columns=raw_dataset.column_names
)

# Check shape and type sanity before saving
sample = tokenized_dataset[0]
for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape} ({v.dtype})")
    else:
        print(f"{k}: type={type(v)}")

# Normalize and squeeze batch dimensions
tokenized_dataset = tokenized_dataset.map(normalize_dtype)
tokenized_dataset = tokenized_dataset.map(squeeze_batch_dims)
tokenized_dataset.set_format("torch")

# Ensure output directory exists and save dataset
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(tokenized_dataset, SAVE_PATH)
print(f"Saved tokenized dataset to: {SAVE_PATH}")