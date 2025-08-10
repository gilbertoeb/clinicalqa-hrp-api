# scripts/merge_datasets.py

# === Imports ===
import torch
from pathlib import Path
from datasets import concatenate_datasets

# === Configuration: Dataset Paths ===
SYNTHETIC_PATH = Path("data/processed/synthea_train_dataset.pt")
MIMIC_PATH = Path("data/processed/mimic_train_dataset_v2.pt")
OUTPUT_PATH = Path("data/processed/combined_train_dataset_v2.pt")

# === Main Logic ===

# Load datasets from disk
synthea_ds = torch.load(SYNTHETIC_PATH, weights_only=False)
mimic_ds = torch.load(MIMIC_PATH, weights_only=False)

# Merge datasets using Hugging Face utility
combined_ds = concatenate_datasets([synthea_ds, mimic_ds])

# Save the combined dataset
torch.save(combined_ds, OUTPUT_PATH)
print(f"Combined dataset saved to: {OUTPUT_PATH}")