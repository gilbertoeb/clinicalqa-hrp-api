# scripts/extract_radiology_samples.py

# === Imports ===
import pandas as pd
import json
from pathlib import Path

# === Configuration ===
CSV_PATH = Path("data/raw/radiology/mimic_cxr_reports_parsed.csv")
OUTPUT_PATH = Path("data/raw/radiology/radiology_sampled_400.jsonl")
NUM_SAMPLES = 400  # Adjust to extract 300–500 reports
RANDOM_SEED = 42

# === Keywords to identify informative reports ===
CLINICAL_KEYWORDS = [
    "consolidation", "atelectasis", "effusion", "opacity", "edema", "pneumothorax",
    "infiltrate", "cardiomegaly", "pleural", "mass", "lesion", "concerning", "abnormal",
    "airspace", "collapse", "interstitial", "fibrosis", "nodule", "infection"
]

# === Functions ===
def is_informative(report):
    """Return True if the report is informative based on length and keywords."""
    if not isinstance(report, str) or len(report) < 300:
        return False
    report_lower = report.lower()
    return "impression" in report_lower or any(kw in report_lower for kw in CLINICAL_KEYWORDS)

# === Main Logic ===

# Load CSV data
df = pd.read_csv(CSV_PATH)

# Filter for informative reports
filtered = df[df["report"].apply(is_informative)].copy()
print(f"Found {len(filtered)} informative reports")

# Sample N unique reports
sample_size = min(NUM_SAMPLES, len(filtered))
sampled = filtered.sample(sample_size, random_state=RANDOM_SEED)

# Ensure output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save sampled reports to JSONL
with open(OUTPUT_PATH, "w") as f:
    for _, row in sampled.iterrows():
        record = {
            "subject_id": row.get("subject_id", None),
            "study_id": row.get("study_id", None),
            "report": row["report"]
        }
        f.write(json.dumps(record) + "\n")

print(f"Saved {sample_size} radiology reports to {OUTPUT_PATH}")