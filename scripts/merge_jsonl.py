# scripts/merge_jsonl.py

# === Imports ===
from pathlib import Path
import json

# === Configuration: Input and Output Paths ===
INPUT_PATHS = [
    Path("data/raw/mimic/generated_qa_examples_200_0152.jsonl"),
    Path("data/raw/mimic/merged_generated_qa.jsonl")
]
OUTPUT_PATH = Path("data/raw/mimic/merged_generated_qa_v2.jsonl")

# === Functions ===
def merge_unique_qa_entries(input_paths, output_path):
    """
    Merge multiple JSONL files, keeping only unique (hadm_id, question) pairs.
    """
    seen = set()
    unique_entries = []

    for path in input_paths:
        with open(path, "r") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    key = (obj["hadm_id"], obj["question"])
                    if key not in seen:
                        seen.add(key)
                        unique_entries.append(obj)
                except Exception as e:
                    print(f"Skipping line in {path.name}: {e}")

    with open(output_path, "w") as fout:
        for item in unique_entries:
            fout.write(json.dumps(item) + "\n")

    print(f"Merged {len(unique_entries)} unique QA entries into {output_path}")

# === Main Logic ===
if __name__ == "__main__":
    merge_unique_qa_entries(INPUT_PATHS, OUTPUT_PATH)