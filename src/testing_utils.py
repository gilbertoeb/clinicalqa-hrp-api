# src/testing_utils.py

# === Imports ===
import json

# === Configuration ===
INPUT_PATH = "data/processed/synthea_qa.jsonl"
OUTPUT_PATH = "data/processed/synthea_qa_cleaned.jsonl"

# === Main Logic ===

def fix_answer_spans(input_path, output_path):
    """
    Fix answer_start indices if the answer text does not match the context span.
    """
    fixed, skipped = 0, 0
    with open(input_path) as f, open(output_path, "w") as out_f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            context = ex["context"]
            answer = ex["answer_text"]
            start = ex["answer_start"]

            # Check if the answer span matches the context
            end = start + len(answer)
            if context[start:end] != answer:
                idx = context.find(answer)
                if idx == -1:
                    print(f"[Skipped] Answer not found in context at line {i}")
                    skipped += 1
                    continue
                ex["answer_start"] = idx
                fixed += 1

            out_f.write(json.dumps(ex) + "\n")
    print(f"Fixed {fixed} entries. Skipped {skipped}. Saved to {output_path}")

if __name__ == "__main__":
    fix_answer_spans(INPUT_PATH, OUTPUT_PATH)