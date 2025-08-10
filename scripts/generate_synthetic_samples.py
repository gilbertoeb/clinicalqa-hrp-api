# scripts/generate_synthetic_samples.py

# === Imports ===
import json
import random
import uuid
from pathlib import Path

# === Configuration and Vocabularies ===
# These vocabularies can be expanded or modified as needed.
DRUGS = [
    "Metformin", "Gabapentin", "Epoetin Alfa", "Amoxicillin",
    "Lisinopril", "Albuterol", "Acetaminophen", "Atorvastatin", "Oxycodone Hydrochloride"
]
DOSAGES = ["0.5 MG", "10 MG", "500 MG", "1 ML", "2 ML", "1000 UNT"]
FORMS = ["Oral Tablet", "Injection", "Suspension", "Capsule", "Auto-Injector"]
BRANDS = [
    "[Epogen]", "[Percocet]", "[Lipitor]", "[Neurontin]", "[Ventolin]",
    "[Glucophage]", "[Zestril]", "[Amoxil]", "[Norvasc]", "[Humulin]"
]

OUTPUT_PATH = Path("data/processed/synthea_train_augmented.jsonl")

# === Functions ===
def generate_sample():
    """
    Generate a synthetic medication-related QA sample.
    """
    drug = random.choice(DRUGS)
    dose = random.choice(DOSAGES)
    form = random.choice(FORMS)
    brand = random.choice(BRANDS)
    medication = f"{dose} {drug} {form} {brand}"
    context = f"Patient {str(uuid.uuid4())} was prescribed {medication}."
    question = "What medication was the patient prescribed?"
    answer_start = context.index(medication)
    return {
        "context": context,
        "question": question,
        "answer_text": medication,
        "answer_start": answer_start
    }

def main(n_samples=120):
    """
    Generate and save synthetic QA samples to a JSONL file.
    """
    samples = [generate_sample() for _ in range(n_samples)]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {n_samples} samples to {OUTPUT_PATH}")

# === Main Logic ===
if __name__ == "__main__":
    main()