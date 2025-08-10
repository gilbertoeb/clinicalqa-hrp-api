# src/model_utils.py

# === Imports ===
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# === Model and Tokenizer Utilities ===

def get_tokenizer(model_name: str):
    """
    Load a tokenizer for the specified model name.
    """
    return AutoTokenizer.from_pretrained(model_name)

def get_model(model_name: str):
    """
    Load a question answering model for the specified model name.
    """
    return AutoModelForQuestionAnswering.from_pretrained(model_name)