from transformers import AutoTokenizer
from pathlib import Path

# Change this to your base model (LLaMA or Mistral)
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
SAVE_DIR = Path("model/tokenizer")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"[✓] Tokenizer saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()
