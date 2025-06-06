import json
from pathlib import Path

INPUT_PATH = Path("./data/labeled_samples.jsonl")
OUTPUT_PATH = Path("./data/lora_formatted/train.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

train_samples = []

with open(INPUT_PATH, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)
        prompt = (
            "Given the following risk section, extract the risk type, severity, and evidence in structured format:\n\n"
            f"{data['source_text']}"
        )
        response = {
            "Risk_Category": data["Risk_Category"],
            "Severity": data["Severity"],
            "Evidence": data["Evidence"]
        }
        train_samples.append({
            "prompt": prompt,
            "response": response
        })

with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
    for item in train_samples:
        json.dump(item, outfile)
        outfile.write("\n")

print(f"[✓] Generated: {OUTPUT_PATH}")
