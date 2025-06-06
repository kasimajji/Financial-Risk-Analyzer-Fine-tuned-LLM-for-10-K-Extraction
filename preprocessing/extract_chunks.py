import os
import json
from pathlib import Path
import re

RAW_DIR = Path("./data/raw_filings")
OUTPUT_PATH = Path("./data/extracted_sections/parsed_chunks.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    """Normalize whitespace in extracted text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def parse_metadata(filename):
    """Extract company and year from filename like 8670_10K_2021_....json"""
    parts = filename.replace(".json", "").split("_")
    if len(parts) >= 3:
        return parts[0], parts[2]
    return "Unknown", "Unknown"

def main():
    parsed = 0
    skipped = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        for file in os.listdir(RAW_DIR):
            if not file.endswith(".json"):
                continue

            filepath = RAW_DIR / file
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[!] Skipping {file} — JSON parsing error.")
                    skipped += 1
                    continue

            company, year = parse_metadata(file)
            item_1a = clean_text(data.get("item_1A", ""))
            item_7 = clean_text(data.get("item_7", ""))

            if not item_1a and not item_7:
                print(f"[!] Skipping {file} — both sections empty.")
                skipped += 1
                continue

            result = {
                "filename": file,
                "company": company,
                "year": year,
                "risk_section": item_1a,
                "mdna_section": item_7
            }

            outfile.write(json.dumps(result) + "\n")
            parsed += 1
            print(f"[✓] Parsed {file}")

    print(f"\n✅ Finished: {parsed} files parsed, {skipped} files skipped.")

if __name__ == "__main__":
    main()
