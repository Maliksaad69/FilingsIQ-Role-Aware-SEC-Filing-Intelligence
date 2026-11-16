
import json
import pandas as pd
from pathlib import Path

# -----------------------------------------
# PATHS
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "Enrichment" else Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "manifest.csv"
INPUT_DIR = BASE_DIR / "extracted"
OUTPUT_DIR = BASE_DIR / "data/enriched"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# 1. LOAD MANIFEST
# -----------------------------------------

manifest = pd.read_csv(MANIFEST_PATH)

# Check for duplicate document IDs
if manifest["doc_id"].duplicated().any():
    duplicates = manifest.loc[
        manifest["doc_id"].duplicated(keep=False), "doc_id"
    ].tolist()
    raise ValueError(f"Duplicate doc_ids in manifest: {duplicates}")

# Build lookup dictionary
manifest_lookup = manifest.set_index("doc_id").to_dict("index")

# -----------------------------------------
# 2. PROCESS ALL JSON FILES
# -----------------------------------------

json_files = list(INPUT_DIR.glob("*.json"))

successful = []
failed = []
missing_metadata = []

for json_path in json_files:

    doc_id = json_path.stem

    try:

        # Find matching manifest metadata
        if doc_id not in manifest_lookup:
            missing_metadata.append(doc_id)
            print(f"[SKIPPED] No manifest entry: {doc_id}")
            continue

        metadata = manifest_lookup[doc_id]

        # Convert NumPy values to Python values
        metadata = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in metadata.items()
        }

        # Load MinerU JSON
        with open(json_path, "r", encoding="utf-8") as f:
            parsed_data = json.load(f)

        # Enrich document
        enriched_data = {
            "doc_id": doc_id,
            "metadata": metadata,
            "parsed_content": parsed_data
        }

        # Save enriched JSON
        output_path = OUTPUT_DIR / json_path.name

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                enriched_data,
                f,
                ensure_ascii=False,
                indent=2
            )

        successful.append(doc_id)

        print(f"[SUCCESS] {doc_id}")

    except Exception as e:

        failed.append({
            "doc_id": doc_id,
            "error": str(e)
        })

        print(f"[FAILED] {doc_id}: {e}")

# -----------------------------------------
# 3. FINAL REPORT
# -----------------------------------------

print("\n" + "=" * 60)
print("ENRICHMENT SUMMARY")
print("=" * 60)

print("Total JSON files:", len(json_files))
print("Successfully enriched:", len(successful))
print("Missing manifest entries:", len(missing_metadata))
print("Failed:", len(failed))

if missing_metadata:
    print("\nMissing metadata:")
    for doc in missing_metadata:
        print("-", doc)

if failed:
    print("\nFailed files:")
    for item in failed:
        print(f"- {item['doc_id']}: {item['error']}")

print("\nOutput directory:", OUTPUT_DIR)
