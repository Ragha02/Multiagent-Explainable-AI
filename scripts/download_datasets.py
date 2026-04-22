"""
scripts/download_datasets.py
============================
Download the 3 Kaggle datasets needed by the MA-XAI pipeline.

Run once before starting the API server:
    python3 scripts/download_datasets.py

Credentials: uses ~/.kaggle/kaggle.json
Format:      {"username": "<user>", "key": "<api_key>"}
"""
import os, sys, subprocess

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = [
    ("abhinand05/crop-production-in-india",   "Dataset 1 — India crop production (246k rows)"),
    ("patelris/crop-yield-prediction-dataset", "Dataset 2 — FAO climate features (rainfall, temp)"),
    ("atharvaingle/crop-recommendation-dataset","Dataset 3 — Soil / NPK features (2,200 rows)"),
]

print("\n🌾 MA-XAI — Kaggle Dataset Downloader")
print(f"   Output: {os.path.abspath(DATA_DIR)}\n")

errors = []
for slug, desc in DATASETS:
    print(f"📦 {desc}")
    cmd = ["kaggle", "datasets", "download", "-d", slug, "--unzip", "-p", DATA_DIR]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"   ✓ Done\n")
    else:
        print(f"   ✗ FAILED:\n{r.stderr.strip()}\n")
        errors.append(slug)

print("─" * 55)
if errors:
    print(f"⚠  {len(errors)} download(s) failed:", errors)
    sys.exit(1)
else:
    print(f"✓ All 3 datasets ready. Files in {DATA_DIR}/:")
    for f in sorted(os.listdir(DATA_DIR)):
        mb = os.path.getsize(os.path.join(DATA_DIR, f)) / 1_048_576
        print(f"   {f:<48} {mb:>7.2f} MB")
