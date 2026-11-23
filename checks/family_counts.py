"""Quick check (~2 min): direct children per edge type for the main base models.
Run:  python3 checks/family_counts.py   -> prints a table and writes results/data/family_counts.csv
Note: queries the live Hub, not the snapshot, so counts drift from the frozen data.
"""
import csv, os
from pathlib import Path
from huggingface_hub import HfApi

os.chdir(Path(__file__).resolve().parent.parent)          # repo root, so relative paths work from anywhere

BASES = [
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b", "google/gemma-2-9b-it",
]
RELS = ["finetune", "adapter", "quantized", "merge"]

api = HfApi()
rows = []
for base in BASES:
    row = {"base_model": base}
    for rel in RELS:
        row[rel] = sum(1 for _ in api.list_models(filter=f"base_model:{rel}:{base}"))
    row["total"] = sum(row[r] for r in RELS)
    rows.append(row)
    print(f"{base:40s} " + "  ".join(f"{r}={row[r]}" for r in RELS + ["total"]), flush=True)

os.makedirs("results/data", exist_ok=True)
with open("results/data/family_counts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["base_model"] + RELS + ["total"])
    w.writeheader(); w.writerows(rows)
print("wrote results/data/family_counts.csv")
