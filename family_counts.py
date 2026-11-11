"""Quick check (~2 min): direct children per edge type for the main base models.
Run:  python3 family_counts.py      -> prints a table and writes results/family_counts.csv
"""
import csv, os
from huggingface_hub import HfApi

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

os.makedirs("results", exist_ok=True)
with open("results/family_counts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["base_model"] + RELS + ["total"])
    w.writeheader(); w.writerows(rows)
print("wrote results/family_counts.csv")
