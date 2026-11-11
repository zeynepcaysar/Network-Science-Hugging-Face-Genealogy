# Hugging Face Model Genealogy

Network Science (HS 2026, University of Zurich), team project.

We study how the Hugging Face model genealogy grows. Each node is a model; each edge points
from a parent model to a model built from it, typed as fine-tune, adapter, quantisation or
merge and dated by the child's upload date. We compare the Llama, Mistral, Qwen and Gemma
families and ask whether models gain children through preferential attachment (popularity)
or fitness (quality).

## Setup

```
bash setup.sh                 # once: creates .venv and installs requirements.txt
source .venv/bin/activate     # in every new terminal
```

The data files are not in this repository (too large). See [`data/README.md`](data/README.md).

## Pipeline

| Step | Script | Time | Output |
| --- | --- | --- | --- |
| 1. Snapshot of the Hub | `snapshot.py` | 30-60 min | `data/models.jsonl` |
| 2. Dates of root models | `fetch_roots.py` | ~1 min | `data/roots.jsonl` |
| 3. Build and clean the network | `analyse.py` | ~3 min | `data/edges.csv`, `data/nodes.csv`, `results/` |
| 4. Uploaders of the biggest hubs | `hub_authors.py` | ~1 min | `results/hub_authors.csv` |
| 5. Test the collapsed network | `check_quant_collapse.py` | ~30 s | `results/quant_collapse_check.txt` |
| 6. Hand check of edges | `edge_check.py sample` / `score` | - | `results/edge_check_*.csv` |
| Direct children of 8 base models | `family_counts.py` | ~2 min | `results/family_counts.csv` |

`python3 snapshot.py --max 20000` is a one-minute test run. `fetch_roots.py --all-missing`
also fetches every other missing parent (~46k calls, several hours; optional).

## How the network is cleaned (`analyse.py`)

Counts of every step are in `results/cleaning.txt`.

- Model IDs that differ only by letter case are merged (the Hub treats them as one model).
- The placeholder date 2022-03-02 is treated as missing.
- Removed: self-loops, edges where the child is more than 1 day older than its parent
  (base and instruct models are often uploaded minutes apart, in either order), and cycles.
- **Swarm flag:** a hub with >= 5,000 children, >= 95% adapters, top-5 uploaders >= 35% and
  >= 20 children per uploader is a swarm hub (`results/swarm_hubs.csv`). Its adapter children
  without children of their own get `is_swarm = True` in `data/nodes.csv`.

`results/family_summary.csv` has three network versions per family (column `version`):
`full`, `no_swarm` and `quant_collapsed` (quantised copies merged into their source).

## Edge check

`edge_check.py sample` draws 202 edges (50 per family, every edge type) for one person
to check. They check each row on the model pages and fill in
`parent_ok`, `type_ok` and `notes`; `edge_check.py score` then gives error rates with 95%
confidence intervals. `results/edge_check_auto.csv` is an automatic pre-check of the same
edges against the model files: rows that are not a clear `y` still need a human check.

## Results so far (snapshot of 5 Nov 2026)

| File | Content |
| --- | --- |
| `cleaning.txt` | What each cleaning step removed |
| `family_summary.csv` | Size, edge-type mix, depth, hubs and merges per family and version |
| `swarm_hubs.csv`, `hub_authors.csv`, `top_authors_by_family.csv` | Automated uploads (bot check) |
| `quant_collapse_check.txt` | Tests of the quantisation-collapsed network |
| `edge_check_score_auto.txt` | Error rates of the automatic edge check |
| `ccdf_out_degree*.png` | Distribution of children per model, by family |
