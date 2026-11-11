# Data

The data files are not on GitHub because they are too large (about 620 MB in total).
Create them yourself, or ask the team for a copy of the snapshot of 5 Nov 2026.

| File | Size | Made by | Content |
| --- | --- | --- | --- |
| `models.jsonl` | ~430 MB | `snapshot.py` | Metadata of every Hub model that declares a parent (snapshot of 5 Nov 2026) |
| `roots.jsonl` | ~1 MB | `fetch_roots.py` | Metadata of the root models of meta-llama, mistralai, Qwen and google |
| `edges.csv` | ~75 MB | `analyse.py` | Cleaned network: parent, child, edge type |
| `nodes.csv` | ~85 MB | `analyse.py` | One row per model: families, generation, date, swarm flag, degrees |
| `edges_quant_collapsed.csv` | ~58 MB | `analyse.py` | Network with quantised copies merged into their source model |

To recreate everything (about 1 hour, mostly the snapshot):

```
python3 snapshot.py
python3 fetch_roots.py
python3 analyse.py
```

A new snapshot gives slightly different numbers than ours, because models are added and
deleted on the Hub every day. Use the team's copy to reproduce the results in `results/`.
