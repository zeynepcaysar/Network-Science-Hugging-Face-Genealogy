# Hugging Face Model Genealogy

We study how the Hugging Face model genealogy grows. Each node is a model; each edge points
from a parent model to a model built from it, typed as fine-tune, adapter, quantisation or
merge and dated by the child's upload date. We compare the Llama, Mistral, Qwen and Gemma
families.

| RQ | Question | Folder |
| --- | --- | --- |
| RQ1 Structure | How are children counts distributed, and do the families branch out wide or grow in long chains? | `rq1_structure/` |
| RQ2 Growth | Do models gain children because they are already popular (preferential attachment) or because of the model itself (fitness)? | `rq2_growth/` |
| RQ3 Edge types | How do fine-tune, adapter, quantisation and merge links differ, and do merges connect families? | `rq3_edge_types/` |
| RQ4 Robustness | When models are removed at random, or the biggest hubs first, how much of each family loses its link to the base model? | `rq4_robustness/` |

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/); it supplies Python 3.13 on
Windows, macOS and Linux (do not use the Microsoft Store Python). Then, in the repository folder
(Git Bash on Windows):

```
bash setup.sh                       # creates .venv with Python 3.13, installs the pinned packages and load.py
source .venv/bin/activate           # macOS / Linux
source .venv/Scripts/activate       # Git Bash on Windows
.venv\Scripts\Activate.ps1          # PowerShell on Windows
```

`constraints.txt` pins every package; after a change to `requirements.txt`, regenerate it with the
command in its header. With the environment active:

```
python -m pytest -m "not data and not slow"   # runs without the data files; use python -m pytest, not a bare pytest
python load.py --verify                       # after copying the data: md5 check against MANIFEST.txt
```

The data files are not stored on GitHub. The frozen dataset (v1) is in the team's shared
folder `data-v1` and goes into `data/` (see [`data/README.md`](data/README.md)).

## Loading the network

`load.py` is the shared entry point to the frozen data and works from any folder:

```python
from load import load, load_tables

G = load()                      # full cleaned network (networkx.DiGraph, parent -> child)
G = load("no_swarm")            # without the automated adapter swarms
G = load("quant_collapsed")     # quantised copies merged into their source model
G = load(aliases=True)          # + undeclared re-uploads linked to the official model
G = load(families=["Qwen"])     # one family only
nodes, edges = load_tables()    # the same as pandas DataFrames
```

Results are reported on all three versions (`full`, `no_swarm`, `quant_collapsed`). Node and edge
attributes are listed at the top of `load.py`. The CSVs are read with `dtype=str,
keep_default_na=False`, since some model ids (`None`, `NA`) would otherwise become missing values.

The first load of each version (with or without aliases) writes a parquet cache to `data/.cache/` (about
44 MB each, up to about 220 MB; git-ignored through `data/*`). Later loads take well under a second. The cache
rebuilds itself when `MANIFEST.txt` changes or a file is damaged; delete the folder to reset it, or set
`HFG_CACHE=off` to bypass it.

## Repository layout

| Folder | Content |
| --- | --- |
| `data_pipeline/` | Builds the network: snapshot, roots, renames, file lists, cleaning (`analyse.py`) |
| `checks/` | Quality checks: edge check, collapsed network, bulk uploaders, live counts |
| `load.py` | Shared loader for the frozen data |
| `rq1_structure/` ... `rq4_robustness/` | One folder per research question; results in `results/rqX/` |
| `results/data/` | Outputs of the data pipeline and the checks |
| `data/` | Data files (not on GitHub) |

## Building the data

| Step | Script | Internet | Time | Output |
| --- | --- | --- | --- | --- |
| 1. Snapshot of the Hub | `data_pipeline/snapshot.py` | yes | 30-60 min | `data/models.jsonl` |
| 2. Root models | `data_pipeline/fetch_roots.py` | yes | ~1 min | `data/roots.jsonl` |
| 3. Renamed official ids | `data_pipeline/fetch_renames.py` | yes | seconds | `data_pipeline/renames.csv` |
| 4. File names per model | `data_pipeline/fetch_files.py` | yes | ~20-60 min | `data/files.jsonl` |
| 5. Build and clean | `data_pipeline/analyse.py` | no | ~3 min | `data/*.csv`, `data/MANIFEST.txt`, `results/data/` |
| 6. Checks | `checks/check_quant_collapse.py`, `checks/hub_authors.py`, `checks/edge_check.py` | no | ~1 min | `results/data/` |

Steps 1-4 query the live Hub, so rerunning them gives a different dataset. The frozen v1 is
rebuilt from its inputs with step 5 only.

The snapshot of 5 Nov 2026 scanned 3,216,156 models and keeps every model that declares a parent
(`base_model` tag), plus every model of meta-llama, mistralai, Qwen and google.

## How the network is built (`data_pipeline/analyse.py`)

Counts of every step are in `results/data/cleaning.txt`.

- **Edges** come from the `base_model` tags of each child; the tag gives the type.
- **Parent ids are cleaned**: case-only variants are merged (the Hub treats them as one model);
  spaces are stripped; local paths that contain a Hub id (`/cache/models/Qwen--Qwen2.5-3B-Instruct`,
  `.../models--org--name/snapshots/...`) and `org--name` spellings are mapped to that id if it is a
  known model; other local paths and `None` are dropped. Official models renamed on the Hub
  (`meta-llama/Meta-Llama-3.1-8B-Instruct` -> `meta-llama/Llama-3.1-8B-Instruct`) are merged
  (`data_pipeline/renames.csv`); otherwise children that declare both ids look like merges.
- **Edge types are corrected from the model files** (`data/files.jsonl`). For edges tagged
  `finetune` or `unknown`: an `adapter_config.json` without full weights makes it an adapter (with
  full weights it is a LoRA already merged into a full model, so it stays a fine-tune), a
  `mergekit_config.yml` makes it a merge. Only `.gguf` weights make an `unknown` edge a quantisation;
  a `finetune` or `merge` edge only if the child's name is the parent's plus a quantisation suffix
  (`Llama-3.2-1B-Instruct-Q4_K_M-GGUF`), because fine-tuned or abliterated models are often released
  only as GGUF. `adapter` and `quantized` tags are kept. The declared tag stays in `rel_declared`;
  `results/data/retyping.csv` counts the changes.
- **Removed**: self-loops, edges where the child is more than 1 day older than its parent (base and
  instruct models are often uploaded minutes apart, in either order), and cycles. The 2022-03-02
  placeholder date is treated as missing.
- **Families**: every model reachable from a root of meta-llama, mistralai, Qwen or Google's Gemma
  line (a root has no declared parent). `generation` = shortest distance to a root. A model reachable
  from two families belongs to both.
- **Swarm flag**: a hub with >= 5,000 children, >= 95% adapters, top-5 uploaders >= 35% and >= 20
  children per uploader is a swarm hub (`results/data/swarm_hubs.csv`). Its adapter children
  without children of their own get `is_swarm = True`.
- **Quantisation-collapsed version**: each quantised copy with one parent is merged into its source
  model and its children are re-attached there (Horwitz et al. 2025); tested by `checks/check_quant_collapse.py`.
- **Aliases (sensitivity check)**: a model with no declared parent, outside the families, whose
  repo name equals an official model's (`NousResearch/Meta-Llama-3-8B` = `meta-llama/Meta-Llama-3-8B`)
  is linked to it with a `reupload` edge in `data/edges_aliases.csv` (`results/data/aliases.csv`).
  Not part of the main network.

## Edge check

A sample of 202 edges (50 per family, every edge type) is checked against the model pages
(`results/data/edge_check_sample.csv`). `results/data/edge_check_auto.csv` holds the same edges
with an automatic check against the model files; rows marked `?` are checked by hand.

```
python3 checks/edge_check.py score results/data/edge_check_auto.csv   # error rates with 95% CIs
python3 checks/edge_check.py retype                                   # corrected types vs model files
```
