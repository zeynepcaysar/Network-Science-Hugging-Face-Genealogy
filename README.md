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

Upload dates that v1 lacks (about 45,000 parents without metadata) can be added from a one-time Hub fetch that was frozen
with an md5 check. This is opt-in: `load()`, `load_tables()` and `load_net()` never change, and the helper only fills dates v1
does not have, never replaces one. See [`data_supplement/README.md`](data_supplement/README.md).

```python
from load import load_net, pool_entry, supplement_sec

net = load_net("full")
entry, delayed = pool_entry(net, node_sec=supplement_sec(net))   # placeholder="date" for the sensitivity run
```

The first load of each version (with or without aliases) writes a parquet cache to `data/.cache/` (about
44 MB each, up to about 220 MB; git-ignored through `data/*`). Later loads take well under a second. The cache
rebuilds itself when `MANIFEST.txt` changes or a file is damaged; delete the folder to reset it, or set
`HFG_CACHE=off` to bypass it.

## Null trees

`load.py` also builds the textbook trees that RQ1 and RQ4 compare the real network with. Both return a `Net`, the array
network of `load_net`: one root at second 0, node `i` one second after node `i - 1`, ids `sim:0000000`, `sim:0000001`, ... and a
`depth` column (root = 0).

- `random_recursive_tree(n)`: every node attaches to a uniformly chosen earlier node. The mean node depth is
  H(n-1) - (n-1)/n, about ln n - 0.42 (the newest node sits at about ln n); a node has k children with probability 2^-(k+1).
- `price_tree(n, a=1.0)`: every node attaches with probability proportional to k + a, where k is the number of children it has so
  far (`a > 0`). With one parent per node (m = 1) the degree tail has exponent 2 + a/m: 3 for `a = 1`, 5 for `a = 3`. The exact
  degree distribution is in `tests/sim_helpers.py` (`price_pk`); a power-law fit at a small `xmin` gives a lower exponent than 2 + a/m
  even on exact trees, because the distribution is only asymptotically a power law.

```python
from load import random_recursive_tree, price_tree, simulate, arrivals_like, load_net

tree = random_recursive_tree(100_000)                       # one root, 99,999 edges, a Net
price = price_tree(100_000, a=1.0)                          # same shape; a = 1 is linear preferential attachment
depth = tree.nodes["depth"].mean()                          # about 11.1 on average for n = 100,000 (exact: 11.09)
members = [price_tree(100_000, seed=("rq1", i)) for i in range(20)]   # an ensemble of 20 different trees
G = tree.to_networkx()                                      # networkx DiGraph, for small trees and checks
```

Seeds. Two calls without `seed` return the same tree (a fixed stream per function and `n`). For an ensemble pass a different label
tuple per member, such as `seed=("rq1", i)` (labels are strings, integers or floats; an int is one label). The tree records its
labels and the numpy version in `net.meta`; the streams are reproducible for the pinned numpy only.

Matched null. The trees above have one root. A real family has many roots (for example 158 Qwen roots, with a mean leaf depth of
1.4-2.4 against ln n of about 12), so comparing the depth of a real family with `ln n` or with a single-root tree is misleading.
Replay the observed arrival schedule instead, with the same attachment rule:

```python
nulls = [simulate(arrivals_like(load_net("no_swarm")), rule="M0", seed=("rq1", i)) for i in range(20)]
```

`simulate` returns a `Net` with the same nodes, roots, arrival times and number of parents per node as the data, attached at
random; `rule="M1"` gives preferential attachment on the same schedule (see the docstring of `simulate` for the other rules).
Its batch step is `SIM_STEP = 0.000625`, confirmed by the RQ2 convergence stage (`results/rq2/sim_convergence.csv` on the
`rq2-growth` branch), calibrated on the full arrival profile only, which holds the largest monthly burst; Phase 8 re-checks it
under measured parameters and may add the `no_swarm` and `quant_collapsed` profiles.

Runtimes measured on one Windows machine: the two wrappers take about 1 s for 1,000,000 nodes (including building the `Net`); `arrivals_like` takes
about 0.2 s on a loaded `no_swarm` net and one `simulate` run on the full schedule at `SIM_STEP = 0.000625` about 1.8 s for
rule M0 and about 6 s for rule M1, M2 or M3 (a larger `step`, for example 0.01, is faster: M1 about 1.8 s); `.to_networkx()`
of a 1,000,000-node net takes about 3 s, so build the graph only for small trees and checks.

## Repository layout

| Folder | Content |
| --- | --- |
| `data_pipeline/` | Builds the network: snapshot, roots, renames, file lists, cleaning (`analyse.py`) |
| `checks/` | Quality checks: edge check, collapsed network, bulk uploaders, live counts |
| `load.py` | Shared loader for the frozen data |
| `data_supplement/` | Upload-date supplement for v1 nodes without a date (one-time Hub fetch, frozen with md5) and the fetch script |
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
