# Data

The data files are not stored on GitHub because of their size. The frozen dataset (v1, built
from the snapshot of 5 Nov 2026) is in the team's shared folder `data-v1`.

## Frozen dataset

| File | Size | Content |
| --- | --- | --- |
| `edges.csv` | 83 MB | Cleaned network: parent, child, rel (corrected edge type), rel_declared (type as tagged by the uploader) |
| `nodes.csv` | 119 MB | One row per model: families, generation, creation date, flags, degrees, author, downloads, likes |
| `edges_quant_collapsed.csv` | 64 MB | Network with quantised copies merged into their source model |
| `edges_aliases.csv` | 13 KB | Undeclared re-uploads linked to the official model (sensitivity check) |
| `MANIFEST.txt` | 1 KB | Build date, row counts and md5 checksums of all files |

These files are read by `load.py`.

## Copying and checking the frozen files

Copy `MANIFEST.txt` and the four CSVs (required) and `models.jsonl`, `files.jsonl`, `roots.jsonl`
(optional, only needed to rebuild) from the shared `data-v1` folder into `data/`. Never overwrite a
file that is already there without checking it first. Then run, with the venv activated:

```
python load.py --verify
```

It prints one `PASS`, `FAIL` or `note` line per entry of `MANIFEST.txt` (md5 of every file) and exits
with 1 if any line is `FAIL`; a `note` means an optional file is absent. `verify_data("data-v1")`
(from `load.py`) checks the shared folder itself. The check only reads files.

If it reports `is a directory` for `data/models.jsonl`, the file was copied inside a folder of the
same name: move the inner file up (`mv data/models.jsonl/models.jsonl data/models.jsonl.tmp`,
`rmdir data/models.jsonl`, `mv data/models.jsonl.tmp data/models.jsonl`).

`md5 data/*.csv` (macOS) or `md5sum data/*.csv` (Linux) against `MANIFEST.txt` remains a manual
fallback.

## Inputs (for rebuilding)

| File | Size | Made by | Content |
| --- | --- | --- | --- |
| `models.jsonl.gz` | 49 MB | `data_pipeline/snapshot.py` | Snapshot of 5 Nov 2026: every model that declares a parent, plus all models of meta-llama, mistralai, Qwen and google |
| `roots.jsonl` | 1 MB | `data_pipeline/fetch_roots.py` | Models of the four organisations |
| `files.jsonl` | 137 MB | `data_pipeline/fetch_files.py` | File names per model, used to correct edge types |

Rebuild (~3 min, ~4.5 GB memory):

```
gunzip -k data/models.jsonl.gz
python3 data_pipeline/analyse.py
```

A new snapshot gives different numbers, since models are added and deleted on the Hub every day.
