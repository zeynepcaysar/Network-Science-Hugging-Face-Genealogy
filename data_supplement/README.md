# Upload-date supplement (`data_supplement/`)

About 45,000 nodes of the frozen v1 network have no upload date (`created_at` is empty in `nodes.csv`): models with no
metadata in the snapshot, and some whose date was the Hub's placeholder. Edges whose parent has no date cannot be placed in
time by the candidate-pool rule, so the parent waits for its first child (delayed entry). This folder holds a one-time fetch of
those dates from the Hub, frozen with an md5 check. The v1 files are never changed and no new snapshot is taken.

| File | What it is |
| --- | --- |
| `fetch_outside.py` | The only code in the repository that talks to the Hub. One pass, run once; analysis code never imports it. |
| `build_date_supplement.py` | Offline: turns the raw fetch into the supplement, reports coverage, `--check` verifies it. |
| `date_supplement.csv` | The supplement: one row per v1 id without a date. Committed, md5-checked, stored without line-ending conversion (`.gitattributes`: `*.csv -text`). |
| `date_supplement.provenance.json` | Freeze date, md5 of the CSV and of the raw fetch, v1 input md5 values, rule texts, counts, v1-vs-fetched agreement and coverage. |

## Using it

```python
from load import load_net, pool_entry, supplement_sec

net = load_net("full")
entry, delayed = pool_entry(net, node_sec=supplement_sec(net))                         # default: placeholder dates stay missing
entry, delayed = pool_entry(net, node_sec=supplement_sec(net, placeholder="date"))     # sensitivity run: use the placeholder date
```

`supplement_sec(net, placeholder="missing", path=None)` returns node seconds equal to `net.node_sec`, with the supplement's
dates filled in where v1 has none. It never replaces a v1 date, ignores ids that are not in the loaded version (a `no_swarm`
net or `families=[...]` subset), matches ids such as `None` and `NA` as strings, and raises `ValueError` or
`FileNotFoundError` when the CSV or its sidecar is missing, changed (md5), incomplete or malformed. It never falls back silently
to v1 dates. A filled date that is later than the node's first child still gives delayed entry in `pool_entry`.

`placeholder="missing"` (default) keeps the Hub's placeholder date missing. The Hub stores `2022-03-02` as the lowest creation
date it has (about 3,000 of the undated ids carry it, for example `distilbert-base-uncased`), v1 blanks it the same way, and
treating it as an upload date would make old models look like March 2022 uploads. `placeholder="date"` uses it and is meant as
the labelled sensitivity run. Any other value raises `ValueError`. `path=` points at another supplement file.

## Columns of `date_supplement.csv`

| Column | Meaning |
| --- | --- |
| `id` | v1 node id (as in `nodes.csv`; read with `dtype=str, keep_default_na=False`) |
| `status` | `ok`, `renamed` (the Hub redirects the id to another repo), `not_found`, `no_access`, `invalid_id`, `error` or `not_fetched` |
| `resolved_id` | the repo the Hub answered with (differs from `id` for renamed ids) |
| `created_at` | the raw Hub value, ISO 8601, for `ok` and `renamed`; empty otherwise |
| `date_is_placeholder` | `True` when `created_at` starts with `2022-03-02` (the Hub's lowest stored date) |
| `shared_repo` | `True` when two or more v1 ids resolve to the same repo; all of them keep the date |
| `source` | `list` (listing pass) or `model_info` (per-model call) |
| `fetched_at` | when the row was fetched (UTC) |
| `v1_reason` | `no_metadata`, `backfilled` or `undated_other`: why v1 has no date |
| `downloads_30d`, `downloads_all_time`, `likes` | popularity at the fetch date, later than the 5 Nov snapshot |
| `note` | for undated family roots: status, probe date and the likely intended repo |

A fetched date fills only ids v1 has no date for; for the others the fetched date is compared with v1 and the agreement is
reported. The undated family roots (repos named by a family prefix that do not exist on the Hub) are documented in `note` and
stay undated, they are never given a date by hand. Their number and list come with the real build.

## Coverage

The share of edges whose parent still has no date, before and after the supplement, with the placeholder kept missing and with
the placeholder used, for `full`, `no_swarm` and `quant_collapsed` and for the edge sets family (the child is in a family),
non-family and all, is in `date_supplement.provenance.json` (key `coverage`) and in `results/data/date_supplement_coverage.csv`.
`results/data/date_supplement_disagreements.csv` lists every id whose fetched date differs from the v1 date.

## The raw fetch is out of band

The raw fetch (`raw_hub_fetch.parquet` and `raw_hub_fetch.provenance.json`) is large and lives in `data-outside/` (git-ignored),
shared through the team folder like `data-v1/`. Its md5 is recorded in `date_supplement.provenance.json`. You only need it to
rebuild the supplement or to verify the rebuild; without it the checks print a `note:` line and still pass.

```bash
python data_supplement/fetch_outside.py --check             # the raw fetch: md5, columns, id set, fetch window
python data_supplement/build_date_supplement.py --check     # the supplement: md5, v1 inputs, coverage, rebuild if the raw file is present
```

The fetch needs no token. An optional `HF_TOKEN` raises the rate limit; export it only in your own shell for the one-time fetch,
never type it on a command line and never commit it. The scripts read it from the environment and never write it down.
