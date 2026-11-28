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
stay undated, they are never given a date by hand. The 8 of them are listed below.

## What the frozen supplement contains

Frozen 2026-11-28 from the one-time Hub pass (anonymous, no token). `date_supplement.csv` md5 `3c9c2fd9959f2875c68f2efe85baf68c`;
the raw fetch `raw_hub_fetch.parquet` md5 `6cfaa4f34ba76fd311ca01abf68fa9d2` (out of band, see below).

- **45,399 rows**, one per id of `nodes.csv` without a v1 `created_at` (45,060 with no metadata in the snapshot, 339 backfilled).
- By status: `ok` 41,113, `renamed` 465, `not_found` 3,820, `invalid_id` 1. 465 ids are flagged `shared_repo` (130 `ok`, 335 `renamed`: two v1 ids
  resolve to one repo) and all keep the date.
- **38,278 real Hub dates**, **3,300 placeholder dates** (`2022-03-02`, flagged `date_is_placeholder`), **3,821 still missing**
  (3,820 `not_found` plus the one `invalid_id`; this includes the 8 undated family roots).
- Fetch window (D-16): first `fetched_at` 2026-11-27T22:15:28Z, last 2026-11-28T12:33:22Z, span 1 day (limit 3). The popularity
  columns (`downloads_30d`, `downloads_all_time`, `likes`) belong to the fetch date 2026-11-27, not to the 5 Nov snapshot.
- Agreement with v1 on the 919,187 ids that have both dates and a non-placeholder fetched date: 919,108 equal at second
  resolution, share **0.99991** (1 differs by under a day, 50 by 1 to 30 days, 28 by more than 30 days, at most 673 days; in all 79
  the fetched date is later than v1's, 77 of them in November 2026, after the 5 Nov snapshot). Every difference is in
  `results/data/date_supplement_disagreements.csv`. A fetched date never replaces a v1 date.

### The 8 undated family roots (D-13)

These repo ids appear as family roots in the v1 network but the Hub returns "repository not found" (they are typos or non-existent
ids, or deleted or private; truly gated repos such as `meta-llama/Llama-3.1-8B` answer anonymously). They stay undated, so they
stay on delayed entry in `pool_entry`, and they remain official roots in the frozen list. The "likely intended" repo is a note
only: it is named only when the fetch returned it, and it never supplies a date. They matter for root counts (8 of the 272
family roots are phantom ids).

| id | child edges in `full` | note |
| --- | --- | --- |
| `Qwen/Qwen-1_8B-Chat` | 12 | not_found, probed 2026-11-27; likely intended `Qwen/Qwen1.5-1.8B-Chat` |
| `Qwen/Qwen3-8B-Instruct` | 4 | not_found, probed 2026-11-27; likely intended `Qwen/Qwen3-8B` |
| `Qwen/Qwen3-32B-Instruct` | 4 | not_found, probed 2026-11-27; likely intended `Qwen/Qwen3-32B` |
| `Qwen/Qwen3___5-4B` | 2 | not_found, probed 2026-11-27; likely intended `Qwen/Qwen3.5-4B` |
| `meta-llama/Llama-3-8B` | 1 | not_found, probed 2026-11-27; likely intended `meta-llama/Meta-Llama-3-8B` |
| `google/gemma-3-4b` | 1 | not_found, probed 2026-11-27; likely intended `google/gemma-3-4b-pt` |
| `Qwen/Qwen3.6-27B-MTP` | 1 | not_found, probed 2026-11-27; likely intended `Qwen/Qwen3.6-27B` |
| `Qwen/Qwen2.5-VL-9B` | 1 | not_found, probed 2026-11-28; none identified in the fetch |

Together they carry 26 child edges (0.006% of the family edges).

## Coverage

The share of edges whose parent still has no date, before and after the supplement, with the placeholder kept missing and with
the placeholder used, for `full`, `no_swarm` and `quant_collapsed` and for the edge sets family (the child is in a family),
non-family and all, is in `date_supplement.provenance.json` (key `coverage`) and in `results/data/date_supplement_coverage.csv`
(the numbers below are the `full` rows; `no_swarm` and `quant_collapsed` are in the CSV).

| `full`, edges with an undated parent | edges | before | after, placeholder missing (default) | after, placeholder used |
| --- | --- | --- | --- | --- |
| family (the child is in a family) | 444,353 | 1.96% (8,711) | 0.14% (619) | 0.14% (618) |
| non-family | 510,258 | 81.64% (416,597) | 22.60% (115,294) | 2.10% (10,725) |
| all | 954,611 | 44.55% (425,308) | 12.14% (115,913) | 1.19% (11,343) |

PROJECT.md's 81.6% is the **non-family** share before the supplement, not the share over all of `full` (44.6%); the three are
reported separately. The family share is the 2.0% of the planning notes. `results/data/date_supplement_disagreements.csv` lists
every id whose fetched date differs from the v1 date.

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
