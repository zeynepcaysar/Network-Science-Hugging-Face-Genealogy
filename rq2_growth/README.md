# RQ2 Growth

Do models gain children because they are already popular (preferential attachment) or because of something about the model itself (fitness)?

Attachment kernel from the dated edges (edge attribute date), fitness, hub turnover.

Load the data through the shared loader, and run every result on all three versions:

```python
from load import load
for version in ["full", "no_swarm", "quant_collapsed"]:
    G = load(version)
```

Save outputs to `results/rq2/`.

## Stages

| Script | What it does | Outputs |
|---|---|---|
| `freeze_official.py` | One-shot: freezes the official/root/mirror list for Qwen, Llama, Mistral and Gemma (already run); `--check` verifies the list, its md5 and the input md5s | `rq2_growth/external/official_list.csv` and `official_list.provenance.json` |
| `build_outside_measures.py` | One-shot freeze of the model-size and outside-measure tables from the out-of-band raw Hub fetch (size from the own safetensors total; downloads, likes and the official flags as frozen for the pilot, in one fetch window of at most 3 calendar days); `--report` regenerates the coverage tables from the frozen tables; `--check` verifies the md5s and re-derives the tables when the raw is present | `rq2_growth/external/model_size.parquet`, `outside_measures.parquet` with their `.provenance.json` sidecars; the `results/rq2/` files listed in its `OUTPUTS` constant (`outside_coverage_*`, `outside_size_regex_*`, `outside_children_check_full`, `numbers_outside`) |
| `pilot_qwen.py` | Qwen pilot: attachment kernel Π(k), first (a, β) with a naive CI, edge-use share | the `results/rq2/` files listed in its `OUTPUTS` constant: `pilot_qwen_pi_k.png`, `pilot_qwen_bins.csv`, `pilot_qwen_params.csv`, `numbers_pilot_qwen.csv`, `pilot_qwen.md` |
| `sim_convergence.py` | Real-profile convergence of the simulator's sub-batch step (D-06): 30 seeds per rule, step f against f / 2 on the `full` arrival profile, halved within the 10 s budget | the `results/rq2/` files listed in its `OUTPUTS` constant: `sim_convergence.csv`, `numbers_sim_convergence.csv`, `sim_convergence_timing.csv` |

The pilot is the exception to the "all three versions" sentence above: it covers Qwen on `full` and `no_swarm` only. `quant_collapsed` and the four matched base models come in Phase 5.

The convergence stage is the second exception: `sim_convergence.py` runs on the `full` arrival profile only, because it calibrates the simulator's sub-batch step (a method setting, not an RQ2 result) and `full` holds the largest monthly burst (2024-09); Phase 8 re-checks the step and may add the `no_swarm` and `quant_collapsed` profiles.

After activating the venv:

```bash
python rq2_growth/freeze_official.py --check
python rq2_growth/build_outside_measures.py --check
python rq2_growth/build_outside_measures.py --report
python rq2_growth/pilot_qwen.py
python rq2_growth/sim_convergence.py
```

`HFG_DATA` and `HFG_RESULTS` point a run at other data and output folders (unset: `data/` and `results/`). `rq2lib/` holds the RQ2 helpers (event table, exposure, classes, kernel).

Later phases read the size and outside-measure tables through `rq2lib.outside` (`read_size_table`, `read_outside_measures`; both check the md5 first) and the pre-registered primary popularity correlate (`downloads_all_time`) from `outside_measures.provenance.json`.
