# RQ3 Edge types

How do fine-tune, adapter, quantisation and merge links differ, and do merges connect families?

Use edge attribute rel (corrected type); rel_declared is the tag as uploaded.

Load the data through the shared loader, and run every result on all three versions:

```python
from load import load
for version in ["full", "no_swarm", "quant_collapsed"]:
    G = load(version)
```

Save outputs to `results/rq3/`.
