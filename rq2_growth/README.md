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
