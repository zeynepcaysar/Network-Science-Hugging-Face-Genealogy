# RQ4 Robustness

When models are removed at random, or the biggest hubs first, how much of each family loses its link to the original base model?

Random vs targeted removal; compare with and without merge edges.

Load the data through the shared loader, and run every result on all three versions:

```python
from load import load
for version in ["full", "no_swarm", "quant_collapsed"]:
    G = load(version)
```

Save outputs to `results/rq4/`.
