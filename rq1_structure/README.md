# RQ1 Structure

How are children counts distributed, and do the families branch out wide or grow in long chains?

Degree fits (power law vs lognormal vs truncated power law, discrete), tree shape (depth, branching, normalised Sackin) vs null trees.

Load the data through the shared loader, and run every result on all three versions:

```python
from load import load
for version in ["full", "no_swarm", "quant_collapsed"]:
    G = load(version)
```

Save outputs to `results/rq1/`.
