# Figure 13: reordering measurements

The two registered measurement stages write their complete outputs under the
stage directories. This thin finalizer validates and publishes the two exact
CSV inputs consumed by the Figure 13 renderer under the experiment's
`data/latest/stages/` directory.

```bash
python3 -m artifact run figure-13-reordering-methods
python3 -m artifact rebuild --only figure-13-reordering-methods --data-slot latest
```
