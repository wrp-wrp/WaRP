# Paper artifact -> Paper Experiment map

Generated from `experiments/PAPER_ARTIFACT_MANIFEST.json` by
`python3 -m artifact catalog write-doc`. Do not hand-edit.

## Ownership rule

Every experiment-backed paper figure or table has exactly one Paper
Experiment owner. Artifacts derived from the same frozen evidence stay
together. Owner directories use `figure-NN-short-name` or
`table-NN-short-name`. Conceptual diagrams and documentation tables have
no experiment directory.

## Experimental figures

| Figure | Paper label | Paper Experiment | Repository artifact(s) |
| --- | --- | --- | --- |
| 1 | `fig:intro-latency-composition` | `experiments/figure-01-s3-range-get-latency/` | `experiments/figure-01-s3-range-get-latency/latency_composition.pdf` |
| 4 | `fig:bg-microbenchmarks` | `experiments/figure-04-object-store-io/` | `experiments/figure-04-object-store-io/surrogate_runtime_semantics.pdf`<br>`experiments/figure-04-object-store-io/request_fanout.pdf` |
| 5 | `fig:planning-frontier` | `experiments/figure-05-planning-frontier/` | `experiments/figure-05-planning-frontier/figure_planning_bw_frontier.pdf` |
| 7 | `fig:eval-real-lance-metrics` | `experiments/figure-07-real-lance-metrics/` | `experiments/figure-07-real-lance-metrics/figure_real_lance_metrics.pdf` |
| 8 | `fig:eval-real-lance-concurrency-scaling` | `experiments/figure-08-sift100m-concurrency-scaling/` | `experiments/figure-08-sift100m-concurrency-scaling/figure_real_lance_concurrency_scaling.pdf` |
| 9 | `fig:eval-qps-cost` | `experiments/figure-09-qps-cost/` | `experiments/figure-09-qps-cost/figure_qps_vs_savings_q4_io64_all6_grid_qps_p95_3x4.pdf` |
| 10 | `fig:eval-p-sweep-closedloop` | `experiments/figure-10-p-sweep-closedloop/` | `experiments/figure-10-p-sweep-closedloop/figure_closed_loop_env3_reorder_only_combined.pdf` |
| 11 | `fig:eval-greedy-locality-shift` | `experiments/figure-11-greedy-locality-shift/` | `experiments/figure-11-greedy-locality-shift/figure_greedy_policy_transfer.pdf` |
| 12 | `fig:eval-planner-overhead-breakdown` | `experiments/figure-12-planner-overhead/` | `experiments/figure-12-planner-overhead/figure_03_real_data_planner_overhead_piecewise.pdf` |
| 13 | `fig:reorder-panels` | `experiments/figure-13-reordering-methods/` | `experiments/figure-13-reordering-methods/figure_reorder_panels.pdf` |
| 14 | `fig:eval-sensitivity-sift1m` | `experiments/figure-14-sift1m-sensitivity/` | `experiments/figure-14-sift1m-sensitivity/figure_sift1m_real_s3_sensitivity_combined.pdf` |

## Conceptual diagrams

| Figure | Paper label | Paper source |
| --- | --- | --- |
| 2 | `fig:intro-plan-example` | `figures/plan-example.pdf` |
| 3 | `fig:warp-pipeline` | `figures/pipeline.pdf` |
| 6 | `fig:reorder-schematic` | `figures/reorder.pdf` |

## Experimental tables

| Table | Paper label | Paper Experiment | Repository artifact(s) |
| --- | --- | --- | --- |
| 1 | `tab:intro-coarse-cluster` | `experiments/table-01-cluster-coarsening/` | `experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.tex` |
| 4 | `tab:eval-payload-accounting` | `experiments/figure-07-real-lance-metrics/` | `experiments/figure-07-real-lance-metrics/table_payload_accounting.tex` |
| 5 | `tab:eval-planner-overhead` | `experiments/figure-12-planner-overhead/` | `experiments/figure-12-planner-overhead/table_03_real_data_planner_overhead_piecewise.tex` |

## Documentation tables

These tables describe datasets or methods; they do not report a measured result.

| Table | Paper label | Paper source |
| --- | --- | --- |
| 2 | `tab:reorder-methods` | `sections/reordering.tex` |
| 3 | `tab:eval-datasets` | `sections/evaluation.tex` |

## Verification

```bash
python3 -m artifact catalog check
python3 -m artifact catalog check-doc
```
