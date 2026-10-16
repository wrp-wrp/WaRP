# Figure 4 four-topology runner

Figure 4's runtime panel is one coordinated experiment across four network
topologies. Configure one local or SSH execution target for each topology and
run the registered experiment. Each target must have this repository and AWS
credentials for the named bucket.

```bash
python3 -m artifact run figure-04-object-store-io -- \
  --topology ap_same_region,ap-host,ap-bucket,ap-northeast-1,30 \
  --topology us_same_region,us-host,us-bucket,us-east-1,300 \
  --topology cross_region,us-host,ap-bucket,ap-northeast-1,300 \
  --topology public_internet,local,ap-bucket,ap-northeast-1,300
```

Fresh `serial_stats.json` files land under the owning experiment's
`data/latest/runtime/`. The request-size ladder and sample counts are validated
before the run is accepted.
