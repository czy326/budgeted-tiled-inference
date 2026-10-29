# Budgeted tiled inference for small-object detection

Code for the manuscript *A Protocol for Budgeted Tiled Inference in Small-Object Detection, with Measured Boundaries*
(Zhengyang Chen, Dawei Liu).

The paper treats the question "where should a fixed inference budget be spent?" as a protocol with three measurable
quantities — the **support** of a target in feature cells, the **budget** in forward passes, and the **prevalence** of
targets across candidate tiles — and reports the range in which that protocol holds.

## What is here

```
code/       experiment scripts (probing, real-data evaluation, sweeps, timing)
figures/    scripts that render Figures 1-4
```

### Scripts and what they produce

| Script | Produces | Paper item |
|---|---|---|
| `code/probe_proposal.py` | AP50 / FPPI / recall-by-support for a set of tile-selection rules at one operating point | Tables 1, 3, 4 |
| `code/probe_bucket_map.py` | shared metrics (AP50, support bins, bucket maps) | used by all probes |
| `code/real_eval.py` | evaluation on sen2ships at native scale (preprocessing, tiling, AP) | Table 2, Section 3.4 |
| `code/lopo_sen2ships.py` | leave-one-port-out fine-tuning on real annotations | Section 3.4 |
| `code/selfpriors_real.py` | variance prior vs coarse-pass confidence vs random on real ports | Table 2 |
| `code/prior_coverage_real.py` | detector-free coverage of selected tiles, plus a water-masked variant | Section 3.4 |
| `code/prevalence_check.py` | coverage gain against prevalence at several tile sizes on real data | Figure 4 |
| `code/cov_sim.py` | the same coverage statistic on the synthetic regime | Figure 4 |
| `code/density_sweep.py` | per-frame binning by target count, to test the density explanation | Section 4.3 (rejected) |
| `code/spar.py` | tile-level prevalence for both domains | Table 5 |
| `code/latency_profile.py` | end-to-end timing split into tile extraction, forwards and merge | Table 4 / cost model |
| `code/render_check.py` | overlay of predictions and annotations on one real scene | Figure 3 |
| `code/run_yolo_arm.py`, `code/temporal_dataset.py`, `code/to_yolo_obb.py` | training a single arm | Section 3.2 |
| `figures/make_figs.py` | Figures 2 and 4 | Figures 2, 4 |
| `figures/fig_f1_real.py` | Figure 1 (real Sentinel-2 scene, three panels) | Figure 1 |
| `figures/fig_f1.py` | the earlier schematic version of Figure 1 | superseded |

## Data

The code reads three external sources. None of them is redistributed here.

- **DOTA** for the synthetic regimes: https://captain-whu.github.io/DOTA/
- **sen2ships** for the real ports (ten ports, Sentinel-2 L2A at 10 m, 251 usable annotations):
  https://github.com/ShipDetectionExperts/sen2ships
- **Finnish coastal set** for the annotation-density check: https://doi.org/10.5281/zenodo.10046342

## Running

```bash
pip install -r requirements.txt
```

Every script is a standalone entry point with `argparse` arguments, so the mixture of settings used in the paper is
visible at the call site rather than hidden in a config file. Two caveats:

1. **Paths are absolute.** The scripts were written against a single working tree and contain paths under
   `/mnt/e/10/p0_sim`. Adjust them, or pass the path arguments explicitly, before running.
2. **A GPU is required** for the probing, fine-tuning and timing scripts. The coverage and prevalence scripts are
   CPU-only and are the fastest way to check the pipeline.

## Method summary

- **Support**: `target_px * (imgsz / max(H, W)) / 8`, the number of stride-8 cells a target occupies in the network.
- **Budget**: one coarse pass plus `k` tile passes per scene, converted to milliseconds by `t ≈ 9 + 9.45 n`.
- **Prevalence**: the fraction of candidate tiles holding at least one target. It decides whether a selection rule
  has anything to select.
- Comparisons hold image, candidate pool and `k` fixed, and change only which tiles are evaluated.

## Result in one paragraph

Magnification dominates input resolution at equal budget (AP50 0.371 to 0.835, against 0.498 for a 1280-pixel input).
Cost has a quadratic merge term that overtakes the forward passes once a scene needs dozens of tiles. Ranking tiles by
a cheap prior beats random selection only when prevalence is low and the prior's bias matches the domain: it leads by
29.9 percentage points of coverage at 27.6% prevalence in simulation, and trails random by 2.4 to 6.5 points on real
ports where prevalence reaches 93.5%. At matched prevalence the two domains take opposite signs, so the boundary lies
in the prior rather than in target density.

## License

No license has been chosen yet. Until one is added, the code is under default copyright.
