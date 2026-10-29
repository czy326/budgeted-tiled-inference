#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the local-variance prior even point at vessels on real imagery?

The simulation says prior-ranked tiles beat random tiles by ~0.2 AP; on fine-tuned real
ports the gap fell to ~0.02. The density explanation was tested and rejected (the gap is
largest in the densest bin). The remaining candidate is the prior itself: on real ports it
may be attracted to docks, terminals and quays.

This measures that directly and without any model: for k selected tiles, what fraction of
ground-truth boxes fall inside them? Prior tiles vs random tiles vs all tiles. If the prior
is no better than random at covering vessels, the collapsed AP gap is fully explained.

Run inside emssm:
  python prior_coverage_real.py --root /mnt/e/sen2ships-main/sen2ships --side 640 --k 4
"""

import argparse
import json
import os
import sys
import zlib

import cv2
import numpy as np

sys.path.insert(0, "/mnt/e/10/p0_sim/train")
from real_eval import load_rgb, slice_boxes  # noqa: E402
from lopo_sen2ships import find_ports, scene_of  # noqa: E402


def var_score(rgb, side):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (9, 9))
    mu2 = cv2.blur(g * g, (9, 9))
    return cv2.blur(np.maximum(mu2 - mu * mu, 0.0), (side, side),
                    borderType=cv2.BORDER_CONSTANT)


def water_from_bands(scene_path, H, W):
    """Water mask from B03/B08 (NDWI > 0), computed from the bands themselves.

    The shipped NDWI product is uint16 with a mean around 35000, so a naive `> 0` test
    marks every pixel as water and the mask becomes a no-op (which is exactly what the
    first attempt did -- it reproduced the unmasked numbers to the last digit).
    """
    import glob
    import rasterio
    d = os.path.dirname(scene_path)
    b3 = [x for x in glob.glob(os.path.join(d, "*B03*.tif*")) if os.path.isfile(x)]
    b8 = [x for x in glob.glob(os.path.join(d, "*B08*.tif*")) if os.path.isfile(x)]
    if not (b3 and b8):
        return None
    with rasterio.open(b3[0]) as s:
        a = s.read(1).astype(np.float32)
    with rasterio.open(b8[0]) as s:
        b = s.read(1).astype(np.float32)
    if a.shape != (H, W):
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
    if b.shape != (H, W):
        b = cv2.resize(b, (W, H), interpolation=cv2.INTER_NEAREST)
    den = np.maximum(a + b, 1.0)
    return (a - b) / den > 0.0


def covered(boxes, centres, side):
    """Fraction of boxes whose centre lies in at least one selected tile."""
    hit = 0
    for (cx, cy, w, h) in boxes:
        for (tx, ty) in centres:
            if abs(cx - tx) <= side / 2.0 and abs(cy - ty) <= side / 2.0:
                hit += 1
                break
    return hit


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/e/sen2ships-main/sen2ships")
    ap.add_argument("--side", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--water-mask", action="store_true",
                    help="score tiles by variance computed only over water (NDWI>0). "
                         "Tests whether the prior fails because docks dominate, which is "
                         "the suspected cause of it losing to random on real ports.")
    ap.add_argument("--out", default="/mnt/e/10/p0_sim/data/support/prior_coverage_real.json")
    args = ap.parse_args(argv)

    rows = []
    tot = {"n": 0, "rank": 0, "rand": 0, "all": 0}
    for port, cf in find_ports():
        p, boxes = scene_of(cf)
        if not os.path.isfile(p):
            print("  %-12s SKIP (no image)" % port)
            continue
        rgb = load_rgb(p, "pct")
        H, W = rgb.shape[:2]
        bs = slice_boxes(H, W, args.side, args.overlap)
        cent = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bs]
        if not cent:
            continue
        smap = var_score(rgb, args.side)
        if args.water_mask:
            wet = water_from_bands(p, H, W)
            if wet is not None:
                g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                mu = cv2.blur(g, (9, 9)); mu2 = cv2.blur(g * g, (9, 9))
                v = np.maximum(mu2 - mu * mu, 0.0)
                v = np.where(wet, v, 0.0)
                frac = cv2.blur(wet.astype(np.float32), (args.side, args.side),
                                borderType=cv2.BORDER_CONSTANT)
                smap = cv2.blur(v, (args.side, args.side),
                                borderType=cv2.BORDER_CONSTANT) * frac
        sh, sw = smap.shape
        val = [float(smap[int(min(max(c[1], 0), sh - 1)), int(min(max(c[0], 0), sw - 1))])
               for c in cent]
        order = sorted(range(len(cent)), key=lambda i: -val[i])
        k = min(args.k, len(cent))
        rank_c = [cent[i] for i in order[:k]]
        rng = np.random.default_rng(zlib.crc32(port.encode()))
        # average over several draws so random is not a single lucky/unlucky sample
        draws = 50
        rand_hits = 0
        for _ in range(draws):
            idx = rng.permutation(len(cent))[:k]
            rand_hits += covered(boxes, [cent[i] for i in idx], args.side)
        rand_hits /= float(draws)
        r = covered(boxes, rank_c, args.side)
        a = covered(boxes, cent, args.side)
        rows.append({"port": port, "n_gt": len(boxes), "tiles": len(cent), "k": k,
                     "cov_rank": r, "cov_rand": rand_hits, "cov_all": a})
        tot["n"] += len(boxes); tot["rank"] += r
        tot["rand"] += rand_hits; tot["all"] += a
        print("  %-12s gt=%3d tiles=%2d k=%d  prior=%4.1f%% rand=%4.1f%% all=%.0f%%" % (
            port, len(boxes), len(cent), k, 100 * r / max(len(boxes), 1),
            100 * rand_hits / max(len(boxes), 1), 100 * a / max(len(boxes), 1)))

    if tot["n"]:
        print("\n合计 gt=%d  prior=%.1f%%  random=%.1f%%  all=%.0f%%  (prior-random=%+.1f pt)" % (
            tot["n"], 100 * tot["rank"] / tot["n"], 100 * tot["rand"] / tot["n"],
            100 * tot["all"] / tot["n"],
            100 * (tot["rank"] - tot["rand"]) / tot["n"]))
    json.dump({"side": args.side, "k": args.k, "ports": rows,
               "total": tot}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
