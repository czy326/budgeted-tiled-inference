#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F1 rebuilt: three equal-size panels, labels only, all numbers in the caption.

Differences from the schematic that preceded it:
  * panel (b) is cropped to the scene's aspect ratio so the three panels occupy the same
    box -- a square tile beside two landscape panels forced unequal widths
  * panels carry only "(a) (b) (c)"; object size, support and prevalence go into the
    caption, where they can be typeset and translated properly

Everything shown is a genuine crop of one real Sentinel-2 scene. Panel (b) is upscaled
with nearest neighbour on purpose: the pixelation is the mechanism under test, so it is
not smoothed away.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/mnt/e/10/p0_sim/train")
from real_eval import load_rgb, slice_boxes  # noqa: E402
from lopo_sen2ships import find_ports, scene_of  # noqa: E402

SIDE = 130
OVERLAP = 0.2
OUT = "/mnt/e/10/p0_sim/data/figs"
MAX_MEDIAN_PX = 14.0       # stay in the dataset's size regime (~12 px median)


def best_water_tile():
    """Crop centred on the largest vessel that is surrounded by water.

    Picking a grid tile always put the subject at the border, because in a port scene
    the vessels line the quay. Centring on a vessel in open water keeps panel (b)
    readable while remaining an ordinary crop of the scene.
    """
    best = None
    for name, cf in find_ports():
        img_path, boxes = scene_of(cf)
        if not os.path.isfile(img_path) or len(boxes) < 5:
            continue
        if float(np.median([b[2] for b in boxes])) > MAX_MEDIAN_PX:
            continue
        rgb = load_rgb(img_path, "pct")
        H, W = rgb.shape[:2]
        thr = np.percentile(rgb.mean(2), 40)
        for (cx, cy, w, h) in boxes:
            x0 = int(max(0, min(W - SIDE, round(cx - SIDE / 2.0))))
            y0 = int(max(0, min(H - SIDE, round(cy - SIDE / 2.0))))
            patch = rgb[y0:y0 + SIDE, x0:x0 + SIDE]
            if patch.shape[0] != SIDE or patch.shape[1] != SIDE:
                continue
            dark = float((patch.mean(2) < thr).mean())
            if dark < 0.85:
                continue
            gt = [b for b in boxes
                  if x0 <= b[0] < x0 + SIDE and y0 <= b[1] < y0 + SIDE]
            score = (round(w), dark)
            if best is None or score > best[0]:
                best = (score, name, rgb, (x0, y0), gt, boxes)
    return best


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    hit = best_water_tile()
    if hit is None:
        sys.exit("no suitable water tile found")
    _, name, rgb, (x0, y0), best_gt, boxes = hit
    H, W = rgb.shape[:2]
    aspect = W / float(H)
    med_w = float(np.median([b[2] for b in boxes]))
    m = 640.0 / float(SIDE)

    # ---------------- (a) whole scene squeezed into the network input
    s = 640.0 / max(H, W)
    small = cv2.resize(rgb, (int(round(W * s)), int(round(H * s))),
                       interpolation=cv2.INTER_AREA)

    # ---------------- (b) operated tile, cropped to the scene aspect
    crop = rgb[y0:y0 + SIDE, x0:x0 + SIDE]
    up = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_NEAREST)
    want_h = int(round(640 / aspect))
    if want_h >= 640:
        up_b, base = up, 0
    else:
        anchor = max(best_gt, key=lambda b: b[2]) if best_gt else None
        vcy = (anchor[1] - y0) * m if anchor is not None else 320.0
        base = int(round(max(0, min(640 - want_h, vcy - want_h / 2.0))))
        up_b = up[base:base + want_h, :]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 7.2 / aspect / 3.0 + 0.45))

    ax = axes[0]
    ax.imshow(small)
    for (cx, cy, w, h) in boxes:
        ax.add_patch(Rectangle((cx * s - w * s / 2, cy * s - h * s / 2),
                               max(w * s, 1.8), max(h * s, 1.8),
                               fill=False, ec="#ff4d4d", lw=0.5))
    ax.set_title("(a)", loc="center", fontsize=11, pad=3)

    ax = axes[1]
    ax.imshow(up_b)
    for i in range(1, up_b.shape[1] // 8):
        ax.axvline(i * 8, color="white", lw=0.16, alpha=0.12)
    for j in range(1, up_b.shape[0] // 8):
        ax.axhline(j * 8, color="white", lw=0.16, alpha=0.12)
    for (cx, cy, w, h) in best_gt:
        ax.add_patch(Rectangle(((cx - x0) * m - w * m / 2,
                                (cy - y0) * m - base - h * m / 2),
                               w * m, h * m, fill=False, ec="#ff4d4d", lw=0.9))
    ax.set_title("(b)", loc="center", fontsize=11, pad=3)

    ax = axes[2]
    ax.imshow(small)
    tiles = slice_boxes(H, W, SIDE, OVERLAP)
    # Overlapping tiles stack alpha if each rectangle is drawn semi-transparent, which
    # turned the occupied area into a solid blob. Paint one mask instead, so the tint is
    # uniform regardless of overlap.
    # the mask must live in the same pixel grid as `small`, otherwise the two imshow
    # calls set different extents and the overlay lands off the image
    mask = np.zeros(small.shape[:2], np.float32)
    occ = 0
    for (tx0, ty0, tx1, ty1) in tiles:
        h_ = any(tx0 <= b[0] < tx1 and ty0 <= b[1] < ty1 for b in boxes)
        occ += h_
        if h_:
            ry0, ry1 = int(round(ty0 * s)), int(round(ty1 * s))
            rx0, rx1 = int(round(tx0 * s)), int(round(tx1 * s))
            mask[ry0:ry1, rx0:rx1] = 1.0
        ax.add_patch(Rectangle((tx0 * s, ty0 * s), (tx1 - tx0) * s, (ty1 - ty0) * s,
                               fill=False,
                               edgecolor=(0.85, 0.33, 0.18) if h_ else "#666666",
                               lw=0.8 if h_ else 0.3,
                               alpha=0.9 if h_ else 0.45))
    tint = np.zeros(small.shape[:2] + (4,), np.float32)
    tint[..., 0], tint[..., 1], tint[..., 2] = 1.00, 0.62, 0.35
    tint[..., 3] = mask * 0.16
    ax.imshow(tint)
    ax.set_title("(c)", loc="center", fontsize=11, pad=3)
    prev = 100.0 * occ / max(len(tiles), 1)

    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        a.set_box_aspect(1.0 / aspect)
        for sp in a.spines.values():
            sp.set_linewidth(0.6)

    fig.tight_layout(pad=0.35)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "F1_concept_real." + ext), dpi=600)
    plt.close(fig)

    meta = {"port": name, "size": [W, H], "n_gt": len(boxes), "tile": SIDE,
            "n_tiles": len(tiles), "occupied": occ, "prevalence": round(prev, 1),
            "median_vessel_px": med_w, "in_net_px": round(med_w * m, 1),
            "support_cells": round(med_w * m / 8.0, 1),
            "panel_aspect": round(up_b.shape[1] / float(up_b.shape[0]), 3),
            "scene_aspect": round(aspect, 3)}
    json.dump(meta, open(os.path.join(OUT, "F1_concept_real.json"), "w"), indent=2)
    print("[f1] %s %dx%d gt=%d | tiles=%d occ=%d prev=%.1f%% | %.0f px -> %.0f px (%.1f cells) | aspect %.2f vs %.2f"
          % (name, W, H, len(boxes), len(tiles), occ, prev, med_w, med_w * m,
             med_w * m / 8.0, meta["panel_aspect"], aspect))


if __name__ == "__main__":
    main()
