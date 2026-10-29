#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the figures the paper skeleton lists as missing (F2, F3, F4).

All numbers are read back from the stored JSONs so the figures cannot drift from the
results. Simulation coverage (needed for F4) is recomputed here rather than copied by
hand, and written to a JSON of its own.

F4 is the core figure: coverage gain of prior-ranked tiles over random tiles, plotted
against tile prevalence, with simulation and real data on the same axes. It is the plot
that shows the effect flipping sign at matched prevalence.

Run inside emssm:
  python make_figs.py --out /mnt/e/10/p0_sim/data/figs
"""

import argparse
import glob
import json
import os
import sys
import zlib

import cv2
import numpy as np

sys.path.insert(0, "/mnt/e/10/p0_sim/train")

SUP = "/mnt/e/10/p0_sim/data/support"
LAYOUT = "/mnt/e/10/p0_sim/data/p0_dota_sim_v4/train/layout/masked_all/test"


def slice_boxes(H, W, side, overlap):
    try:
        from sahi.slicing import get_slice_bboxes
        return [list(b) for b in get_slice_bboxes(
            H, W, slice_height=side, slice_width=side,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            auto_slice_resolution=False)]
    except Exception:                                    # noqa: BLE001
        st = max(1, int(round(side * (1 - overlap))))
        def ax(n):
            if n <= side:
                return [0]
            xs = list(range(0, n - side + 1, st))
            if xs[-1] != n - side:
                xs.append(n - side)
            return xs
        return [[x, y, x + side, y + side] for y in ax(H) for x in ax(W)]


def sim_coverage(sides=(130, 196), k=4):
    """Recompute prior-vs-random GT coverage on the simulated test set."""
    out = []
    for side in sides:
        tn = tp = gt = cp = cr = 0.0
        for ip in sorted(glob.glob(os.path.join(LAYOUT, "images", "*.png"))):
            bgr = cv2.imread(ip)
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            stem = os.path.splitext(os.path.basename(ip))[0]
            boxes = []
            lp = os.path.join(LAYOUT, "labelTxt", stem + ".txt")
            if os.path.isfile(lp):
                for line in open(lp, errors="ignore"):
                    t = line.split()
                    if len(t) < 9:
                        continue
                    pts = np.array(t[:8], float).reshape(4, 2)
                    c = pts.mean(0)
                    boxes.append((c[0], c[1]))
            bs = slice_boxes(H, W, side, 0.2)
            cent = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bs]
            if not cent or not boxes:
                continue
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mu = cv2.blur(g, (9, 9))
            mu2 = cv2.blur(g * g, (9, 9))
            sm = cv2.blur(np.maximum(mu2 - mu * mu, 0.0), (side, side),
                          borderType=cv2.BORDER_CONSTANT)
            sh, sw = sm.shape
            val = [float(sm[int(min(max(c[1], 0), sh - 1)), int(min(max(c[0], 0), sw - 1))])
                   for c in cent]
            order = sorted(range(len(cent)), key=lambda i: -val[i])
            kk = min(k, len(cent))

            def cov(sel):
                return sum(1 for (cx, cy) in boxes
                           if any(abs(cx - tx) <= side / 2 and abs(cy - ty) <= side / 2
                                  for (tx, ty) in sel))

            pos = sum(1 for (tx, ty) in cent
                      if any(abs(cx - tx) <= side / 2 and abs(cy - ty) <= side / 2
                             for (cx, cy) in boxes))
            rng = np.random.default_rng(zlib.crc32(stem.encode()))
            ra = 0.0
            for _ in range(30):
                idx = rng.permutation(len(cent))[:kk]
                ra += cov([cent[i] for i in idx])
            ra /= 30.0
            tn += len(cent); tp += pos; gt += len(boxes)
            cp += cov([cent[i] for i in order[:kk]]); cr += ra
        out.append({"domain": "sim", "side": side, "prev": 100 * tp / tn,
                    "cov_prior": 100 * cp / gt, "cov_rand": 100 * cr / gt,
                    "gain": 100 * (cp - cr) / gt, "n_gt": int(gt)})
    return out


def load(tag):
    p = os.path.join(SUP, tag if tag.endswith(".json") else tag + ".json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)["conditions"]


def fig_f2(plt, out):
    """AP50 versus viewing scale: no zoom / bigger input / budgeted zoom."""
    c = load("v4_curve_s2026")
    if not c:
        return
    labels = ["single pass\n(full scene)", "1024 px input", "1280 px input",
              "zoom k=4", "zoom k=8\n(prior-ranked)"]
    vals = [c["base"]["ap50"], c["full1024"]["ap50"], c["full1280"]["ap50"],
            c["sahiRank4"]["ap50"], c["sahiRank8"]["ap50"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.bar(range(len(vals)), vals, color=["#9aa4b2", "#7c93b8", "#4f6f9e",
                                                "#2f5d8a", "#1f3f66"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, "%.3f" % v,
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AP50", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Viewing scale dominates: zoom beats a larger input", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "F2_scale." + ext), dpi=600)
    plt.close(fig)


def fig_f3(plt, out):
    """AP50 versus tile budget for four selection rules."""
    c = load("v4_curve_s2026")
    if not c:
        return
    ks = ["2", "4", "8"]
    xs = [c["sahiRank%s" % k]["windows_per_image"] for k in ks]
    series = [("prior-ranked", "sahiRank", "#1f3f66", "o"),
              ("random", "sahiRand", "#c05a3a", "s"),
              ("geometry (centre-out)", "sahiMatch", "#8a8f98", "^"),
              ("uniform grid", "unif", "#4f8a5f", "d")]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for name, key, col, mk in series:
        ys = [c["%s%s" % (key, k)]["ap50"] for k in ks]
        ax.plot(xs, ys, marker=mk, color=col, lw=1.6, ms=5, label=name)
    if "sahi130_o20" in c:
        ax.axhline(c["sahi130_o20"]["ap50"], color="#333", ls="--", lw=1.2)
        ax.text(xs[0] * 1.05, c["sahi130_o20"]["ap50"] + 0.012,
                "unbudgeted slicing (%.1f tiles)" % c["sahi130_o20"]["windows_per_image"],
                fontsize=7.5, color="#333")
    ax.set_xlabel("tiles evaluated per image", fontsize=9)
    ax.set_ylabel("AP50", fontsize=9)
    ax.set_title("Selection rule matters at every budget", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "F3_budget." + ext), dpi=600)
    plt.close(fig)


def fig_f4(plt, out, sim_rows, real_path):
    """The core figure: coverage gain vs tile prevalence, two domains on one axis."""
    real = []
    if os.path.isfile(real_path):
        real = json.load(open(real_path, encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    if sim_rows:
        ax.plot([r["prev"] for r in sim_rows], [r["gain"] for r in sim_rows],
                marker="o", color="#1f3f66", lw=1.6, ms=6, label="simulation (DOTA-derived)")
        for r in sim_rows:
            ax.annotate("side %d" % r["side"], (r["prev"], r["gain"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                        color="#1f3f66")
    if real:
        ax.plot([r["prev"] for r in real], [r["gain"] for r in real],
                marker="s", color="#c05a3a", lw=1.6, ms=6, label="real Sentinel-2 ports")
        for r in real:
            ax.annotate("%d px" % r["side"], (r["prev"], r["gain"]),
                        textcoords="offset points", xytext=(5, 7), fontsize=7,
                        color="#c05a3a")
    ax.axhline(0.0, color="#333", lw=1.0)
    ax.set_xlabel("tile prevalence  (% of tiles containing a target)", fontsize=9)
    ax.set_ylabel("coverage gain of prior over random  (pt)", fontsize=9)
    ax.set_title("Same prevalence, opposite sign: the hand prior does not transfer",
                 fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "F4_prevalence_gain." + ext), dpi=600)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/e/10/p0_sim/data/figs")
    ap.add_argument("--skip-sim", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                         "axes.titlesize": 10, "figure.dpi": 120})

    sim_rows = []
    p_sim = os.path.join(SUP, "cov_sim.json")
    if args.skip_sim and os.path.isfile(p_sim):
        sim_rows = json.load(open(p_sim, encoding="utf-8"))
    else:
        print("[figs] recomputing simulation coverage ...")
        sim_rows = sim_coverage()
        json.dump(sim_rows, open(p_sim, "w", encoding="utf-8"), indent=2)
    for r in sim_rows:
        print("   sim side=%d prev=%.1f%% prior=%.1f%% rand=%.1f%% gain=%+.1f" % (
            r["side"], r["prev"], r["cov_prior"], r["cov_rand"], r["gain"]))

    fig_f2(plt, args.out)
    fig_f3(plt, args.out)
    fig_f4(plt, args.out, sim_rows, os.path.join(SUP, "prevalence_real.json"))
    print("[figs] ->", args.out)
    for f in sorted(glob.glob(os.path.join(args.out, "*.png"))):
        print("   ", os.path.basename(f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
