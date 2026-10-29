#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F1: the three quantities this paper is built on, drawn rather than described.

(a) whole scene squeezed into the network input -> objects a few pixels, support below one cell
(b) the same objects inside a native-scale crop -> support around five cells
(c) a 4x4 candidate pool with a quarter of the tiles carrying a target -> prevalence 25%

The point of (c) is that prevalence, not object count, is what decides whether choosing
tiles is a meaningful problem.
"""

import os
import sys

import numpy as np


def main(out="/mnt/e/10/p0_sim/data/figs"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    os.makedirs(out, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))

    sea = "#dfe7ef"
    land = "#c9c2b4"
    ship = "#1f3f66"
    rng = np.random.default_rng(7)

    # ---------------- (a) whole scene at imgsz
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0), 1, 1, color=sea))
    ax.add_patch(Rectangle((0.62, 0.45), 0.38, 0.55, color=land))
    for _ in range(14):
        x, y = rng.uniform(0.06, 0.9), rng.uniform(0.06, 0.9)
        ax.add_patch(Rectangle((x, y), 0.008, 0.02, color=ship))
    ax.grid(False)
    ax.set_title("(a) whole scene into 640 px\nobject ~5 px, support < 1 cell", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)

    # ---------------- (b) native-scale crop
    ax = axes[1]
    ax.add_patch(Rectangle((0, 0), 1, 1, color=sea))
    ax.add_patch(Rectangle((0.05, 0.6), 0.35, 0.4, color=land))
    for _ in range(5):
        x, y = rng.uniform(0.12, 0.85), rng.uniform(0.12, 0.85)
        ax.add_patch(Rectangle((x, y), 0.05, 0.12, color=ship))
    # feature-cell grid, stride 8 of a 640 input -> 80 cells
    for k in range(1, 8):
        ax.axvline(k / 8.0, color="white", lw=0.6, alpha=0.9)
        ax.axhline(k / 8.0, color="white", lw=0.6, alpha=0.9)
    ax.set_title("(b) native-scale crop\nobject ~40 px, support ~5 cells", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)

    # ---------------- (c) prevalence over a candidate pool
    ax = axes[2]
    n = 4
    occupied = {(0, 1), (1, 1), (2, 0), (3, 2)}
    for i in range(n):
        for j in range(n):
            occ = (i, j) in occupied
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                   facecolor="#b9cbe0" if occ else sea,
                                   edgecolor="white", lw=1.2))
            if occ:
                for _ in range(2):
                    x = j + rng.uniform(0.2, 0.7)
                    y = n - 1 - i + rng.uniform(0.2, 0.7)
                    ax.add_patch(Circle((x, y), 0.05, color=ship))
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_title("(c) 16 candidate tiles, 4 occupied\nprevalence = 25%", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "F1_concept." + ext), dpi=300)
    plt.close(fig)
    print("[figs] ->", os.path.join(out, "F1_concept.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/10/p0_sim/data/figs")
