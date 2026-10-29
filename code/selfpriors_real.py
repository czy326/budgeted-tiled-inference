#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Can an in-domain (learned) tile score replace the hand-crafted variance prior?

The hand prior loses to random on real ports at every tile scale (+29.6 pt in simulation,
-2.4..-6.5 pt on sen2ships). Before redesigning anything, test the cheapest learned score
available: the coarse pass's own confidence. Every tile gets the max detection score the
whole-scene pass produced inside it, so selection costs nothing extra and uses the
in-domain detector rather than a fixed filter.

Arms, all at the same tile budget:
  var    local variance of the tile (the hand prior under test)
  self   max detection confidence from the coarse pass inside the tile (free, learned)
  rand   random tiles (control)

Reported per arm: how much ground truth the selected tiles cover, and AP50 after running
the full-resolution passes on them. Each port is scored with the LOPO model that never
saw it.

Run inside emssm:
  python selfpriors_real.py --k 4 --side 640 --out ../data/support/selfprior_real.json
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
from real_eval import load_rgb, slice_boxes  # noqa: E402
from lopo_sen2ships import find_ports, scene_of  # noqa: E402
from probe_bucket_map import ap50  # noqa: E402

WEIGHTS = "/mnt/e/10/p0_sim/data/lopo"


def var_score(rgb, side):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (9, 9))
    mu2 = cv2.blur(g * g, (9, 9))
    return cv2.blur(np.maximum(mu2 - mu * mu, 0.0), (side, side),
                    borderType=cv2.BORDER_CONSTANT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--side", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import batch_probiou

    arms = ["var", "self", "rand"]
    pooled = {a: {"s": [], "t": [], "cov": 0} for a in arms}
    n_gt_total = [0]
    rows = []

    ports = find_ports()
    for port, cf in ports:
        wd = os.path.join(WEIGHTS, port)
        w = os.path.join(wd, "weights", "best.pt")
        if not os.path.isfile(w):
            w2 = os.path.join(wd, "weights", "last.pt")
            if not os.path.isfile(w2):
                print("  %-12s SKIP (no LOPO weights)" % port); continue
            w = w2
        p, boxes = scene_of(cf)
        if not os.path.isfile(p) or not boxes:
            print("  %-12s SKIP (no image/labels)" % port); continue

        rgb = load_rgb(p, "pct")
        H, W = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gt = np.array([[cx, cy, ww, hh, 0.0] for (cx, cy, ww, hh) in boxes], np.float32)
        n_gt_total[0] += len(gt)

        model = YOLO(w)

        def infer(a):
            r = model.predict(source=a, imgsz=args.imgsz, conf=0.001, max_det=300,
                              device=args.device, verbose=False)[0]
            if r.obb is None or len(r.obb) == 0:
                return np.zeros((0, 6), np.float32)
            q = np.concatenate([r.obb.xywhr.cpu().numpy(),
                                r.obb.conf.cpu().numpy()[:, None]], 1)
            return q[np.argsort(-q[:, 5])]

        coarse = infer(bgr)

        bs = slice_boxes(H, W, args.side, args.overlap)
        cent = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bs]
        if not cent:
            continue

        smap = var_score(rgb, args.side)
        sh, sw = smap.shape
        vscore = [float(smap[int(min(max(c[1], 0), sh - 1)), int(min(max(c[0], 0), sw - 1))])
                  for c in cent]
        # learned, free: strongest coarse detection whose centre falls in the tile
        sscore = []
        for (tx, ty) in cent:
            m = 0.0
            for row in coarse:
                if abs(row[0] - tx) <= args.side / 2.0 and abs(row[1] - ty) <= args.side / 2.0:
                    m = max(m, float(row[5]))
            sscore.append(m)

        k = min(args.k, len(cent))
        rng = np.random.default_rng(zlib.crc32(port.encode()))
        sel = {
            "var": [cent[i] for i in sorted(range(len(cent)), key=lambda i: -vscore[i])[:k]],
            "self": [cent[i] for i in sorted(range(len(cent)), key=lambda i: -sscore[i])[:k]],
            "rand": [cent[i] for i in rng.permutation(len(cent))[:k]],
        }

        def cover(sel_cent):
            return sum(1 for (cx, cy, ww, hh) in boxes
                       if any(abs(cx - tx) <= args.side / 2 and abs(cy - ty) <= args.side / 2
                              for (tx, ty) in sel_cent))

        def full(sel_cent):
            """Run the full-resolution passes on the selected tiles and pool scores/TPS."""
            preds = []
            for (cx, cy) in sel_cent:
                x0 = int(max(0, min(W - args.side, round(cx - args.side / 2.0))))
                y0 = int(max(0, min(H - args.side, round(cy - args.side / 2.0))))
                win = bgr[y0:y0 + args.side, x0:x0 + args.side]
                if win.size == 0:
                    continue
                if win.shape[0] != args.side or win.shape[1] != args.side:
                    win = cv2.resize(win, (args.side, args.side))
                q = infer(win)
                if len(q):
                    q = q.copy(); q[:, 0] += x0; q[:, 1] += y0
                preds.append(q)
            preds.append(np.zeros((0, 6), np.float32))
            pr = np.concatenate(preds, 0)
            if len(pr) > 1:
                o = np.argsort(-pr[:, 5])
                b = torch.as_tensor(pr[o, :5], dtype=torch.float32)
                with torch.no_grad():
                    iou = batch_probiou(b, b).cpu().numpy()
                keep, sup = [], np.zeros(len(b), bool)
                for i in range(len(b)):
                    if sup[i]:
                        continue
                    keep.append(i); sup |= (iou[i] > 0.5); sup[i] = True
                pr = pr[o][np.array(keep, int)]
            used = np.zeros(len(gt), bool)
            iou = None
            if len(gt) and len(pr):
                with torch.no_grad():
                    iou = batch_probiou(torch.as_tensor(gt, dtype=torch.float32),
                                        torch.as_tensor(pr[:, :5], dtype=torch.float32)
                                        ).cpu().numpy()
            return pr, used, iou

        row = {"port": port, "n_gt": len(gt), "tiles": len(cent), "k": k}
        for a in arms:
            pr, used, iou = full(sel[a])
            for i in range(len(pr)):
                ok = False
                if iou is not None:
                    col = iou[:, i].copy(); col[used] = -1.0
                    j = int(np.argmax(col))
                    if col[j] >= 0.5:
                        used[j] = True; ok = True
                pooled[a]["s"].append(float(pr[i, 5]))
                pooled[a]["t"].append(1.0 if ok else 0.0)
            c = cover(sel[a])
            pooled[a]["cov"] += c
            row["%s_cov" % a] = c
        rows.append(row)
        print("  %-12s gt=%3d tiles=%2d  cov var=%3d self=%3d rand=%3d" % (
            port, len(gt), len(cent), row["var_cov"], row["self_cov"], row["rand_cov"]))

    n = n_gt_total[0]
    out = {"k": args.k, "side": args.side, "n_gt": n, "ports": rows, "conditions": {}}
    for a in arms:
        out["conditions"][a] = {
            "ap50": ap50(pooled[a]["s"], pooled[a]["t"], n) if pooled[a]["s"] else None,
            "cov": pooled[a]["cov"],
            "cov_frac": pooled[a]["cov"] / n if n else None,
            "recall": (sum(pooled[a]["t"]) / n) if n else None,
        }
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n[%d gt] %-6s %8s %8s %8s" % (n, "arm", "cover%", "AP50", "recall"))
    for a in arms:
        v = out["conditions"][a]
        print("        %-6s %7.1f%% %8.4f %8.3f" % (
            a, 100 * (v["cov_frac"] or 0), v["ap50"] or 0, v["recall"] or 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
