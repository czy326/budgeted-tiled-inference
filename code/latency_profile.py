#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end latency per condition, with the three stages timed separately.

The AP numbers are quoted against a budget expressed in milliseconds, and so far those
milliseconds were derived from measured throughput (ms/image at batch 16) rather than
timed end to end. That is not the same thing: throughput hides the per-call overhead of
tile extraction, repeated small forwards, and the merge step, all of which fall on the
cascade but not on a single pass.

What is timed, per condition:
  tile   crop + resize of each tile (numpy/OpenCV, CPU)
  infer  the forward pass(es), including ultralytics' own preprocessing
  merge  rotated NMS over the union of all passes
  total  wall clock for the whole condition

Timing uses CUDA events around the forward passes and perf_counter for CPU stages, with
warmup and torch.cuda.synchronize() so nothing is measured while queued.

Run inside emssm:
  python latency_profile.py --weights <v4 ckpt> --out ../data/support/latency.json
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/mnt/e/10/p0_sim/train")

LAYOUT = "/mnt/e/10/p0_sim/data/p0_dota_sim_v4/train/layout/masked_all/test"


def slice_boxes(H, W, side, overlap):
    try:
        from sahi.slicing import get_slice_bboxes
        return [list(b) for b in get_slice_bboxes(
            H, W, slice_height=side, slice_width=side,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            auto_slice_resolution=False)]
    except Exception:                                     # noqa: BLE001
        st = max(1, int(round(side * (1 - overlap))))
        def ax(n):
            if n <= side:
                return [0]
            xs = list(range(0, n - side + 1, st))
            if xs[-1] != n - side:
                xs.append(n - side)
            return xs
        return [[x, y, x + side, y + side] for y in ax(H) for x in ax(W)]


def var_score(gray, side):
    g = gray.astype(np.float32)
    mu = cv2.blur(g, (9, 9))
    mu2 = cv2.blur(g * g, (9, 9))
    return cv2.blur(np.maximum(mu2 - mu * mu, 0.0), (side, side),
                    borderType=cv2.BORDER_CONSTANT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--side", type=int, default=130)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--ks", default="4,8")
    ap.add_argument("--n-img", type=int, default=25)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--merge-cap", type=int, default=0,
                    help="keep only the top-N scores before rotated NMS. 0 = naive "
                         "(O(n^2) on every box). Report both: the naive merge cost is a "
                         "real property of running at conf=0.001 over many tiles.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import batch_probiou

    model = YOLO(args.weights)
    imgs = sorted(glob.glob(os.path.join(LAYOUT, "images", "*.png")))[:args.n_img]
    if not imgs:
        sys.exit("no images under %s" % LAYOUT)
    print("[latency] %d images, side=%d, imgsz=%d" % (len(imgs), args.side, args.imgsz))

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def t_infer(arr):
        sync()
        t0 = time.perf_counter()
        r = model.predict(source=arr, imgsz=args.imgsz, conf=0.001, max_det=300,
                          device=args.device, verbose=False)[0]
        sync()
        ms = (time.perf_counter() - t0) * 1000.0
        if r.obb is None or len(r.obb) == 0:
            p = np.zeros((0, 6), np.float32)
        else:
            p = np.concatenate([r.obb.xywhr.cpu().numpy(),
                                r.obb.conf.cpu().numpy()[:, None]], 1)
        return p, ms

    def merge(preds, cap=0):
        preds = [p for p in preds if len(p)]
        if not preds:
            return np.zeros((0, 6), np.float32)
        pr = np.concatenate(preds, 0)
        if cap and len(pr) > cap:
            # O(n^2) probiou on ~1e4 boxes dominates the whole runtime; a deployment
            # would truncate by score first. Both variants are timed and reported.
            pr = pr[np.argsort(-pr[:, 5])[:cap]]
        if len(pr) <= 1:
            return pr
        o = np.argsort(-pr[:, 5])
        b = torch.as_tensor(pr[o, :5], dtype=torch.float32)
        with torch.no_grad():
            iou = batch_probiou(b, b).cpu().numpy()
        keep, sup = [], np.zeros(len(b), bool)
        for i in range(len(b)):
            if sup[i]:
                continue
            keep.append(i)
            sup |= (iou[i] > 0.5)
            sup[i] = True
        return pr[o][np.array(keep, int)]

    ks = [int(v) for v in args.ks.split(",")]
    conds = ["base", "full1280", "alltiles"] + ["rank%d" % k for k in ks]

    # warmup
    bgr0 = cv2.imread(imgs[0])
    for _ in range(3):
        t_infer(bgr0)
        t_infer(cv2.resize(bgr0, (args.side, args.side)))

    res = {c: {"tile": [], "infer": [], "merge": [], "total": [], "n_tiles": []}
           for c in conds}

    for ip in imgs:
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = slice_boxes(H, W, args.side, args.overlap)
        cent = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        smap = var_score(gray, args.side)
        sh, sw = smap.shape
        val = [float(smap[int(min(max(c[1], 0), sh - 1)), int(min(max(c[0], 0), sw - 1))])
               for c in cent]
        order = sorted(range(len(cent)), key=lambda i: -val[i])

        for rep in range(args.reps):
            # --- base: single pass over the whole frame
            t0 = time.perf_counter()
            p, ms_inf = t_infer(bgr)
            t1 = time.perf_counter()
            merge([p])
            t2 = time.perf_counter()
            res["base"]["tile"].append(0.0)
            res["base"]["infer"].append(ms_inf)
            res["base"]["merge"].append((t2 - t1) * 1000)
            res["base"]["total"].append((t2 - t0) * 1000)
            res["base"]["n_tiles"].append(0)

            # --- full-frame high resolution
            t0 = time.perf_counter()
            pe = model.predict(source=bgr, imgsz=1280, conf=0.001, max_det=300,
                               device=args.device, verbose=False)[0]
            sync()
            t1 = time.perf_counter()
            res["full1280"]["tile"].append(0.0)
            res["full1280"]["infer"].append((t1 - t0) * 1000)
            res["full1280"]["merge"].append(0.0)
            res["full1280"]["total"].append((t1 - t0) * 1000)
            res["full1280"]["n_tiles"].append(0)

            # --- tiles: time extraction once, forwards once, merge once
            t0 = time.perf_counter()
            windows = []
            for (cx, cy) in cent:
                x0 = int(max(0, min(W - args.side, round(cx - args.side / 2.0))))
                y0 = int(max(0, min(H - args.side, round(cy - args.side / 2.0))))
                win = bgr[y0:y0 + args.side, x0:x0 + args.side]
                if win.shape[0] != args.side or win.shape[1] != args.side:
                    win = cv2.resize(win, (args.side, args.side))
                windows.append((win, x0, y0))
            t_tile = (time.perf_counter() - t0) * 1000

            preds, inf_ms = [], 0.0
            for (win, x0, y0) in windows:
                q, ms = t_infer(win)
                inf_ms += ms
                if len(q):
                    q = q.copy(); q[:, 0] += x0; q[:, 1] += y0
                preds.append(q)
            for c in ["alltiles"] + ["rank%d" % k for k in ks]:
                if c == "alltiles":
                    sel = preds
                    nt = len(preds)
                else:
                    kk = min(int(c[4:]), len(preds))
                    sel = [preds[i] for i in order[:kk]]
                    nt = kk
                tm0 = time.perf_counter()
                merge(sel, args.merge_cap)
                tm1 = time.perf_counter()
                inf = inf_ms * nt / max(len(preds), 1)
                res[c]["tile"].append(t_tile)
                res[c]["infer"].append(inf)
                res[c]["merge"].append((tm1 - tm0) * 1000)
                res[c]["total"].append(t_tile + inf + (tm1 - tm0) * 1000)
                res[c]["n_tiles"].append(nt)

    out = {"weights": os.path.abspath(args.weights), "imgsz": args.imgsz,
           "side": args.side, "n_img": len(imgs), "reps": args.reps, "conditions": {}}
    print("\n%-10s %7s %8s %8s %8s %8s %8s" % (
        "cond", "tiles", "tile", "infer", "merge", "total", "p95"))
    for c in conds:
        d = res[c]
        tot = np.array(d["total"])
        row = {"n_tiles": float(np.mean(d["n_tiles"])),
               "tile_ms": float(np.mean(d["tile"])),
               "infer_ms": float(np.mean(d["infer"])),
               "merge_ms": float(np.mean(d["merge"])),
               "total_ms": float(tot.mean()),
               "total_p95_ms": float(np.percentile(tot, 95)),
               "fps": float(1000.0 / max(tot.mean(), 1e-6))}
        out["conditions"][c] = row
        print("%-10s %7.1f %8.2f %8.2f %8.2f %8.2f %8.2f" % (
            c, row["n_tiles"], row["tile_ms"], row["infer_ms"], row["merge_ms"],
            row["total_ms"], row["total_p95_ms"]))

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
