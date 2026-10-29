#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-cloud-bucket AP50 for OBB checkpoints.

Why this exists
---------------
The pre-registered H1 is about the **high-cloud bucket** (p >= 0.6), but the
first verdict was computed on aggregate val mAP. With cloud coverage spread
evenly over five buckets, an effect that only exists on cloudy frames is diluted
by ~4x in the aggregate. This script closes that gap without retraining: it
re-evaluates existing checkpoints, bucketing every frame by its own coverage.

Method
------
* one frame at a time, ``conf`` low (0.001) so AP sees the full PR curve
* GT from DOTA labelTxt (absolute xywhr), predictions from ``Results.obb``
* matching by probabilistic IoU >= 0.5, greedy by score, one GT per prediction
* AP50 by all-point interpolation, computed **independently per bucket**

Run inside emssm (needs torch):
    python probe_bucket_map.py --weights <ckpt> \
        --layout ../data/p0_dota_sim_v2/train/layout/masked_all \
        --split val --out ../data/bucket_map/C_2026.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image


def read_dota_xywhr(path):
    """DOTA labelTxt -> (N,5) cx,cy,w,h,theta in pixels."""
    rows = []
    if not os.path.isfile(path):
        return np.zeros((0, 5), np.float64)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 9 or p[0] in ("imagesource:", "gsd:"):
                continue
            try:
                pts = np.array([float(v) for v in p[:8]], np.float64).reshape(4, 2)
            except ValueError:
                continue
            cx, cy = pts.mean(axis=0)
            e1 = np.linalg.norm(pts[1] - pts[0])
            e2 = np.linalg.norm(pts[2] - pts[1])
            if e1 >= e2:
                w, h, th = e1, max(e2, 1e-3), np.arctan2(*(pts[1] - pts[0])[::-1])
            else:
                w, h, th = e2, max(e1, 1e-3), np.arctan2(*(pts[2] - pts[1])[::-1])
            rows.append([cx, cy, w, h, th])
    return np.asarray(rows, np.float64).reshape(-1, 5)


def stem_coverage(layout):
    """stem -> frame cloud coverage, from frame_index.json."""
    with open(os.path.join(layout, "frame_index.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    out = {}
    for cid, info in meta["crops"].items():
        for fr in info["frames"]:
            out[fr["stem"]] = float(fr.get("coverage", 0.0))
    return out


def bucket_of(cov, edges=(0.2, 0.4, 0.6, 0.8)):
    b = 0
    for i, e in enumerate(edges):
        if cov >= e:
            b = i + 1
    return b


def ap50(scores, tps, n_gt):
    """All-point interpolated AP at a single IoU threshold."""
    if n_gt == 0:
        return None
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, np.float64)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--neck-ablate", default="none",
                    choices=["none", "shuffle", "zero"],
                    help="evaluate a checkpoint with the D3 ablation active")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    try:
        from ultralytics.utils.metrics import probiou
    except Exception:                                          # noqa: BLE001
        from ultralytics.utils.tal import probiou               # fallback path

    cov_of = stem_coverage(args.layout)
    img_dir = os.path.join(args.layout, args.split, "images")
    lbl_dir = os.path.join(args.layout, args.split, "labelTxt")
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(img_dir, "*.png")))
    if args.limit:
        stems = stems[:args.limit]
    if not stems:
        sys.exit("no images under %s" % img_dir)

    model = YOLO(args.weights)
    handles = []
    if args.neck_ablate != "none":
        import neck_ablate
        handles = neck_ablate.install(model.model, args.neck_ablate)

    # bucket -> {"scores": [], "tps": [], "n_gt": int, "n_frames": int}
    B = {i: {"scores": [], "tps": [], "n_gt": 0, "n_frames": 0}
         for i in range(5)}

    for k, stem in enumerate(stems):
        ip = os.path.join(img_dir, stem + ".png")
        gt = read_dota_xywhr(os.path.join(lbl_dir, stem + ".txt"))
        if gt.shape[0] == 0:
            continue
        cov = cov_of.get(stem)
        if cov is None:
            continue
        b = bucket_of(cov)
        cell = B[b]
        cell["n_frames"] += 1
        cell["n_gt"] += int(gt.shape[0])

        res = model.predict(source=ip, imgsz=args.imgsz, conf=args.conf,
                            max_det=args.max_det, device=args.device,
                            verbose=False)[0]
        if res.obb is None or len(res.obb) == 0:
            continue
        pred = np.concatenate([res.obb.xywhr.cpu().numpy(),
                               res.obb.conf.cpu().numpy()[:, None]], axis=1)
        pred = pred[np.argsort(-pred[:, 5])]

        used = np.zeros(gt.shape[0], bool)
        for row in pred:
            import torch
            iou = probiou(torch.as_tensor(gt[:, :5], dtype=torch.float32),
                          torch.as_tensor(row[None, :5], dtype=torch.float32))
            iou = np.asarray(iou).reshape(-1)
            iou[used] = -1.0
            j = int(np.argmax(iou))
            hit = iou[j] >= args.iou
            if hit:
                used[j] = True
            cell["scores"].append(float(row[5]))
            cell["tps"].append(1.0 if hit else 0.0)

        if (k + 1) % 50 == 0:
            print("  %d/%d frames" % (k + 1, len(stems)), flush=True)

    out = {"weights": os.path.abspath(args.weights), "layout": args.layout,
           "split": args.split, "imgsz": args.imgsz, "iou_thr": args.iou,
           "buckets": {}}
    for b, cell in B.items():
        a = ap50(cell["scores"], cell["tps"], cell["n_gt"])
        out["buckets"][str(b)] = {
            "ap50": a, "n_gt": cell["n_gt"], "n_frames": cell["n_frames"],
            "n_pred": len(cell["scores"]),
        }
    # aggregate over buckets, weighted by GT count (sanity vs model.val())
    tot_gt = sum(c["n_gt"] for c in B.values())
    if tot_gt:
        out["weighted_ap50"] = float(sum(
            (B[b]["n_gt"] / tot_gt) * (ap50(B[b]["scores"], B[b]["tps"], B[b]["n_gt"]) or 0.0)
            for b in B))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[bucket_map] %s" % os.path.basename(os.path.abspath(args.weights)))
    for b in range(5):
        v = out["buckets"][str(b)]
        print("  bucket %d (cov %.1f-%.1f): AP50=%s  nGT=%d frames=%d"
              % (b, b * 0.2, (b + 1) * 0.2,
                 ("%.4f" % v["ap50"]) if v["ap50"] is not None else "-",
                 v["n_gt"], v["n_frames"]))
    print("  weighted AP50 = %.4f  -> %s" % (out.get("weighted_ap50", 0.0), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
