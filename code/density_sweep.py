#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the placement effect depend on target density?

Real ports turned out to be dense (busan: 54 vessels over 4 tiles), and there the
prior-vs-random gap collapsed from +0.197 in simulation to +0.015..+0.020. The obvious
explanation is sparsity: if any tile already contains vessels, choosing which tiles to
look at buys little.

This measures it directly: run the same four conditions over the simulated test set, then
pool the per-frame (score, TP) records **by the number of GT boxes in that frame** and
report AP50 per density bin. Frames are the unit of binning because AP is pooled, not
averaged -- binning by frame keeps n_gt correct inside each bin.

Run inside emssm:
  python density_sweep.py --weights <v4 ckpt> --ks 4 --side 130 --out ../data/support/density.json
"""

import argparse
import glob
import json
import os
import zlib
import sys

import cv2
import numpy as np

sys.path.insert(0, "/mnt/e/10/p0_sim/train")
from probe_bucket_map import ap50  # noqa: E402

BINS = [(0, 0), (1, 3), (4, 8), (9, 15), (16, 10 ** 6)]


def parse_labelTxt(path, W, H):
    """DOTA labelTxt -> (cx, cy, w, h, theta) in pixels, plus difficulty flags."""
    out = []
    if not os.path.isfile(path):
        return np.zeros((0, 5), np.float32), np.zeros(0, bool)
    for line in open(path, encoding="utf-8", errors="ignore"):
        t = line.strip()
        if not t:
            continue
        parts = t.split()
        if len(parts) < 9:
            continue
        try:
            pts = np.array(parts[:8], dtype=np.float32).reshape(4, 2)
        except ValueError:
            continue
        diff = len(parts) >= 10 and parts[9] in ("1", "1.0")
        c = pts.mean(0)
        e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
        w = float(np.linalg.norm(e1))
        h = float(np.linalg.norm(e2))
        ang = float(np.arctan2(e1[1], e1[0]))
        if w < h:
            w, h, ang = h, w, ang + np.pi / 2.0
        out.append([c[0], c[1], w, h, ang])
        if diff:
            out[-1].append(1.0)
    arr = np.array([r[:5] for r in out], np.float32).reshape(-1, 5)
    diffs = np.array([len(r) > 5 for r in out], bool)
    return arr, diffs


def slice_centres(H, W, side, overlap):
    try:
        from sahi.slicing import get_slice_bboxes
        bs = get_slice_bboxes(H, W, slice_height=side, slice_width=side,
                              overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                              auto_slice_resolution=False)
        return [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bs]
    except Exception:                                   # noqa: BLE001
        st = max(1, int(round(side * (1 - overlap))))
        def ax(n):
            if n <= side:
                return [n / 2.0]
            xs = list(range(0, n - side + 1, st))
            if xs[-1] != n - side:
                xs.append(n - side)
            return [x + side / 2.0 for x in xs]
        return [(x, y) for y in ax(H) for x in ax(W)]


def var_score(g, side):
    g = g.astype(np.float32)
    mu = cv2.blur(g, (9, 9))
    mu2 = cv2.blur(g * g, (9, 9))
    return cv2.blur(np.maximum(mu2 - mu * mu, 0.0), (side, side),
                    borderType=cv2.BORDER_CONSTANT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layout", default="/mnt/e/10/p0_sim/data/p0_dota_sim_v4/train/layout/masked_all")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--side", type=int, default=130)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import batch_probiou

    model = YOLO(args.weights)
    imgs = sorted(glob.glob(os.path.join(args.layout, args.split, "images", "*.png")))
    lab_dir = os.path.join(args.layout, args.split, "labelTxt")
    print("[density] %d frames, side=%d k=%d" % (len(imgs), args.side, args.k))

    def infer(a):
        r = model.predict(source=a, imgsz=args.imgsz, conf=args.conf, max_det=300,
                          device=args.device, verbose=False)[0]
        if r.obb is None or len(r.obb) == 0:
            return np.zeros((0, 6), np.float32)
        q = np.concatenate([r.obb.xywhr.cpu().numpy(),
                            r.obb.conf.cpu().numpy()[:, None]], 1)
        return q[np.argsort(-q[:, 5])]

    def nms(pr, thr=None):
        thr = args.nms if thr is None else thr
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
            sup |= (iou[i] > thr)
            sup[i] = True
        return pr[o][np.array(keep, int)]

    recs = []          # one dict per frame
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        gt, diffs = parse_labelTxt(os.path.join(lab_dir, stem + ".txt"), W, H)
        keep = ~diffs
        gt = gt[keep]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        per = {"stem": stem, "n_gt": int(len(gt)), "cond": {}}

        def score(tag, pr):
            pr = nms(pr)
            tp = []
            used = np.zeros(len(gt), bool)
            if len(gt) and len(pr):
                with torch.no_grad():
                    iou = batch_probiou(torch.as_tensor(gt, dtype=torch.float32),
                                        torch.as_tensor(pr[:, :5], dtype=torch.float32)
                                        ).cpu().numpy()
                for i in range(len(pr)):
                    col = iou[:, i].copy(); col[used] = -1.0
                    j = int(np.argmax(col))
                    if col[j] >= 0.5:
                        used[j] = True; tp.append(1.0); continue
                    tp.append(0.0)
            else:
                tp = [0.0] * len(pr)
            per["cond"][tag] = {"s": [float(v) for v in pr[:, 5]], "t": tp}

        score("base", infer(bgr))
        cs = slice_centres(H, W, args.side, args.overlap)
        wp, off = [], []
        for (cx, cy) in cs:
            x0 = int(max(0, min(W - args.side, round(cx - args.side / 2.0))))
            y0 = int(max(0, min(H - args.side, round(cy - args.side / 2.0))))
            win = bgr[y0:y0 + args.side, x0:x0 + args.side]
            if win.size == 0:
                wp.append(np.zeros((0, 6), np.float32)); off.append((0, 0)); continue
            if win.shape[0] != args.side or win.shape[1] != args.side:
                win = cv2.resize(win, (args.side, args.side))
            q = infer(win)
            if len(q):
                q = q.copy(); q[:, 0] += x0; q[:, 1] += y0
            wp.append(q); off.append((x0, y0))
        per["n_tiles"] = len(cs)
        score("alltiles", np.concatenate(wp + [np.zeros((0, 6), np.float32)], 0))

        smap = var_score(gray, args.side)
        sh, sw = smap.shape
        val = []
        for (cx, cy) in cs:
            yy = int(min(max(cy, 0), sh - 1)); xx = int(min(max(cx, 0), sw - 1))
            val.append(float(smap[yy, xx]))
        order = sorted(range(len(cs)), key=lambda i: -val[i])
        rng = np.random.default_rng(zlib.crc32(stem.encode()))
        k = min(args.k, len(cs))
        for tag, idx in (("rankR", order[:k]), ("randR", list(rng.permutation(len(cs))[:k]))):
            sel = [wp[i] for i in idx if i < len(wp)]
            score(tag, np.concatenate(sel + [np.zeros((0, 6), np.float32)], 0))
        per["k_used"] = k
        recs.append(per)
        if len(recs) % 50 == 0:
            print("   %d frames" % len(recs))

    conds = ["base", "alltiles", "rankR", "randR"]
    out = {"weights": os.path.abspath(args.weights), "side": args.side, "k": args.k,
           "n_frames": len(recs), "bins": []}
    for lo, hi in BINS:
        fs = [r for r in recs if lo <= r["n_gt"] <= hi and r["k_used"] > 0]
        if not fs:
            continue
        n_gt = sum(r["n_gt"] for r in fs)
        row = {"bin": "%d-%s" % (lo, "inf" if hi > 10 ** 5 else hi),
               "n_frames": len(fs), "n_gt": n_gt,
               "gt_per_frame": n_gt / len(fs),
               "tiles": float(np.mean([r["n_tiles"] for r in fs])), "cond": {}}
        for c in conds:
            s_, t_ = [], []
            for r in fs:
                s_ += r["cond"][c]["s"]; t_ += r["cond"][c]["t"]
            row["cond"][c] = {"ap50": ap50(s_, t_, n_gt) if s_ else None,
                              "recall": (sum(t_) / n_gt) if n_gt else None,
                              "n_pred": len(s_)}
        row["gap"] = (row["cond"]["rankR"]["ap50"] - row["cond"]["randR"]["ap50"]
                      if row["cond"]["rankR"]["ap50"] is not None else None)
        out["bins"].append(row)

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n%-9s %6s %6s %5s  %-8s %-8s %-8s %-8s %7s" % (
        "bin(n_gt)", "frames", "n_gt", "tile", "base", "alltiles", "rankR", "randR", "gap"))
    for r in out["bins"]:
        f = lambda c: ("%.4f" % r["cond"][c]["ap50"]) if r["cond"][c]["ap50"] is not None else "  n/a "
        print("%-9s %6d %6d %5.1f  %-8s %-8s %-8s %-8s %+7.3f" % (
            r["bin"], r["n_frames"], r["n_gt"], r["tiles"],
            f("base"), f("alltiles"), f("rankR"), f("randR"),
            r["gap"] if r["gap"] is not None else float("nan")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
