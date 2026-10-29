#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Sentinel-2 evaluation on sen2ships (10 world ports, 10 m, COCO labels).

Why this matters more than another synthetic run
------------------------------------------------
Everything so far lives in a simulation built from DOTA. This uses real L2A imagery
with human annotations: median vessel 12 px, max 60 px at 10 m -- roughly the v4
training regime (13 px in a 640 input), so it is a fair first transfer test.

The key contrast is field of view, not just magnification:
  base      one pass over the whole scene at imgsz=640 (1351 px -> 640, objects ~5 px)
  tiles     native-scale 640 px tiles (objects stay ~12 px), covering the scene
  rank/rand at the same tile budget, only the selection rule differs

If the gain survives here, the claim is not an artefact of the simulation's canvases.

Caveats kept explicit in the output: only 299 annotations over 10 scenes, axis-aligned
COCO boxes are converted to OBB with theta=0, and no per-port tuning is done.

Run inside emssm:
    python real_eval.py --weights <ckpt> --root /mnt/e/sen2ships-main/sen2ships \
        --ks 4,8 --side 640 --out ../data/support/real_sen2ships.json
"""

import argparse
import glob
import json
import math
import zlib
import os
import sys

import cv2
import numpy as np

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from probe_bucket_map import ap50  # noqa: E402


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    chi2 = (abs(b - c) - 1.0) ** 2 / n
    return math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))


def load_rgb(path, norm="pct"):
    """Load a scene as uint8 RGB. Handles 16-bit TIFF and ordinary jpg/png."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as s:
            a = s.read().astype(np.float32)
        if a.shape[0] >= 3:
            a = np.transpose(a[:3], (1, 2, 0))
        else:
            a = np.repeat(a[:1].transpose(1, 2, 0), 3, axis=2)
        if norm == "pct":
            lo, hi = np.percentile(a, 2), np.percentile(a, 98)
            a = (a - lo) / max(hi - lo, 1.0) * 255.0
        elif norm == "linear":          # mimic a plain 16-bit -> 8-bit downcast
            a = a / 256.0
        elif norm == "reflect":         # L2A reflectance (1e4) -> visible range
            a = a / 10000.0 * 255.0 * 2.5
        return np.clip(a, 0, 255).astype(np.uint8)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cannot read %s" % path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def var_score(rgb, side):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (9, 9))
    mu2 = cv2.blur(g * g, (9, 9))
    return cv2.blur(np.maximum(mu2 - mu * mu, 0.0), (side, side),
                    borderType=cv2.BORDER_CONSTANT)


def slice_boxes(H, W, side, overlap):
    """Slice geometry from the sahi library when available, else the same formula."""
    try:
        from sahi.slicing import get_slice_bboxes
        return [list(b) for b in get_slice_bboxes(
            H, W, slice_height=side, slice_width=side,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            auto_slice_resolution=False)]
    except Exception:                                      # noqa: BLE001
        stride = max(1, int(round(side * (1.0 - overlap))))
        def axis(n):
            if n <= side:
                return [0]
            xs = list(range(0, n - side + 1, stride))
            if xs[-1] != n - side:
                xs.append(n - side)
            return xs
        return [[x, y, min(x + side, W), min(y + side, H)]
                for y in axis(H) for x in axis(W)]


def pool_ap(records, n_gt):
    s = [r[0] for r in records]
    t = [r[1] for r in records]
    return ap50(s, t, n_gt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", default="/mnt/e/sen2ships-main/sen2ships")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--side", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--ks", default="4,8")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--norm", default="pct", choices=["pct", "linear", "reflect"],
                    help="radiometric mapping to uint8. The sim model was trained on "
                         "8-bit DOTA-style imagery, so this choice matters a lot; it is "
                         "checked rather than assumed.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import batch_probiou

    ks = [int(v) for v in args.ks.split(",")]
    conds = ["base", "alltiles"] + ["rankR%d" % k for k in ks] + ["randR%d" % k for k in ks]
    pooled = {c: {"scores": [], "tps": []} for c in conds}
    n_gt_total = {"v": 0}
    per_port = []

    model = YOLO(args.weights)

    def infer(img, imgsz=None):
        r = model.predict(source=img, imgsz=(imgsz or args.imgsz), conf=args.conf,
                          max_det=300, device=args.device, verbose=False)[0]
        if r.obb is None or len(r.obb) == 0:
            return np.zeros((0, 6), np.float32)
        p = np.concatenate([r.obb.xywhr.cpu().numpy(),
                            r.obb.conf.cpu().numpy()[:, None]], 1)
        return p[np.argsort(-p[:, 5])]

    def rot_nms(preds, thr):
        if len(preds) <= 1:
            return preds
        order = np.argsort(-preds[:, 5])
        b = torch.as_tensor(preds[order, :5], dtype=torch.float32)
        with torch.no_grad():
            iou = batch_probiou(b, b).cpu().numpy()
        keep, sup = [], np.zeros(len(b), bool)
        for i in range(len(b)):
            if sup[i]:
                continue
            keep.append(i)
            sup |= (iou[i] > thr)
            sup[i] = True
        return preds[order][np.array(keep, int)]

    def score_cond(cond, gt, preds):
        preds = rot_nms(preds, args.nms)
        if len(gt) == 0:
            for row in preds:
                pooled[cond]["scores"].append(float(row[5]))
                pooled[cond]["tps"].append(0.0)
            return
        used = np.zeros(len(gt), bool)
        if len(preds):
            with torch.no_grad():
                iou = batch_probiou(torch.as_tensor(gt[:, :5], dtype=torch.float32),
                                   torch.as_tensor(preds[:, :5], dtype=torch.float32)
                                   ).cpu().numpy()
        for i, row in enumerate(preds):
            col = iou[:, i].copy()
            col[used] = -1.0
            j = int(np.argmax(col)) if len(col) else -1
            ok = j >= 0 and col[j] >= args.iou
            if ok:
                used[j] = True
            pooled[cond]["scores"].append(float(row[5]))
            pooled[cond]["tps"].append(1.0 if ok else 0.0)

    cocos = sorted(glob.glob(os.path.join(args.root, "**", "*coco.json"), recursive=True))
    if not cocos:
        sys.exit("no coco json under %s" % args.root)
    print("[real] %d coco files" % len(cocos))

    for cf in cocos:
        d = json.load(open(cf, encoding="utf-8"))
        d_dir = os.path.dirname(cf)
        imgs = d.get("images", [])
        anns = d.get("annotations", [])
        # The coco ships two entries per scene (RGB + NDWI of the same extent) and the
        # annotations hang off one specific id. Picking the entry by name silently
        # dropped every annotation on five ports (n_gt=0), so resolve it by annotation
        # count instead and only then locate a readable image.
        counts = {}
        for a in anns:
            counts[a.get("image_id")] = counts.get(a.get("image_id"), 0) + 1
        scene_id = max(counts, key=counts.get) if counts else None
        scene = next((im for im in imgs if im.get("id") == scene_id), None)
        if scene is None:
            scene = imgs[0] if imgs else {"file_name": "", "id": scene_id}
        path = os.path.join(d_dir, str(scene.get("file_name", "")))
        if not os.path.isfile(path):
            cand = sorted(glob.glob(os.path.join(d_dir, "*rgb*.*"))) + \
                   sorted(glob.glob(os.path.join(d_dir, "*True_color*.*")))
            cand = [c for c in cand if "ndwi" not in os.path.basename(c).lower()]
            if not cand:
                print("  %-28s SKIP (no image)" % os.path.basename(d_dir))
                continue
            path = cand[0]
        try:
            rgb = load_rgb(path, args.norm)
        except Exception as exc:                            # noqa: BLE001
            print("  %-28s SKIP (%s)" % (os.path.basename(d_dir), exc))
            continue
        H, W = rgb.shape[:2]

        gt = []
        for a in anns:
            if scene_id is not None and a.get("image_id") != scene_id:
                continue
            x, y, w, h = a["bbox"]
            gt.append([x + w / 2.0, y + h / 2.0, w, h, 0.0])
        gt = np.array(gt, np.float32).reshape(-1, 5)
        n_gt_total["v"] += len(gt)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        score_cond("base", gt, infer(bgr))

        boxes = slice_boxes(H, W, args.side, args.overlap)
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        win_preds, win_of = [], []
        for (cx, cy) in centers:
            x0 = int(max(0, min(W - args.side, round(cx - args.side / 2.0))))
            y0 = int(max(0, min(H - args.side, round(cy - args.side / 2.0))))
            win = bgr[y0:y0 + args.side, x0:x0 + args.side]
            if win.size == 0:
                win_preds.append(np.zeros((0, 6), np.float32))
                win_of.append((x0, y0, args.side))
                continue
            win = cv2.resize(win, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            p = infer(win)
            mm = args.imgsz / float(args.side)
            if len(p):
                p = p.copy()
                p[:, 0] = p[:, 0] / mm + x0
                p[:, 1] = p[:, 1] / mm + y0
                p[:, 2] = p[:, 2] / mm
                p[:, 3] = p[:, 3] / mm
            win_preds.append(p)
            win_of.append((x0, y0, args.side))

        score_cond("alltiles", gt, np.concatenate(win_preds + [np.zeros((0, 6), np.float32)], 0))

        smap = var_score(rgb, args.side)
        sh, sw = smap.shape
        def sc(i):
            cx, cy = centers[i]
            return float(smap[int(min(max(cy, 0), sh - 1)), int(min(max(cx, 0), sw - 1))])
        order = sorted(range(len(centers)), key=lambda i: -sc(i))
        rs = np.random.default_rng(zlib.crc32(cf.encode()))
        for k in ks:
            for tag, idx in (("rankR%d" % k, order[:k]),
                             ("randR%d" % k, list(rs.permutation(len(centers))[:k]))):
                sel = [win_preds[i] for i in idx if i < len(win_preds)]
                sel.append(np.zeros((0, 6), np.float32))
                score_cond(tag, gt, np.concatenate(sel, 0))

        per_port.append({"port": os.path.basename(d_dir), "hw": [int(W), int(H)],
                         "n_gt": int(len(gt)), "n_tiles": len(centers)})
        print("  %-12s %dx%d gt=%3d tiles=%2d" % (os.path.basename(d_dir), W, H, len(gt), len(centers)))

    out = {"weights": os.path.abspath(args.weights), "root": args.root,
           "imgsz": args.imgsz, "side": args.side, "ks": ks,
           "n_gt": n_gt_total["v"], "ports": per_port, "conditions": {}}
    for c in conds:
        sc_, tp_ = pooled[c]["scores"], pooled[c]["tps"]
        out["conditions"][c] = {
            "ap50": ap50(sc_, tp_, n_gt_total["v"]) if sc_ else None,
            "n_pred": len(sc_),
            "recall": (float(np.sum(tp_)) / n_gt_total["v"]) if sc_ else None,
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n[real] n_gt=%d" % n_gt_total["v"])
    for c in conds:
        v = out["conditions"][c]
        print("  %-9s AP50=%s  n_pred=%d  R=%s" % (
            c, ("%.4f" % v["ap50"]) if v["ap50"] is not None else "  n/a",
            v["n_pred"], ("%.3f" % v["recall"]) if v["recall"] is not None else "n/a"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
