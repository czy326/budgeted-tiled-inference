#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leave-one-port-out fine-tuning on sen2ships, then measure the placement effect there.

Why tiles for training too: the whole scenes are up to 2201 px wide, so training on them
at imgsz=640 would shrink a 12 px vessel to ~4 px and the model would learn the wrong
scale. Training on native 640 px tiles keeps vessels at their real size, and matches the
tile size used by the placement arms at test time.

Pipeline per fold:
  train on tiles from 9 ports  ->  predict on every tile of the held-out port
  ->  pool scores/TPS across folds (each port is always predicted by a model that never
      saw it) and report AP50 for base / all tiles / prior-ranked / random.

Run inside emssm:
  python lopo_sen2ships.py --folds busan,cartagena --epochs 60 --out ../data/support/lopo_smoke.json
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/mnt/e/10/p0_sim/train")
from real_eval import load_rgb, slice_boxes, var_score  # noqa: E402
from probe_bucket_map import ap50  # noqa: E402

ROOT = "/mnt/e/sen2ships-main/sen2ships"
WORK = "/mnt/e/10/p0_sim/data/sen2ships_tiles"


def find_ports():
    out = []
    for cf in sorted(glob.glob(os.path.join(ROOT, "**", "*coco.json"), recursive=True)):
        d = json.load(open(cf, encoding="utf-8"))
        if not d.get("annotations"):
            continue
        out.append((os.path.basename(os.path.dirname(cf)), cf))
    return out


def scene_of(cf):
    """Return (path, [(cx,cy,w,h)] in native pixels) for one port.

    The coco holds two entries (RGB + NDWI of the same extent); annotations hang off
    one id. Resolving by annotation count instead of by filename is what fixed five
    ports silently reporting zero GT.
    """
    d = json.load(open(cf, encoding="utf-8"))
    dd = os.path.dirname(cf)
    cnt = {}
    for a in d["annotations"]:
        cnt[a["image_id"]] = cnt.get(a["image_id"], 0) + 1
    sid = max(cnt, key=cnt.get) if cnt else None
    sc = next((i for i in d["images"] if i["id"] == sid), None)
    p = os.path.join(dd, str(sc.get("file_name", ""))) if sc else ""
    if not os.path.isfile(p):
        cand = [c for c in sorted(glob.glob(os.path.join(dd, "*rgb*.*")))
                if "ndwi" not in os.path.basename(c).lower()]
        cand += [c for c in sorted(glob.glob(os.path.join(dd, "*True_color*.*")))
                 if "ndwi" not in os.path.basename(c).lower()]
        p = cand[0] if cand else ""
    boxes = []
    for a in d["annotations"]:
        if sid is not None and a["image_id"] != sid:
            continue
        x, y, w, h = a["bbox"]
        boxes.append((x + w / 2.0, y + h / 2.0, w, h))
    return p, boxes


def tile_dataset(port, cf, side, overlap, out_dir):
    """Write native-size tiles + OBB labels (theta=0) for one port. Returns tile count."""
    p, boxes = scene_of(cf)
    if not os.path.isfile(p):
        return 0
    rgb = load_rgb(p, "pct")
    H, W = rgb.shape[:2]
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    n = 0
    for (x0, y0, x1, y1) in slice_boxes(H, W, side, overlap):
        x1, y1 = min(x1, W), min(y1, H)
        tw, th = x1 - x0, y1 - y0
        if tw < side * 0.5 or th < side * 0.5:
            continue
        tile = rgb[y0:y1, x0:x1]
        if tile.shape[0] != side or tile.shape[1] != side:
            tile = cv2.resize(tile, (side, side), interpolation=cv2.INTER_LINEAR)
            sxx, syy = side / float(tw), side / float(th)
        else:
            sxx = syy = 1.0
        rows = []
        for (cx, cy, w, h) in boxes:
            gx, gy = cx - x0, cy - y0
            if not (-w / 2 <= gx - w / 2 and gx + w / 2 <= tw + w / 2):
                pass
            # clip in tile coordinates
            lx = max(0.0, (cx - w / 2) - x0)
            rx = min(float(tw), (cx + w / 2) - x0)
            ty = max(0.0, (cy - h / 2) - y0)
            by = min(float(th), (cy + h / 2) - y0)
            if rx - lx < 2.0 or by - ty < 2.0:
                continue
            area_ratio = ((rx - lx) * (by - ty)) / max(w * h, 1e-6)
            if area_ratio < 0.4:
                continue
            cx2, cy2 = (lx + rx) / 2.0 * sxx, (ty + by) / 2.0 * syy
            w2, h2 = (rx - lx) * sxx, (by - ty) * syy
            cn = side * sxx
            hn = side * syy
            xs = [(cx2 - w2 / 2) / cn, (cx2 + w2 / 2) / cn, (cx2 + w2 / 2) / cn,
                  (cx2 - w2 / 2) / cn]
            ys = [(cy2 - h2 / 2) / hn, (cy2 - h2 / 2) / hn, (cy2 + h2 / 2) / hn,
                  (cy2 + h2 / 2) / hn]
            rows.append("0 " + " ".join("%.6f" % v for pxy in zip(xs, ys) for v in pxy))
        name = "%s_t%04d" % (port, n)
        cv2.imwrite(os.path.join(out_dir, "images", name + ".png"),
                    cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        with open(os.path.join(out_dir, "labels", name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + ("\n" if rows else ""))
        n += 1
    return n


def ensure_tiles(ports, side, overlap):
    if os.path.isdir(os.path.join(WORK, "images")) and \
            len(os.listdir(os.path.join(WORK, "images"))) > 0:
        print("[tiles] reuse %s" % WORK)
        return
    total = 0
    for port, cf in ports:
        k = tile_dataset(port, cf, side, overlap, WORK)
        total += k
        print("  %-12s tiles=%d" % (port, k))
    print("[tiles] total %d -> %s" % (total, WORK))


def write_yaml(train_ports, val_ports, side):
    imgs = sorted(glob.glob(os.path.join(WORK, "images", "*.png")))

    def pick(pred):
        return [p for p in imgs
                if any(os.path.basename(p).startswith(pfx) for pfx in pred)]

    tr, va = pick(train_ports), pick(val_ports)
    lst = os.path.join(WORK, "lists")
    os.makedirs(lst, exist_ok=True)
    trf, vaf = os.path.join(lst, "train.txt"), os.path.join(lst, "val.txt")
    open(trf, "w").write("\n".join(tr) + "\n")
    open(vaf, "w").write("\n".join(va) + "\n")
    y = os.path.join(lst, "data.yaml")
    open(y, "w").write("\n".join([
        "path: %s" % WORK, "train: %s" % trf, "val: %s" % vaf,
        "names:", "  0: vessel", ""]))
    return y, len(tr), len(va)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="", help="comma list of held-out ports; empty = all")
    ap.add_argument("--init", default="yolo11n-obb.yaml",
                    help="starting weights; yaml trains from scratch, a .pt fine-tunes")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--side", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--ks", default="4,8")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import batch_probiou

    ports = find_ports()
    names = [p for p, _ in ports]
    print("[ports] %d: %s" % (len(ports), ", ".join(names)))
    ensure_tiles(ports, args.side, args.overlap)

    folds = [f for f in args.folds.split(",") if f.strip()] or names
    ks = [int(v) for v in args.ks.split(",")]
    conds = ["base", "alltiles"] + ["rankR%d" % k for k in ks] + ["randR%d" % k for k in ks]
    pooled = {c: {"s": [], "t": []} for c in conds}
    n_gt_total = [0]
    per_fold = []

    for held in folds:
        if held not in names:
            print("  skip unknown port %s" % held); continue
        tr_prefix = tuple("%s_t" % n for n in names if n != held)
        va_prefix = ("%s_t" % held,)
        y, ntr, nva = write_yaml(tr_prefix, va_prefix, args.side)
        if nva == 0:
            print("  %-12s no val tiles, skip" % held); continue
        print("== fold %s: train=%d val=%d" % (held, ntr, nva))
        m = YOLO(args.init)
        m.train(data=y, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=2, plots=False, verbose=False,
                project="/mnt/e/10/p0_sim/data/lopo", name=held, exist_ok=True)
        best = os.path.join("/mnt/e/10/p0_sim/data/lopo", held, "weights", "best.pt")
        wpath = best if os.path.isfile(best) else os.path.join(
            "/mnt/e/10/p0_sim/data/lopo", held, "weights", "last.pt")
        model = YOLO(wpath)

        # evaluate on the held-out port, tile by tile
        cf = dict(ports)[held]
        p_img, gt_boxes = scene_of(cf)
        rgb = load_rgb(p_img, "pct")
        H, Wd = rgb.shape[:2]
        gt = np.array([[cx, cy, w, h, 0.0] for (cx, cy, w, h) in gt_boxes], np.float32)
        gt = gt.reshape(-1, 5)
        n_gt_total[0] += len(gt)

        def infer(a):
            r = model.predict(source=a, imgsz=args.imgsz, conf=0.001, max_det=300,
                              device=args.device, verbose=False)[0]
            if r.obb is None or len(r.obb) == 0:
                return np.zeros((0, 6), np.float32)
            q = np.concatenate([r.obb.xywhr.cpu().numpy(),
                                r.obb.conf.cpu().numpy()[:, None]], 1)
            return q[np.argsort(-q[:, 5])]

        def nms(preds, thr=0.5):
            if len(preds) <= 1:
                return preds
            o = np.argsort(-preds[:, 5])
            b = torch.as_tensor(preds[o, :5], dtype=torch.float32)
            with torch.no_grad():
                iou = batch_probiou(b, b).cpu().numpy()
            keep, sup = [], np.zeros(len(b), bool)
            for i in range(len(b)):
                if sup[i]:
                    continue
                keep.append(i); sup |= (iou[i] > thr); sup[i] = True
            return preds[o][np.array(keep, int)]

        def score(cond, pr):
            pr = nms(pr)
            if len(gt) and len(pr):
                with torch.no_grad():
                    iou = batch_probiou(torch.as_tensor(gt, dtype=torch.float32),
                                        torch.as_tensor(pr[:, :5], dtype=torch.float32)
                                        ).cpu().numpy()
            used = np.zeros(len(gt), bool)
            for i, row in enumerate(pr):
                ok = False
                if len(gt):
                    col = iou[:, i].copy(); col[used] = -1.0
                    j = int(np.argmax(col))
                    if col[j] >= 0.5:
                        used[j] = True; ok = True
                pooled[cond]["s"].append(float(row[5]))
                pooled[cond]["t"].append(1.0 if ok else 0.0)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        score("base", infer(bgr))
        boxes = slice_boxes(H, Wd, args.side, args.overlap)
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        wp = []
        for (cx, cy) in centers:
            x0 = int(max(0, min(Wd - args.side, round(cx - args.side / 2.0))))
            y0 = int(max(0, min(H - args.side, round(cy - args.side / 2.0))))
            win = bgr[y0:y0 + args.side, x0:x0 + args.side]
            if win.size == 0:
                wp.append(np.zeros((0, 6), np.float32)); continue
            if win.shape[0] != args.side or win.shape[1] != args.side:
                win = cv2.resize(win, (args.side, args.side))
            q = infer(win)
            if len(q):
                q = q.copy(); q[:, 0] += x0; q[:, 1] += y0
            wp.append(q)
        score("alltiles", np.concatenate(wp + [np.zeros((0, 6), np.float32)], 0))

        smap = var_score(rgb, args.side)
        sh, sw = smap.shape
        val = [float(smap[int(min(max(c[1], 0), sh - 1)), int(min(max(c[0], 0), sw - 1))])
               for c in centers]
        order = sorted(range(len(centers)), key=lambda i: -val[i])
        rng = np.random.default_rng(2026)
        for k in ks:
            for tag, idx in (("rankR%d" % k, order[:k]),
                             ("randR%d" % k, list(rng.permutation(len(centers))[:k]))):
                sel = [wp[i] for i in idx if i < len(wp)]
                score(tag, np.concatenate(sel + [np.zeros((0, 6), np.float32)], 0))
        per_fold.append({"port": held, "n_gt": int(len(gt)), "n_tiles": len(centers),
                         "n_train_tiles": int(ntr), "weights": wpath})
        print("  fold %s done: gt=%d tiles=%d" % (held, len(gt), len(centers)))

    out = {"init": args.init, "epochs": args.epochs, "side": args.side,
           "ks": ks, "folds": per_fold, "n_gt": n_gt_total[0], "conditions": {}}
    for c in conds:
        s_, t_ = pooled[c]["s"], pooled[c]["t"]
        out["conditions"][c] = {
            "ap50": ap50(s_, t_, n_gt_total[0]) if s_ else None,
            "n_pred": len(s_),
            "recall": (float(np.sum(t_)) / n_gt_total[0]) if s_ else None,
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n[lopo] n_gt=%d" % n_gt_total[0])
    for c in conds:
        v = out["conditions"][c]
        print("  %-9s AP50=%s R=%s n_pred=%d" % (
            c, ("%.4f" % v["ap50"]) if v["ap50"] is not None else " n/a",
            ("%.3f" % v["recall"]) if v["recall"] is not None else "n/a", v["n_pred"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
