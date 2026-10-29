#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Axis ①, deployable version: propose a few windows, magnify them, union with the
full-image detections. No GT is used to place windows.

Why the decomposition matters
-----------------------------
The oracle test (one window per GT) showed magnification recovers ~66 pp recall on
low-support objects. But that used GT positions. A real system must *propose* the
windows, and at 18 px the detector's own full-image detections are exactly what is
weak. So there are two distinct failure points, and they must be separated:

    coverage(proposal)  : is the object inside some selected window at all?
    detection(given coverage) : given it is inside, is it found after magnification?

If coverage is high and recall is low, the detector is still the limit. If coverage
is low, the proposal step is the limit and the direction becomes "how to propose
cheaply", not "how to fuse features".

Conditions, all at the same window budget
-----------------------------------------
base      : full image at imgsz 640
prop_k    : k windows centred on the highest-scoring full-image detections (spaced)
unif_k    : k windows on a uniform grid (no proposals at all)
+union    : window detections are unioned with the full-image detections, which is
            what a deployed cascade would do (large objects come from the base pass)

Run inside emssm:
    python probe_proposal.py --weights <ckpt> \
        --layout ../data/p0_dota_sim_v3/train/layout/masked_all --split test \
        --ks 4,9 --side 320 --out ../data/support/proposal_test.json
"""

import argparse
import json
import zlib
import math
import os
import sys

import cv2
import numpy as np

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from probe_bucket_map import ap50, read_dota_xywhr  # noqa: E402

STRIDE_REF = 8
BINS = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 100.0]


def poly_long_side(pts):
    e1 = np.linalg.norm(pts[1] - pts[0])
    e2 = np.linalg.norm(pts[2] - pts[1])
    return float(max(e1, e2))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1.0) ** 2 / n
    return chi2, math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--side", type=int, default=320, help="window side in original px")
    ap.add_argument("--key-arms-only", action="store_true",
                    help="skip the prop/unif arms and keep only alltiles + prior-ranking + "
                         "random. Used when the point is the placement contrast and the "
                         "full six-arm suite would not fit in the runtime budget.")
    ap.add_argument("--min-sep-frac", type=float, default=0.5,
                    help="keep selected window centres at least side*frac apart. "
                         "0.5 (default) allows only ~5 windows on a 393 px canvas, so "
                         "a k=8 run would be silently capped; lower it when the point of "
                         "the run is to match a larger window budget.")
    ap.add_argument("--ks", default="4,9")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.5,
                    help="cross-pass rotated NMS threshold on the union")
    ap.add_argument("--sahi-geom", default="own", choices=["own", "official"],
                    help="own: tile centres from the built-in sliding formula. "
                         "official: same centres from sahi.slicing.get_slice_bboxes, "
                         "so the slicing geometry is the published library's. Use "
                         "official for anything that goes into the paper.")
    ap.add_argument("--sahi", default="",
                    help="comma list of side:overlap, e.g. '196:0.2,262:0.0'. "
                         "Overlapping tiles covering the image, merged by the same "
                         "cross-pass NMS: this is the SAHI-style slicing baseline, and "
                         "the comparison the paper needs. 262:0.0 is roughly the same "
                         "window budget as prop4, so the two are directly comparable.")
    ap.add_argument("--propose", default="model",
                    choices=["model", "tophat", "var", "grad", "random"],
                    help="model: centres from the detector's own low-confidence boxes "
                         "(circular, seed-dependent). tophat: centres from a "
                         "morphological white top-hat saliency map -- deterministic "
                         "and independent of the detector.")
    ap.add_argument("--all-frames", action="store_true",
                    help="iterate every date of each crop instead of frame 0")
    ap.add_argument("--extra-full", default="1024,1280",
                    help="extra single-pass conditions at these imgsz, as the "
                         "equal-compute control for the cascade")
    ap.add_argument("--limit-crops", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import probiou

    ks = [int(v) for v in args.ks.split(",")]
    S = args.side
    m = args.imgsz / float(S)
    root = os.path.dirname(os.path.dirname(args.layout.rstrip("/\\")))
    p0 = os.path.dirname(root)
    img_raw = os.path.join(p0, "images", "raw", args.split)
    lbl_dir = os.path.join(args.layout, args.split, "labelTxt")

    with open(os.path.join(args.layout, "frame_index.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    crops = [(cid, info) for cid, info in meta["crops"].items()
             if info["split"] == args.split and info["frames"]]
    if args.limit_crops:
        crops = crops[:args.limit_crops]
    if not crops:
        sys.exit("no crops for split %s" % args.split)

    model = YOLO(args.weights)

    def infer(img, imgsz=None):
        res = model.predict(source=img, imgsz=(imgsz or args.imgsz), conf=args.conf,
                            max_det=300, device=args.device, verbose=False)[0]
        if res.obb is None or len(res.obb) == 0:
            return np.zeros((0, 6), np.float32)
        p = np.concatenate([res.obb.xywhr.cpu().numpy(),
                            res.obb.conf.cpu().numpy()[:, None]], 1)
        return p[np.argsort(-p[:, 5])]

    def rot_nms(preds, thr):
        """Greedy rotated-box NMS. Vectorised via fast_nms (1.5 ms vs 9.9 ms naive).

        Without cross-pass suppression the same object found in three overlapping
        windows stays as three boxes: two become false positives, which both inflates
        FPPI and drags AP down. Any cascade needs it.
        """
        try:
            from fast_nms import rot_nms_fast
            return rot_nms_fast(preds, thr)
        except Exception:                                   # noqa: BLE001
            pass
        _naive = True
        if len(preds) <= 1:
            return preds
        order = np.argsort(-preds[:, 5])
        keep = []
        boxes = preds[order]
        while len(boxes):
            keep.append(boxes[0])
            if len(boxes) == 1:
                break
            iou = probiou(torch.as_tensor(boxes[0:1, :5], dtype=torch.float32),
                          torch.as_tensor(boxes[1:, :5], dtype=torch.float32))
            iou = np.asarray(iou).reshape(-1)
            boxes = boxes[1:][iou < thr]
        return np.stack(keep, 0)

    def score_map(img, kind, side):
        """Integrated saliency for a window of `side` pixels, for several priors.

        Keeping the priors swappable is the point: if the effect only appears with
        one hand-tuned filter it is a scene artefact, not a mechanism. `random`
        exists as a control -- if random centres do as well, then the gain comes
        from something other than the prior (e.g. uniform grids clipping at edges).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if kind == "tophat":
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            base = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_TOPHAT, ker).astype(np.float32)
        elif kind == "var":
            mu = cv2.blur(gray, (9, 9))
            mu2 = cv2.blur(gray * gray, (9, 9))
            base = np.maximum(mu2 - mu * mu, 0.0)
        elif kind == "grad":
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            base = np.sqrt(gx * gx + gy * gy)
        else:
            raise ValueError(kind)
        return cv2.blur(base, (side, side), borderType=cv2.BORDER_CONSTANT)

    def centers_from_score(score, k, side, min_sep_frac):
        H, W = score.shape
        out = []
        r2 = (side * min_sep_frac) ** 2
        score = score.copy()
        for _ in range(k):
            y, x = np.unravel_index(int(np.argmax(score)), score.shape)
            if score[y, x] <= 0:
                break
            out.append((float(x), float(y)))
            yy, xx = np.ogrid[:H, :W]
            score[(yy - y) ** 2 + (xx - x) ** 2 <= r2] = -1.0
        return out

    def prior_centers(img, k, side, kind, stem, min_sep_frac):
        if kind == "random":
            rs = np.random.default_rng(zlib.crc32(stem.encode()) )
            H, W = img.shape[:2]
            out = []
            for _ in range(k * 20):
                if len(out) >= k:
                    break
                cx, cy = rs.uniform(0, W), rs.uniform(0, H)
                _sep = side * min_sep_frac
                if all((cx - a) ** 2 + (cy - b) ** 2 >= _sep * _sep for a, b in out):
                    out.append((cx, cy))
            return out
        return centers_from_score(score_map(img, kind, side), k, side, min_sep_frac)


    def sahi_centers(H, W, side, overlap):
        """Tile centres covering the image with overlap.

        With --sahi-geom official this defers the geometry to
        sahi.slicing.get_slice_bboxes so the slicing is the library's, not ours.
        A reviewer will not accept a home-made reimplementation of a named method,
        so this switch exists. If sahi is missing we fall back to our own formula and
        say so loudly rather than silently reporting "official" numbers.

        Note on merging: SAHI's default merge is NMM, which assumes axis-aligned
        boxes. Our detections are rotated, so cross-pass rotated NMS is the correct
        adaptation and is what both arms use -- the comparison stays about geometry.
        """
        if args.sahi_geom == "official":
            try:
                from sahi.slicing import get_slice_bboxes
                boxes = get_slice_bboxes(H, W, slice_height=int(side),
                                         slice_width=int(side),
                                         overlap_height_ratio=overlap,
                                         overlap_width_ratio=overlap,
                                         auto_slice_resolution=False)
                return [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
            except Exception as exc:                       # noqa: BLE001
                if not getattr(args, "_geom_warned", False):
                    print("   [warn] sahi not usable (%s: %s) -> falling back to the "
                          "built-in slice geometry" % (type(exc).__name__, exc))
                    args._geom_warned = True
        stride = max(1, int(round(side * (1.0 - overlap))))
        def axis(n):
            if n <= side:
                return [n / 2.0]
            xs = list(range(0, n - side + 1, stride))
            if xs[-1] != n - side:
                xs.append(n - side)
            return [x + side / 2.0 for x in xs]
        return [(cx, cy) for cy in axis(H) for cx in axis(W)]

    def hits_union(cond, gt, base_preds, win_preds):
        """Per-GT matched flags against base ∪ windows, plus pooled AP bookkeeping."""
        preds = np.concatenate([base_preds] + win_preds, 0) if win_preds else base_preds
        preds = rot_nms(preds, args.nms)
        if len(preds) == 0:
            pool[cond]["n_gt"] += int(len(gt))
            return np.zeros(len(gt), bool)
        preds = preds[np.argsort(-preds[:, 5])]
        used = np.zeros(len(gt), bool)
        hit = np.zeros(len(gt), bool)
        for row in preds:
            iou = probiou(torch.as_tensor(gt[:, :5], dtype=torch.float32),
                          torch.as_tensor(row[None, :5], dtype=torch.float32))
            iou = np.asarray(iou).reshape(-1).copy()
            iou[used] = -1.0
            j = int(np.argmax(iou))
            ok = iou[j] >= args.iou
            if ok:
                used[j] = True
                hit[j] = True
            pool[cond]["scores"].append(float(row[5]))
            pool[cond]["tps"].append(1.0 if ok else 0.0)
        pool[cond]["n_gt"] += int(len(gt))
        return hit

    def window_preds(img, centers, s_side=None):
        out = []
        H, W = img.shape[:2]
        s_side = s_side or S
        for cx, cy in centers:
            x0 = int(round(cx - s_side / 2.0))
            y0 = int(round(cy - s_side / 2.0))
            x0 = max(0, min(max(W - s_side, 0), x0))
            y0 = max(0, min(max(H - s_side, 0), y0))
            win = img[y0:y0 + s_side, x0:x0 + s_side]
            if win.size == 0:
                continue
            win = cv2.resize(win, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            p = infer(win)
            if len(p):
                mm = args.imgsz / float(s_side)
                p = p.copy()
                p[:, 0] = p[:, 0] / mm + x0
                p[:, 1] = p[:, 1] / mm + y0
                p[:, 2] = p[:, 2] / mm
                p[:, 3] = p[:, 3] / mm
            out.append(p)
        return out

    extra = [int(v) for v in args.extra_full.split(",") if v.strip()]
    sahi = []
    for tok in [t for t in args.sahi.split(",") if t.strip()]:
        sd, ov = tok.split(":")
        sahi.append((int(sd), float(ov)))
    sahi_names = ["sahi%d_o%02d" % (sd, int(round(ov * 100))) for sd, ov in sahi]
    conds = (["base"] + ["prop%d" % k for k in ks] + ["unif%d" % k for k in ks]
             + ["propMatch%d" % k for k in ks]
             + ["sahiMatch%d" % k for k in ks] + ["sahiRank%d" % k for k in ks]
             + ["sahiRand%d" % k for k in ks]
             + sahi_names + ["full%d" % s for s in extra])
    pool = {c: {"scores": [], "tps": [], "n_gt": 0} for c in conds}
    img_stats = {"n": 0}
    rec = {c: [] for c in conds}
    cover = {c: [] for c in conds}          # was the GT inside at least one window
    win_count = {c: 0 for c in conds}

    def run_image(stem):
        ip = os.path.join(img_raw, stem + ".png")
        if not os.path.isfile(ip):
            return
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None:
            return
        H, W = img.shape[:2]
        gt = read_dota_xywhr(os.path.join(lbl_dir, stem + ".txt"))
        if gt.shape[0] == 0:
            return
        sizes = []
        with open(os.path.join(lbl_dir, stem + ".txt"), "r",
                  encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.split()
                if len(p) >= 9:
                    pts = np.array([float(v) for v in p[:8]], np.float64).reshape(4, 2)
                    sizes.append(poly_long_side(pts))
        if len(sizes) != gt.shape[0]:
            return
        sizes = np.array(sizes, float)
        r_full = float(args.imgsz) / float(max(H, W))
        sup = sizes * r_full / STRIDE_REF

        base = infer(img)
        img_stats["n"] += 1
        # SAHI-style slicing: independent of k, so it must run once per image.
        # Running it inside the per-k loop silently doubled windows_per_image (caught
        # by a self-check: 17.5 where the tile geometry could only give 8.8).
        for (s_side, s_ov) in sahi:
            cname = "sahi%d_o%02d" % (s_side, int(round(s_ov * 100)))
            sc_centers = sahi_centers(H, W, s_side, s_ov)
            ws = window_preds(img, sc_centers, s_side)
            win_count[cname] += len(sc_centers)
            hs = hits_union(cname, gt, base, ws)
            for gi in range(gt.shape[0]):
                cov = 0.0
                for cx, cy in sc_centers:
                    if abs(gt[gi, 0] - cx) <= s_side / 2.0 and \
                            abs(gt[gi, 1] - cy) <= s_side / 2.0:
                        cov = 1.0
                        break
                rec[cname].append((sup[gi], float(hs[gi])))
                cover[cname].append((sup[gi], cov))

        for s_extra in extra:
            pe = infer(img, s_extra)
            he = hits_union("full%d" % s_extra, gt, pe, [])
            for gi in range(gt.shape[0]):
                rec["full%d" % s_extra].append((sup[gi], float(he[gi])))
                cover["full%d" % s_extra].append((sup[gi], 1.0))
        h0 = hits_union("base", gt, base, [])
        for gi in range(gt.shape[0]):
            rec["base"].append((sup[gi], float(h0[gi])))
            cover["base"].append((sup[gi], 1.0))

        for k in ks:
            if args.propose != "model":
                centers = prior_centers(img, k, S, args.propose, stem, args.min_sep_frac)
            else:
                centers, taken = [], []
                for row in base:
                    cx, cy = float(row[0]), float(row[1])
                    sep = S * args.min_sep_frac
                    if all((cx - a) ** 2 + (cy - b) ** 2 >= sep * sep
                           for a, b in taken):
                        taken.append((cx, cy))
                        centers.append((cx, cy))
                    if len(centers) >= k:
                        break
            if args.key_arms_only:
                # budget still comes from how many centres the prior would take
                win_count["prop%d" % k] += len(centers)
                centers_for_budget = centers
            else:
                wp = window_preds(img, centers)
                win_count["prop%d" % k] += len(centers)
            if args.key_arms_only:
                centers = centers_for_budget
            else:
                hv = hits_union("prop%d" % k, gt, base, wp)
                for gi in range(gt.shape[0]):
                    cov = 0.0
                    for cx, cy in centers:
                        if abs(gt[gi, 0] - cx) <= S / 2.0 and \
                                abs(gt[gi, 1] - cy) <= S / 2.0:
                            cov = 1.0
                            break
                    rec["prop%d" % k].append((sup[gi], float(hv[gi])))
                    cover["prop%d" % k].append((sup[gi], cov))

            # uniform grid
            if args.key_arms_only:
                ucs = []
            g = int(math.ceil(math.sqrt(k))) if not args.key_arms_only else 1
            ucs = []
            for i in range(g):
                for j in range(g):
                    ucs.append(((j + 0.5) * W / g, (i + 0.5) * H / g))
            ucs = ucs[:k]
            if args.key_arms_only:
                wu, hu = [], None
            else:
                wu = window_preds(img, ucs)
                win_count["unif%d" % k] += len(ucs)
                hu = hits_union("unif%d" % k, gt, base, wu)
            if hu is not None:
                for gi in range(gt.shape[0]):
                    cov = 0.0
                    for cx, cy in ucs:
                        if abs(gt[gi, 0] - cx) <= S / 2.0 and \
                                abs(gt[gi, 1] - cy) <= S / 2.0:
                            cov = 1.0
                            break
                    rec["unif%d" % k].append((sup[gi], float(hu[gi])))
                    cover["unif%d" % k].append((sup[gi], cov))

            # --- strictly matched slicing: identical tile size, identical window count ---
            # The un-matched comparison confounded placement with budget (8.75 SAHI tiles
            # vs 7.9 prior windows). Here both use side S and the same n per image, so the
            # only difference is *which* tiles are used.
            n_win = len(centers)
            if n_win:
                grid = sahi_centers(H, W, S, 0.2)
                # The slice grid can hold fewer tiles than the prior selected (small
                # canvases), which silently broke the matched budget on the first
                # attempt (3.0 vs 4.0 windows). Trim both sides to n_ref instead.
                n_ref = min(n_win, len(grid))
                cam = (W / 2.0, H / 2.0)
                near = sorted(grid, key=lambda p: (p[0] - cam[0]) ** 2 +
                              (p[1] - cam[1]) ** 2)[:n_ref]
                _kind = args.propose if args.propose in ("tophat", "var", "grad") else "var"
                smap = score_map(img, _kind, S)
                sh, sw = smap.shape

                def _sv(p, _m=smap, _h=sh, _w=sw):
                    yy = int(min(max(p[1], 0), _h - 1))
                    xx = int(min(max(p[0], 0), _w - 1))
                    return float(_m[yy, xx])

                ranked = sorted(grid, key=lambda p: -_sv(p))[:n_ref]
                # prior_centers returns centres in descending-score order, so a plain
                # slice keeps the top ones and puts every condition at the same budget.
                trimmed = centers[:n_ref]
                # Random subset of the SAME slice grid at the same budget. This is the
                # fair geometric opponent: sahiMatch (centre-out) is arguably a weak
                # ordering, so without this a reviewer can call it a straw man.
                _rr = np.random.default_rng(zlib.crc32((stem + "|" + str(k)).encode()))
                _perm = _rr.permutation(len(grid))[:n_ref]
                rand_sel = [grid[i] for i in _perm]
                arms = [("sahiRank%d" % k, ranked), ("sahiRand%d" % k, rand_sel)]
                if not args.key_arms_only:
                    arms = [("propMatch%d" % k, trimmed), ("sahiMatch%d" % k, near)] + arms
                for tag, sel in arms:
                    wsel = window_preds(img, sel)
                    win_count[tag] += len(sel)
                    hsel = hits_union(tag, gt, base, wsel)
                    for gi in range(gt.shape[0]):
                        cov = 0.0
                        for cx, cy in sel:
                            if abs(gt[gi, 0] - cx) <= S / 2.0 and \
                                    abs(gt[gi, 1] - cy) <= S / 2.0:
                                cov = 1.0
                                break
                        rec[tag].append((sup[gi], float(hsel[gi])))
                        cover[tag].append((sup[gi], cov))
    for ci, (cid, info) in enumerate(crops):
        stems = ([fr["stem"] for fr in info["frames"]]
                 if args.all_frames else [info["frames"][0]["stem"]])
        for stem in stems:
            run_image(stem)
        if (ci + 1) % 5 == 0:
            print("  %d/%d crops" % (ci + 1, len(crops)), flush=True)

    def curve(rows):
        arr = np.array(rows, float)
        idx = np.digitize(arr[:, 0], BINS) - 1
        return [float(arr[idx == b, 1].mean()) if (idx == b).sum() else None
                for b in range(len(BINS) - 1)]

    out = {"weights": os.path.abspath(args.weights), "split": args.split,
           "side": S, "magnification": m, "ks": ks, "bins": BINS, "conditions": {}}
    for c in conds:
        if not rec[c]:
            # --key-arms-only leaves some arms unmeasured; skip them instead of
            # crashing on a 1-D empty array in the reporting block.
            continue
        arr = np.array(rec[c], float)
        covarr = np.array(cover[c], float)
        low = arr[:, 0] < 2.5
        out["conditions"][c] = {
            "recall_by_support": curve(rec[c]),
            "recall_low_support": float(arr[low, 1].mean()) if low.sum() else None,
            "coverage_low_support": float(covarr[low, 1].mean()) if low.sum() else None,
            "n_low": int(low.sum()),
            "windows_per_image": (win_count[c] / max(img_stats["n"], 1)) if c not in ("base",)
                                 else 1.0,
            "n": len(rec[c]),
            "passes_per_image": (float(c.replace("full", "")) and 1.0)
                                if c.startswith("full") else
                                (1.0 + win_count[c] / max(img_stats["n"], 1) if c != "base" else 1.0),
        }

    paired = {}
    base_arr = np.array(rec["base"], float)
    for c in conds:
        if c == "base" or not rec[c]:
            continue
        arr = np.array(rec[c], float)
        n = min(len(base_arr), len(arr))
        low = base_arr[:n, 0] < 2.5
        b = int(((base_arr[:n, 1][low] == 1) & (arr[:n, 1][low] == 0)).sum())
        cc = int(((base_arr[:n, 1][low] == 0) & (arr[:n, 1][low] == 1)).sum())
        chi2, p = mcnemar(b, cc)
        paired[c] = {"n_low": int(low.sum()), "base_only": b, "cond_only": cc,
                     "delta": (cc - b) / max(int(low.sum()), 1), "p": p}
    out["paired_vs_base_low_support"] = paired

    # Precision side: recall alone is not the claim. Unioning many window passes at
    # conf=0.001 floods false positives, so AP and FPPI decide whether the gain is real.
    for c in conds:
        if not rec[c]:
            continue
        sc, tp, ng = pool[c]["scores"], pool[c]["tps"], pool[c]["n_gt"]
        s = np.array(sc, float)
        t = np.array(tp, float)
        keep = s >= 0.25
        out["conditions"][c]["ap50"] = ap50(sc, tp, ng)
        out["conditions"][c]["fppi_025"] = float((t[keep] == 0).sum()) / max(img_stats["n"], 1)
        out["conditions"][c]["n_pred_025_per_image"] = float(keep.sum()) / max(img_stats["n"], 1)
        out["conditions"][c]["recall_at_025"] = (
            float((t[keep] == 1).sum()) / ng) if ng else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[proposal] %s  split=%s  side=%d (m=%.2f)" % (
        os.path.basename(os.path.abspath(args.weights)), args.split, S, m))
    labels = ["%.1f-%.1f" % (BINS[i], BINS[i + 1]) for i in range(len(BINS) - 1)]
    print("  %-8s %8s %8s %8s   lowRec  lowCov  win/img" % (
        "cond", "AP50", "R@.25", "FPPI"))
    for c in conds:
        if c not in out["conditions"]:
            continue
        d = out["conditions"][c]
        print("  %-8s %8.4f %8.3f %8.2f   %6.3f  %6.3f  %6.1f" % (
            c, d["ap50"] if d["ap50"] is not None else -1,
            d["recall_at_025"] or 0.0, d["fppi_025"],
            d["recall_low_support"] or 0.0, d["coverage_low_support"] or 0.0,
            d["windows_per_image"]))
    print("  recall by support:")
    print("  %-8s" % "cond" + "".join("%9s" % l for l in labels))
    for c in conds:
        if c not in out["conditions"]:
            continue
        d = out["conditions"][c]
        print("  %-8s" % c + "".join(
            ("%9.3f" % v) if v is not None else "        -"
            for v in d["recall_by_support"]))
    print("  paired on low support (vs base):")
    for c, d in paired.items():
        print("    %-8s Δ=%+.4f (n=%d, b=%d c=%d, p=%.3g)"
              % (c, d["delta"], d["n_low"], d["base_only"], d["cond_only"], d["p"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
