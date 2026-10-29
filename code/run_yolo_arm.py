#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train one arm with ultralytics YOLO-OBB and write a traceable run_meta.json.

Replaces the earlier "python tools/train.py <mmrotate cfg>" placeholder: with
the OBB baseline we do not need an external config file, every knob is an
argument and gets recorded.

Example
-------
    python run_yolo_arm.py --arm A --seed 2026 \
        --data ../data/yolo_obb/best/data.yaml \
        --out ../data/runs --epochs 40 --imgsz 640 --batch 8 --device 0

run_meta.json records what A1/A2/A3 need: real parameter count, the argv used,
the ultralytics version, and the git commit of this repo if available.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np


def git_commit(cwd):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                          # noqa: BLE001
        return ""


def count_params(model):
    try:
        return int(sum(p.numel() for p in model.model.parameters()))
    except Exception:                                          # noqa: BLE001
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", required=True, help="data.yaml")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--weights", default="yolo11n-obb.yaml",
                    help="model yaml (from scratch) or .pt (finetune)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no-val", action="store_true",
                    help="skip per-epoch validation. Needed for the neck "
                         "ablation: a broken branch emits thousands of boxes at "
                         "conf=0.001, making NMS-dominated val ~60x slower than "
                         "training (4.5 min/epoch vs 30 s). Evaluate the final "
                         "weights once with probe_bucket_map instead.")
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--weight-mode", default="", choices=["", "reliability", "uniform"])
    ap.add_argument("--frame-index", default=None,
                    help="layout dir holding frame_index.json (required for C/E)")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--no-online", action="store_true",
                    help="do not refresh the score cache during training")
    ap.add_argument("--drift-every", type=int, default=20,
                    help="measure the drift/delta_c probes every N loss calls; "
                         "1 = every batch (use for short post-hoc probes)")
    ap.add_argument("--neck-ablate", default="none",
                    choices=["none", "shuffle", "zero"],
                    help="D3: destroy the spatial correspondence of the top-down "
                         "deep tensor without changing any weight or channel count")
    ap.add_argument("--amp", action="store_true",
                    help="default off: the AMP self-check downloads yolo26n.pt from "
                         "github, which is unreachable here and costs ~35s of retries")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    from ultralytics import __version__ as ulver
    from ultralytics import YOLO

    run_dir = os.path.abspath(os.path.join(args.out, args.arm, str(args.seed)))
    os.makedirs(run_dir, exist_ok=True)

    hook_state = None
    if args.weight_mode:
        if not args.frame_index:
            sys.exit("--weight-mode needs --frame-index (the layout with "
                     "frame_index.json); without it there is no (crop, frame) map")
        import p0_tal_hook as H
        layout_meta = H.load_layout_meta(args.frame_index)
        H.set_layout(layout_meta)
        H.patch_tal(verbose=True)
        hook_state = H.patch_loss(layout_meta, weight_mode=args.weight_mode,
                                  tau=args.tau, online=not args.no_online,
                                  drift_every=args.drift_every)
        print("[run_yolo_arm] hook armed: crops=%d weight_mode=%s tau=%.1f online=%s"
              % (len(layout_meta), args.weight_mode, args.tau, not args.no_online))

    model = YOLO(args.weights)
    if args.neck_ablate != "none":
        import neck_ablate

        # Model.train() REBUILDS the architecture from model.yaml
        # (trainer.get_model(cfg=self.model.yaml)) and then deepcopies it into
        # ModelEMA, so surgery applied here would be silently discarded -- the
        # run would train an intact model and every number would look normal.
        # Apply it from callbacks, after both objects exist, and assert it stuck.
        def _apply_ablation(trainer):
            n1 = neck_ablate.install(trainer.model, args.neck_ablate, verbose=False)
            n2 = 0
            ema = getattr(trainer, "ema", None)
            inner = getattr(ema, "ema", None) if ema is not None else None
            if inner is not None:
                n2 = neck_ablate.install(inner, args.neck_ablate, verbose=False)
            got = neck_ablate.count_ablated(trainer.model)
            print("[neck_ablate] on_train_start: model=+%d ema=+%d total_ablated=%d"
                  % (n1, n2, got))
            if got < 1:
                raise RuntimeError("neck ablation did not stick on trainer.model")

        model.add_callback("on_train_start", _apply_ablation)
        model.add_callback("on_train_epoch_start", _apply_ablation)
    kwargs = dict(data=os.path.abspath(args.data), epochs=args.epochs,
                  imgsz=args.imgsz, batch=args.batch, device=args.device,
                  workers=args.workers, patience=args.patience,
                  project=os.path.dirname(run_dir),
                  name=os.path.basename(run_dir), exist_ok=True,
                  seed=args.seed, deterministic=True, plots=False,
                  amp=args.amp, val=not args.no_val, verbose=True)
    if args.lr0 is not None:
        kwargs["lr0"] = args.lr0

    t0 = datetime.now(timezone.utc)
    results = model.train(**kwargs)

    meta = {
        "arm": args.arm, "seed": args.seed,
        "weight_mode": args.weight_mode,
        "baseline": "ultralytics YOLO-OBB (%s)" % args.weights,
        "ultralytics": ulver,
        "data": os.path.abspath(args.data),
        "imgsz": args.imgsz, "batch": args.batch, "epochs": args.epochs,
        "lr0": args.lr0, "patience": args.patience,
        "neck_ablate": args.neck_ablate,
        "params": count_params(model),
        "code_commit": git_commit(os.path.dirname(os.path.abspath(__file__))),
        "started_utc": t0.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "out_dir": run_dir,
        "notes": args.notes,
    }
    try:
        metrics = results.results_dict if results is not None else {}
        for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                  "metrics/precision(B)", "metrics/recall(B)"):
            if k in metrics:
                meta[k] = float(metrics[k])
    except Exception:                                          # noqa: BLE001
        pass

    if hook_state is not None:
        st = dict(hook_state["stats"])
        cache = hook_state["cache"]
        filled = total = 0
        for arr in cache.values():
            filled += int(np.isfinite(arr).any(axis=(1, 2)).sum())
            total += int(arr.shape[0])
        gap_n = st.pop("gap_n", 0) or 0
        raw_drift = st.pop("drift", {}) or {}
        drift = {}
        for k in sorted(raw_drift):
            s, n = raw_drift[k]
            drift[k] = {"mean_drift": (s / n) if n else None, "n": int(n)}
        raw_dc = st.pop("delta_c", {}) or {}
        delta_c = {}
        for k in sorted(raw_dc):
            sc, ss, n = raw_dc[k]
            delta_c[k] = {
                "mean_delta_cur": (sc / n) if n else None,
                "mean_delta_cons": (ss / n) if n else None,
                "reduction": ((sc - ss) / n) if n else None,   # >0 = pulled towards the clean frame
                "n": int(n),
            }
        meta["p0_hook"] = {
            "weight_mode": hook_state["weight_mode"],
            "tau": hook_state["tau"],
            "online": hook_state["online"],
            "stats": st,
            "cache_fill": {"filled_frames": filled, "total_frames": total},
            "mean_consensus_gap": (st.get("consensus_gap", 0.0) / gap_n) if gap_n else 0.0,
            "drift_by_cloud_bucket": drift,
            "delta_c_by_cloud_bucket": delta_c,
        }

    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # A probe that silently measured nothing looks exactly like "no effect".
    # Fail loudly instead: the artifacts stay on disk, but the exit code is
    # non-zero so a batch script or a human cannot mistake it for a clean run.
    bad = []
    if hook_state is not None:
        h = meta["p0_hook"]
        if not h["stats"].get("drift_calls"):
            bad.append("drift measured 0 times")
        if not h["stats"].get("delta_c_calls"):
            bad.append("delta_c measured 0 rows")
        if h["stats"].get("measure_error"):
            bad.append("measure_error=%d" % h["stats"]["measure_error"])
        if not h["cache_fill"]["filled_frames"]:
            bad.append("score cache never filled")
    if bad:
        print("[run_yolo_arm] PROBE BROKEN: %s" % "; ".join(bad))
        return 3

    print("[run_yolo_arm] arm=%s seed=%d params=%s"
          % (args.arm, args.seed, meta["params"]))
    for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if k in meta:
            print("  %-20s %.4f" % (k, meta[k]))
    print("  run_meta -> %s" % os.path.join(run_dir, "run_meta.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
