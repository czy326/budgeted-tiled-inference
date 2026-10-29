#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render one real scene with GT (green) and the sim-trained model's detections (red).

The point is to tell apart two very different failures before writing anything down:
a model that is blind to real ships, versus one that fires on the right objects but is
badly calibrated (which would show up as low AP with visually correct boxes).
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/mnt/e/10/p0_sim/train")
from real_eval import load_rgb  # noqa: E402

from ultralytics import YOLO  # noqa: E402

ROOT = "/mnt/e/sen2ships-main/sen2ships"
W = "/mnt/e/10/p0_sim/data/v4_full/V4_base_full/2026/weights/last.pt"


def main(port="busan", conf=0.05):
    cf = glob.glob(os.path.join(ROOT, "**", "%s*coco.json" % port), recursive=True)[0]
    d = json.load(open(cf, encoding="utf-8"))
    dd = os.path.dirname(cf)
    cnt = {}
    for a in d["annotations"]:
        cnt[a["image_id"]] = cnt.get(a["image_id"], 0) + 1
    sid = max(cnt, key=cnt.get)
    sc = next(i for i in d["images"] if i["id"] == sid)
    p = os.path.join(dd, sc["file_name"])
    if not os.path.isfile(p):
        p = sorted(glob.glob(os.path.join(dd, "*rgb*.*")))[0]

    rgb = load_rgb(p, "pct")
    H, Wd = rgb.shape[:2]
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for a in d["annotations"]:
        if a["image_id"] != sid:
            continue
        x, y, w, h = a["bbox"]
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    m = YOLO(W)
    r = m.predict(source=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=640, conf=conf,
                  max_det=60, device="0", verbose=False)[0]
    n = 0
    if r.obb is not None and len(r.obb):
        pts = r.obb.xyxyxyxy.cpu().numpy()
        for q in pts:
            cv2.polylines(vis, [q.astype(np.int32)], True, (0, 0, 255), 2)
            n += 1
    out = "/mnt/e/10/p0_sim/data/support/%s_check.png" % port
    cv2.imwrite(out, vis)
    print("%s gt=%d preds@%.2f=%d size=%dx%d -> %s" % (port, cnt[sid], conf, n, Wd, H, out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "busan")
