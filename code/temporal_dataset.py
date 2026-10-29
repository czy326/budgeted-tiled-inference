#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal dataset wrapper for arms C/E.

Contract: a layout produced by make_ann.py, whose frame_index.json maps each
crop to its T frames (stem, coverage, dt_days, per-instance visibility).

Key correctness point: geometric augmentation MUST be identical across all T
frames of a crop, otherwise the temporal correspondence is destroyed. This
wrapper draws the flip flags ONCE per sample and applies the same flags to every
frame and every polygon.

Framework-free (numpy + PIL). If torch is available, `to_torch` also returns
tensors; otherwise numpy arrays are returned. It is NOT an mmrotate Dataset
subclass (that needs the trainer's pipeline); use it as the reference
implementation and the unit-test target.
"""
import json
import os

import numpy as np
from PIL import Image


def load_frame_index(layout_root):
    p = os.path.join(layout_root, "frame_index.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_dota_polys(path):
    polys, labels, diffs = [], [], []
    if not os.path.isfile(path):
        return (np.zeros((0, 8), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.int64))
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                xy = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            polys.append(xy)
            labels.append(hash(parts[8]) % 100000)  # placeholder; map via name2id if needed
            diffs.append(int(float(parts[9])) if len(parts) > 9 else 0)
    if not polys:
        return (np.zeros((0, 8), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.int64))
    return (np.asarray(polys, np.float32), np.asarray(labels, np.int64), np.asarray(diffs, np.int64))


def poly_to_obb(poly):
    """(N,8) polygon -> (N,5) cx,cy,w,h,theta."""
    p = np.asarray(poly, np.float32).reshape(-1, 4, 2)
    if p.shape[0] == 0:
        return np.zeros((0, 5), np.float32)
    center = p.mean(axis=1)
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 2] - p[:, 1]
    w = np.linalg.norm(v1, axis=1)
    h = np.linalg.norm(v2, axis=1)
    theta = np.arctan2(v1[:, 1], v1[:, 0])
    return np.stack([center[:, 0], center[:, 1], w, h, theta], axis=1).astype(np.float32)


def apply_flip_img(img, hflip, vflip):
    if hflip:
        img = img[:, ::-1]
    if vflip:
        img = img[::-1]
    return np.ascontiguousarray(img)


def apply_flip_poly(poly, width, height, hflip, vflip):
    p = np.array(poly, np.float64, copy=True)
    if hflip:
        p[:, 0::2] = (width - 1) - p[:, 0::2]
    if vflip:
        p[:, 1::2] = (height - 1) - p[:, 1::2]
    return p.astype(np.float32)


class TemporalShipDataset:
    def __init__(self, layout_root, split="train", T=12, training=True,
                 seed=2026, hflip_p=0.5, vflip_p=0.5, size=None):
        self.root = layout_root
        self.split = split
        self.T = T
        self.training = training
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.index = load_frame_index(layout_root)
        self.crops = [(cid, info) for cid, info in self.index["crops"].items()
                      if info["split"] == split]
        self.crops.sort(key=lambda x: x[0])
        if not self.crops:
            raise RuntimeError(f"no crops for split={split} under {layout_root}")

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        cid, info = self.crops[idx]
        frames = info["frames"][:self.T]
        while len(frames) < self.T:            # pad defensively
            frames = frames + [frames[-1]]
        img_dir = os.path.join(self.root, self.split, "images")
        imgs = []
        H = W = None
        for fr in frames:
            ip = os.path.join(img_dir, fr["stem"] + ".png")
            arr = np.asarray(Image.open(ip).convert("RGB"), dtype=np.uint8)
            H, W = arr.shape[:2]
            imgs.append(arr)
        imgs = np.stack(imgs, axis=0)          # (T,H,W,3)

        lp = os.path.join(self.root, self.split, "labelTxt", frames[0]["stem"] + ".txt")
        polys, labels, diffs = read_dota_polys(lp)
        vis = np.asarray([fr.get("vis", [1.0] * len(polys)) for fr in frames], np.float32)
        if vis.shape[1] != len(polys):         # align defensively
            n = min(vis.shape[1], len(polys))
            vis = vis[:, :n]
            polys, labels, diffs = polys[:n], labels[:n], diffs[:n]
        dt = np.asarray([fr.get("dt_days", 0) for fr in frames], np.float32)

        if self.size is not None and (H != self.size or W != self.size):
            s = int(self.size)
            imgs = np.stack([np.asarray(Image.fromarray(im).resize((s, s), Image.BICUBIC))
                             for im in imgs], axis=0)
            if len(polys):
                polys = np.array(polys, np.float32, copy=True)
                polys[:, 0::2] *= (s / float(W))
                polys[:, 1::2] *= (s / float(H))
            H = W = s

        if self.training:
            hflip = self.rng.random() < self.hflip_p
            vflip = self.rng.random() < self.vflip_p
        else:
            hflip = vflip = False
        if hflip or vflip:
            imgs = np.stack([apply_flip_img(im, hflip, vflip) for im in imgs], axis=0)
            polys = apply_flip_poly(polys, W, H, hflip, vflip)

        return {
            "crop_id": cid,
            "imgs": imgs,                       # (T,H,W,3) uint8
            "gt_polys": polys,                  # (N,8) float32
            "gt_obb": poly_to_obb(polys),       # (N,5) cx,cy,w,h,theta
            "gt_labels": labels,                # (N,)
            "gt_difficult": diffs,              # (N,)
            "dt": dt,                           # (T,)
            "vis": vis,                         # (T,N)
            "hflip": hflip, "vflip": vflip,
            "img_shape": (H, W),
        }


def collate_temporal(batch):
    """Stack a list of samples. Images -> (B,T,H,W,3) uint8."""
    shapes = {b["imgs"].shape for b in batch}
    if len(shapes) > 1:
        raise ValueError(f"batch has mixed spatial sizes {shapes}; "
                         f"construct the dataset with size=... to fix")
    out = {"crop_id": [b["crop_id"] for b in batch],
           "imgs": np.stack([b["imgs"] for b in batch], axis=0),
           "dt": np.stack([b["dt"] for b in batch], axis=0),
           "img_shape": [b["img_shape"] for b in batch]}
    out["gt_polys"] = [b["gt_polys"] for b in batch]
    out["gt_obb"] = [b["gt_obb"] for b in batch]
    out["gt_labels"] = [b["gt_labels"] for b in batch]
    out["gt_difficult"] = [b["gt_difficult"] for b in batch]
    out["vis"] = [b["vis"] for b in batch]
    return out


def to_torch(sample):
    try:
        import torch
    except ImportError:
        return sample
    s = dict(sample)
    s["imgs"] = torch.from_numpy(sample["imgs"]).permute(0, 3, 1, 2).float()
    s["dt"] = torch.from_numpy(sample["dt"])
    s["vis"] = torch.from_numpy(sample["vis"])
    s["gt_obb"] = torch.from_numpy(sample["gt_obb"])
    s["gt_polys"] = torch.from_numpy(sample["gt_polys"])
    return s


def _selfcheck():
    import sys
    layout = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "_selftest", "out",
                     "train", "layout", "masked_all"))
    ds = TemporalShipDataset(layout, split="train", T=6, training=True, seed=1, size=224)
    assert len(ds) > 0, "empty dataset"
    s = ds[0]
    T, H, W, C = s["imgs"].shape
    assert (T, C, H, W) == (6, 3, 224, 224), s["imgs"].shape
    assert s["dt"].shape == (T,)
    assert s["vis"].shape[0] == T
    assert s["gt_polys"].shape[1] == 8
    assert s["gt_obb"].shape[1] == 5
    assert s["gt_obb"].shape[0] == s["gt_polys"].shape[0]

    # shared flip consistency: same flags applied to every frame
    poly = np.array([[2, 2, 10, 2, 10, 6, 2, 6]], np.float32)
    f = apply_flip_poly(poly, 20, 10, True, False)
    assert np.allclose(f[0, 0::2], [17, 9, 9, 17]), f
    a = np.arange(4 * 5 * 3).reshape(4, 5, 3)
    fa = np.stack([apply_flip_img(x, True, False) for x in a], 0)
    assert np.array_equal(fa, a[:, :, ::-1])

    b = collate_temporal([ds[0], ds[1]])
    assert b["imgs"].shape[:2] == (2, 6)
    assert len(b["gt_obb"]) == 2
    print(f"[temporal_dataset] selfcheck OK  sample={s['crop_id']} "
          f"imgs={s['imgs'].shape} N={s['gt_polys'].shape[0]} dt={s['dt'].tolist()}")


if __name__ == "__main__":
    _selfcheck()
