"""Topic A - đo độ nhạy của projection LiDAR->ảnh với calibration drift (không dùng random, chạy lại ra cùng số).

Cách chạy (từ gốc repo):
    python -m src.calib_sweep --data-root data/kitti_mini --frames 000001 000004 000008 000011 000019 000049
    python -m src.calib_sweep --help

Metric cho mỗi (frame, mức perturb):
  inside_fov_pct    % điểm LiDAR còn rơi trong ảnh (depth>0 và trong khung hình)
  box_hit_pct       với các điểm nằm TRONG box 3D GT (tính bằng calib gốc), % điểm sau khi
                    perturb vẫn rơi vào box 2D GT của đúng object đó
  median_shift_px   độ dịch pixel trung vị của các điểm đó so với calib gốc
  align_score       điểm khớp biên (depth edge vs image edge), xem edge_alignment_score()
Kết quả lưu ở results/calib_sweep.csv và results/calib_sweep_by_range.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from starter.datasets import load_frame
from starter.projection import box3d_corners_cam, perturb_extrinsic, project_velo_to_image, velo_to_cam

# mức perturb: (tên trục, giá trị). 0 = baseline.
LEVELS = (
    [("none", 0.0)]
    + [("yaw_deg", v) for v in (0.25, 0.5, 1.0, 2.0, 3.0)]
    + [("pitch_deg", v) for v in (0.25, 0.5, 1.0, 2.0, 3.0)]
    + [("roll_deg", v) for v in (0.25, 0.5, 1.0, 2.0, 3.0)]
    + [("t_cm", v) for v in (2.0, 5.0, 10.0)]
)
CLASSES = ("Car", "Pedestrian", "Cyclist", "Van", "Truck")
RANGE_BINS = [(0, 15, "0-15 m"), (15, 30, "15-30 m"), (30, 1e9, "30+ m")]


def make_calib(calib, axis: str, value: float):
    if axis == "none":
        return calib
    if axis == "t_cm":  # dịch 'value' cm theo y của LiDAR (sang trái) - hướng gây lệch ngang
        return perturb_extrinsic(calib, t_xyz_m=(0.0, value / 100.0, 0.0))
    return perturb_extrinsic(calib, **{axis: value})


def points_in_box3d(points_cam: np.ndarray, obj) -> np.ndarray:
    """Mask điểm (N,3, camera frame) nằm trong box 3D (xoay theo rotation_y)."""
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    rel = points_cam - obj.location
    x = c * rel[:, 0] - s * rel[:, 2]
    z = s * rel[:, 0] + c * rel[:, 2]
    y = rel[:, 1]
    return (np.abs(x) <= l / 2) & (np.abs(z) <= w / 2) & (y <= 0.0) & (y >= -h)


def project_all(points: np.ndarray, calib, image_shape):
    """Trả về uv (N,2) cho MỌI điểm (NaN nếu không hợp lệ) và mask hợp lệ."""
    uv, depth, mask = project_velo_to_image(points, calib, image_shape)
    full = np.full((len(points), 2), np.nan)
    full[mask] = uv
    return full, mask


def edge_alignment_score(image: np.ndarray, uv: np.ndarray, depth: np.ndarray,
                         depth_jump_ratio: float = 0.08) -> float:
    """Alignment score kiểu Levinson & Thrun: LiDAR có biên độ sâu (nhảy depth) thì ảnh phải có biên màu.

    1. Dựng depth map thưa (HxW) từ điểm đã chiếu, lấp lỗ bằng min-filter 7x7 (vật gần che vật xa).
    2. Điểm biên độ sâu = điểm có chênh lệch depth > depth_jump_ratio*depth so với láng giềng trong cửa sổ 7x7.
    3. Score = trung bình độ lớn gradient ảnh (Sobel, làm mờ nhẹ, chuẩn hoá về [0,1]) tại các điểm biên đó.
    Score càng cao = biên LiDAR càng trùng biên ảnh. Chuẩn hoá theo frame ở bước so sánh (ratio với baseline).
    """
    h, w = image.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)
    grad /= max(float(np.percentile(grad, 99)), 1e-6)
    grad = np.clip(grad, 0, 1)

    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, d = ui[ok], vi[ok], depth[ok]
    big = np.float32(1e6)
    dmap = np.full((h, w), big, np.float32)
    np.minimum.at(dmap, (vi, ui), d.astype(np.float32))
    k = np.ones((7, 7), np.uint8)
    dmin = cv2.erode(dmap, k)                                # min depth trong cửa sổ
    dmap_inv = np.where(dmap >= big, -1.0, dmap).astype(np.float32)
    dmax = cv2.dilate(dmap_inv, k)                           # max depth trong cửa sổ (bỏ ô trống)
    is_edge = ((dmax[vi, ui] - dmin[vi, ui]) > depth_jump_ratio * np.maximum(d, 1.0)) & (dmin[vi, ui] < big)
    if is_edge.sum() < 20:
        return float("nan")
    return float(grad[vi[is_edge], ui[is_edge]].mean())


def evaluate_frame(data_root: str, frame_id: str):
    fr = load_frame(data_root, frame_id)
    pts, img, calib = fr["points"], fr["image"], fr["calib"]
    pts = pts[np.isfinite(pts[:, :3]).all(axis=1)]
    # object hợp lệ: ít bị cắt, có >= 15 điểm bên trong box 3D
    cam0 = velo_to_cam(pts[:, :3], calib)
    objs = []
    for o in fr["labels"]:
        if o.type not in CLASSES or o.truncated > 0.5 or o.occluded > 1:
            continue
        m = points_in_box3d(cam0, o)
        if m.sum() >= 15:
            objs.append((o, m))
    uv0, mask0 = project_all(pts, calib, img.shape)
    rows, rows_range = [], []
    for axis, val in LEVELS:
        cal = make_calib(calib, axis, val)
        uv, mask = project_all(pts, cal, img.shape)
        score = edge_alignment_score(img, uv[mask], velo_to_cam(pts[mask, :3], cal)[:, 2])
        hits, tot, shifts = 0, 0, []
        for o, m in objs:
            x1, y1, x2, y2 = o.bbox
            u, v = uv[m, 0], uv[m, 1]
            ins = mask[m] & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            hits += int(ins.sum())
            tot += int(m.sum())
            both = mask[m] & mask0[m]
            if both.any():
                shifts.append(np.hypot(*(uv[m][both] - uv0[m][both]).T))
            dist = float(np.linalg.norm(o.location))
            rb = next(name for lo, hi, name in RANGE_BINS if lo <= dist < hi)
            rows_range.append(dict(frame=frame_id, perturb=axis, value=val, range=rb, cls=o.type,
                                   n_pts=int(m.sum()), n_hit=int(ins.sum())))
        rows.append(dict(frame=frame_id, perturb=axis, value=val,
                         inside_fov_pct=100 * mask.mean(),
                         box_hit_pct=100 * hits / max(tot, 1),
                         n_obj=len(objs), n_obj_pts=tot,
                         median_shift_px=float(np.median(np.concatenate(shifts))) if shifts else float("nan"),
                         align_score=score))
    return rows, rows_range


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep calibration drift và đo độ lệch projection (topic A)")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frames", nargs="+", default=["000001", "000004", "000008", "000011", "000019", "000049"])
    ap.add_argument("--out", default="results/calib_sweep.csv")
    args = ap.parse_args()
    all_rows, all_range = [], []
    for f in args.frames:
        r, rr = evaluate_frame(args.data_root, f)
        all_rows += r
        all_range += rr
        print("done", f)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df["align_ratio"] = df["align_score"] / df.groupby("frame")["align_score"].transform(
        lambda s: s.iloc[0])  # chia cho baseline (dòng 'none') của từng frame
    df.to_csv(out, index=False, float_format="%.4f")
    pd.DataFrame(all_range).groupby(["perturb", "value", "range"], as_index=False)[["n_pts", "n_hit"]].sum().assign(
        box_hit_pct=lambda d: 100 * d.n_hit / d.n_pts).to_csv(
        out.with_name(out.stem + "_by_range.csv"), index=False, float_format="%.3f")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
