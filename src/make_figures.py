"""Vẽ biểu đồ + ảnh minh hoạ cho topic A từ results/calib_sweep*.csv và dữ liệu KITTI.

    python -m src.calib_sweep          # tạo CSV trước
    python -m src.make_figures
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calib_sweep import make_calib
from starter.datasets import load_frame
from starter.projection import draw_box2d, overlay_points, project_velo_to_image

OUT = Path("results/figures")
DET_THR = 0.95  # align_ratio < 0.95 => báo drift


def curves():
    d = pd.read_csv("results/calib_sweep.csv")
    r = pd.read_csv("results/calib_sweep_by_range.csv")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for p, unit in [("yaw_deg", "deg"), ("pitch_deg", "deg"), ("roll_deg", "deg"), ("t_cm", "cm")]:
        g = d[d.perturb.isin(["none", p])].groupby("value")[["box_hit_pct", "median_shift_px"]].mean()
        ax[0].plot(g.index, g.box_hit_pct, "o-", label=p)
    ax[0].set(xlabel="mức perturb (độ hoặc cm)", ylabel="% điểm trong box 3D vẫn rơi vào box 2D GT",
              title="Box-hit theo mức drift (6 frame KITTI)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    for rng in ["0-15 m", "15-30 m", "30+ m"]:
        g = r[(r.perturb.isin(["none", "yaw_deg"])) & (r.range == rng)]
        ax[1].plot(g.value, g.box_hit_pct, "o-", label=rng)
    ax[1].set(xlabel="yaw (độ)", ylabel="% box-hit", title="Yaw drift: càng xa càng tệ"); ax[1].legend(); ax[1].grid(alpha=.3)
    for p in ["yaw_deg", "pitch_deg", "roll_deg", "t_cm"]:
        g = d[d.perturb.isin(["none", p])].groupby("value").align_ratio.mean()
        ax[2].plot(g.index, g.values, "o-", label=p)
    ax[2].axhline(DET_THR, color="k", ls="--", label=f"ngưỡng {DET_THR}")
    ax[2].set(xlabel="mức perturb", ylabel="align_score / baseline", title="Alignment score (Sobel edge)")
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "sweep_curves.png", dpi=130); plt.close(fig)

    # tỉ lệ frame bị phát hiện drift theo ngưỡng
    det = d[d.perturb != "none"].assign(detected=lambda x: x.align_ratio < DET_THR) \
        .groupby(["perturb", "value"], as_index=False).agg(
            detected_frames=("detected", "sum"), n_frames=("detected", "size"), mean_ratio=("align_ratio", "mean"),
            mean_box_hit_pct=("box_hit_pct", "mean"))
    det.to_csv("results/drift_detection.csv", index=False, float_format="%.3f")
    print(det.to_string())


def demo_overlays():
    """Basic: 3 ảnh overlay ở 3 khoảng cách (gần <6m, trung bình, xa)."""
    for f, tag in [("000019", "near"), ("000011", "mid"), ("000004", "far")]:
        fr = load_frame("data/kitti_mini", f)
        uv, dep, _ = project_velo_to_image(fr["points"], fr["calib"], fr["image"].shape)
        vis = overlay_points(fr["image"], uv, dep)
        for o in fr["labels"]:
            vis = draw_box2d(vis, o.bbox, label=f"{o.type} {np.linalg.norm(o.location):.0f}m")
        cv2.imwrite(str(OUT / f"demo_overlay_{tag}_{f}.png"), vis)


def fail_images():
    """Failure 01: yaw 1.0 và 2.0 độ trên frame có người đi bộ ở xa -> điểm trượt khỏi box."""
    fr = load_frame("data/kitti_mini", "000011")
    panels = []
    for yaw in (0.0, 1.0, 2.0):
        cal = make_calib(fr["calib"], "yaw_deg", yaw) if yaw else fr["calib"]
        uv, dep, _ = project_velo_to_image(fr["points"], cal, fr["image"].shape)
        vis = overlay_points(fr["image"], uv, dep, radius=2)
        for o in fr["labels"]:
            vis = draw_box2d(vis, o.bbox, label=o.type)
        crop = vis[120:300, 380:1000].copy()
        cv2.putText(crop, f"yaw = {yaw} deg", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(crop)
    cv2.imwrite(str(OUT / "fail_01_yaw_1_2deg_pedestrian.png"), np.vstack(panels))

    # Failure 02: translation 10 cm không bị alignment score phát hiện + score tự thân không đơn điệu
    d = pd.read_csv("results/calib_sweep.csv")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for f, g in d[d.perturb.isin(["none", "t_cm"])].groupby("frame"):
        ax.plot(g.value, g.align_ratio, "o-", label=f"frame {f:06d}")
    ax.axhline(DET_THR, color="k", ls="--", label=f"ngưỡng {DET_THR}")
    ax.set(xlabel="dịch ngang LiDAR (cm)", ylabel="align_score / baseline",
           title="FAIL: dịch 10 cm (~5 px) không làm score tụt dưới ngưỡng")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fail_02_translation_undetected_by_score.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    curves(); demo_overlays(); fail_images()
