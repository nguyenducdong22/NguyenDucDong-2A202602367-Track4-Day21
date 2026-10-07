"""Bonus B3 - đo latency p50/p95 của các bước kiểm tra calibration (bỏ lần chạy đầu, >= 20 lần).

    python -m src.latency --data-root data/kitti_mini --frame 000011 --runs 30
Ghi results/latency.csv kèm mô tả phần cứng (CPU, số nhân, phiên bản Python/OpenCV).
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from starter.datasets import load_frame
from starter.projection import project_velo_to_image, velo_to_cam
from src.calib_sweep import edge_alignment_score


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def timeit(fn, runs: int) -> np.ndarray:
    fn()  # lần đầu (warm-up) bị bỏ
    out = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo latency projection và alignment score")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", default="000011")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default="results/latency.csv")
    a = ap.parse_args()
    fr = load_frame(a.data_root, a.frame)
    pts, img, cal = fr["points"], fr["image"], fr["calib"]
    uv, depth, mask = project_velo_to_image(pts, cal, img.shape)
    steps = {
        "velo_to_cam": lambda: velo_to_cam(pts[:, :3], cal),
        "project_velo_to_image": lambda: project_velo_to_image(pts, cal, img.shape),
        "edge_alignment_score": lambda: edge_alignment_score(img, uv, depth),
    }
    steps["total_check"] = lambda: (lambda r: edge_alignment_score(img, r[0], r[1]))(project_velo_to_image(pts, cal, img.shape))
    hw = f"{cpu_name()} | {__import__('os').cpu_count()} cores | Python {platform.python_version()} | OpenCV {cv2.__version__}"
    rows = []
    for name, fn in steps.items():
        t = timeit(fn, a.runs)
        rows.append(dict(step=name, runs=a.runs, n_points=len(pts), p50_ms=np.percentile(t, 50),
                         p95_ms=np.percentile(t, 95), mean_ms=t.mean(), hardware=hw))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False, float_format="%.2f")
    print(pd.DataFrame(rows).drop(columns="hardware").to_string(index=False)); print(hw)


if __name__ == "__main__":
    main()
