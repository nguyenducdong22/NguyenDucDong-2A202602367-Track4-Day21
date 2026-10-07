# Nop bai len GitHub, chia commit theo checkpoint. Chay tu goc repo (PowerShell):
#   powershell -ExecutionPolicy Bypass -File tools/submit.ps1
$ErrorActionPreference = "Stop"
python tools/check_submission.py
if ($LASTEXITCODE -ne 0) { Write-Host "check_submission con loi, dung."; exit 1 }

function Commit($msg, $paths) {
    git add -- $paths
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { git commit -m $msg } else { Write-Host "Khong co thay doi cho: $msg" }
}
Commit "CP2: viet velo_to_cam, cam_to_image + demo overlay" @("starter/projection.py","results/figures/demo_overlay_far_000004.png","results/figures/demo_overlay_mid_000011.png","results/figures/demo_overlay_near_000019.png")
Commit "CP3: calibration drift sweep (yaw/pitch/roll/translation)" @("src/__init__.py","src/calib_sweep.py","results/calib_sweep.csv","results/calib_sweep_by_range.csv","results/drift_detection.csv","results/figures/sweep_curves.png")
Commit "CP4: failure case + alignment score" @("src/make_figures.py","results/figures/fail_01_yaw_1_2deg_pedestrian.png","results/figures/fail_02_translation_undetected_by_score.png")
Commit "CP5: nuScenes, latency, colab notebook, report" @("src/latency.py","src/run_on_colab.ipynb","results/calib_sweep_nuscenes.csv","results/latency.csv","report/REPORT.md","tools/submit.ps1")
git add -A
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { git commit -m "CP5: final cleanup" }
git push origin main
Write-Host "Commit hash cuoi (nop len LMS):"
git rev-parse HEAD
