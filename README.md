# Classical Visual Odometry (Incremental SfM) — `run_inference.py`

A monocular **classical incremental Structure-from-Motion** visual-odometry pipeline for
KITTI-style sequences. Given a folder of images it estimates the camera trajectory
(up to a global scale), aligns it to ground truth with a similarity transform, and
writes a trajectory plot plus pose/timing files.

It mirrors the TSformer-VO deep-learning method's CLI and outputs so the two can be
compared directly:

```bash
python run_inference.py --sequences 08 09 10 22
```

---

## 1. What the script does

For **each** sequence you pass:

1. Loads the image list, intrinsics `K` (from `calib.txt`), and ground-truth poses (if present).
2. Runs the incremental SfM pipeline: two-view initialisation → PnP tracking against a
   local map → map expansion by triangulation → (optional) bundle adjustment.
3. Aligns the estimated trajectory to ground truth with a closed-form **Umeyama** similarity
   transform and computes the **Absolute Trajectory Error (ATE)**.
4. Saves the artefacts (plot, poses, timing) under the output directory.

The pipeline is **monocular**, so the absolute scale is recovered only at evaluation time,
through the alignment to ground truth.

---

## 2. Requirements

- **Python 3.8+**
- Packages: `opencv-python`, `numpy`, `scipy`, `matplotlib`, `tqdm`

Install into a fresh environment:

```bash
pip install opencv-python numpy scipy matplotlib tqdm
```

> **This machine:** the project's working environment is the `deeplearning` conda env,
> which already has all of the above. Run with its interpreter directly:
> ```powershell
> C:/Users/EFO/miniconda3/envs/deeplearning/python.exe run_inference.py --sequences 10
> ```

Verify the imports before a long run:

```bash
python -c "import cv2, scipy, numpy, matplotlib, tqdm; print('ok', cv2.__version__)"
```

---

## 3. Expected dataset layout

The loader expects the standard KITTI-odometry layout:

```
<data_root>/
  <seq>/
    image_<camera_id>/*.png      # the frames (e.g. image_0 = left grayscale)
    calib.txt                    # lines 'P0:' .. 'P3:'  (3x4 projection matrices)
<poses_dir>/
  <seq>.txt                      # 12-value 3x4 ground-truth rows (optional)
```

- `K` is taken from the projection matrix line matching `--camera_id` (`P0` for camera 0).
- Ground truth is **optional**: without `<seq>.txt`, the run still produces a trajectory,
  but `scale_to_gt`/`ate_m` are reported as `n/a` and the plot has no GT overlay.
- `--data_root` defaults to the sibling TSformer-VO dataset
  (`../Robotics-Masters-Related/.../TSformer-VO/data/sequences_png`). Point it elsewhere
  with `--data_root` if your images live somewhere else.

---

## 4. Quick start (smoke test)

Run a short sequence on the first 120 frames to confirm everything works end-to-end:

```bash
python run_inference.py --sequences 10 --max_frames 120
```

Expected: a progress bar, a console line like
`views=119 landmarks=4123 ATE=25.190 m scale=0.512`, and these files created:

```
results/classical_vo/trajectory_10.png
results/classical_vo/pred_poses_10.npy
results/classical_vo/pred_poses/10.txt
results/classical_vo/timing.json
```

Open `trajectory_10.png`: a solid-blue estimate should track the dashed-red ground truth.

---

## 5. Command-line reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--sequences` | `08 09 10 22` | One or more sequence IDs to process. |
| `--data_root` | sibling TSformer-VO dataset | Root folder holding `<seq>/image_*/` + `calib.txt`. |
| `--camera_id` | `0` | Which camera's images/intrinsics to use (`image_<id>`, `P<id>`). |
| `--poses_dir` | `<data_root>/poses` | Folder with ground-truth `<seq>.txt` files. |
| `--output_dir` | `results/classical_vo` | Where plots/poses/timing are written. |
| `--stride` | `1` | Use every *N*-th frame (e.g. `2` halves the frame count). |
| `--max_frames` | `0` (all) | Cap the number of (strided) frames processed. |
| `--bundle-adjustment` / `--no-bundle-adjustment` | on | Enable/disable BA (see §6). |

See all flags at any time with:

```bash
python run_inference.py --help
```

---

## 6. With vs. without bundle adjustment

Bundle adjustment (BA) refines poses and 3D points to reduce drift. You can toggle it to
produce the two baselines used in the report:

```bash
# WITH BA (default) — refined poses
python run_inference.py --sequences 09 --output_dir results/classical_vo

# WITHOUT BA — raw PnP / essential-matrix poses
python run_inference.py --sequences 09 --no-bundle-adjustment --output_dir results/classical_vo_no_ba
```

`--no-bundle-adjustment` switches **both** stages off (per-frame motion-only BA *and* the
local sliding-window BA). Use a separate `--output_dir` so the two sets of results don't
overwrite each other.

---

## 7. Outputs explained

All written under `--output_dir` (default `results/classical_vo/`):

| File | Contents |
|------|----------|
| `trajectory_<seq>.png` | Estimate (solid blue) vs. ground truth (dashed red), with start/end markers. |
| `pred_poses_<seq>.npy` | Estimated world→camera poses, one 4×4 matrix per registered view. |
| `pred_poses/<seq>.txt` | The same poses in KITTI's 12-value `3×4`-per-line text format. |
| `timing.json` | Per-sequence metrics (accumulated across runs into one JSON object). |

Each `timing.json` entry looks like:

```json
"10": {
  "elapsed_s": 243.12,
  "frames": 1201,
  "registered_views": 1200,
  "landmarks": 41873,
  "ms_per_frame": 202.4,
  "scale_to_gt": 0.512,
  "ate_m": 25.19
}
```

- **`scale_to_gt`** — the Umeyama scale. Near `1.0` ≈ already metric; far from `1.0`
  reveals the monocular scale gap.
- **`ate_m`** — Absolute Trajectory Error in metres after alignment (lower is better).

---

## 8. Common recipes

```bash
# Full default set (08, 09, 10, 22)
python run_inference.py

# A specific KITTI sequence, all frames
python run_inference.py --sequences 07

# Faster pass on a long sequence (skip every other frame)
python run_inference.py --sequences 00 --stride 2

# Sequence 22 (Isaac Sim): its frames are in image_2, so use camera 2
python run_inference.py --sequences 22 --camera_id 2

# Point at a dataset in a custom location
python run_inference.py --sequences 04 --data_root D:/datasets/kitti/sequences --poses_dir D:/datasets/kitti/poses
```

---

## 9. Visualising saved results

Already-saved `pred_poses_<seq>.npy` can be re-rendered in **raw / aligned / anchored**
frames (without re-running inference) with the helper tool:

```bash
python tools/visualize_poses.py --sequences 07 09 10 --mode all
```

This writes `trajectory_<seq>_<mode>.png` into a `views/` subfolder of the results dir.

---

