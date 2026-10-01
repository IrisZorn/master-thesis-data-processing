# Master Thesis Project

Code for my master's thesis at Johannes Kepler University Linz: object detection and tracking in Cuphead gameplay footage (YOLOv8), plus map generation using optical flow (RAFT).

## Project structure

```
src/
  detect/            # detection and tracking on gameplay videos
  map_generation/    # stitching level maps from frames
  my_utils/          # shared helpers (incl. RAFT feature extraction)
  process_data/      # dataset creation
  training/          # YOLOv8 training and evaluation
RAFT/                # modified copy of RAFT (see below)
```

## Setup

Requires Python 3.11.

```bash
conda create -n master_thesis_project python=3.11
conda activate master_thesis_project
pip install -r requirements.txt
bash RAFT/download_models.sh   # RAFT optical flow weights
```

Scripts use relative paths and must be run from the folder they are in. The project root, `src/` and `RAFT/core/` must be on `PYTHONPATH` (PyCharm does this automatically when they are marked as source roots). From the project root:

```powershell
# Windows (PowerShell)
$env:PYTHONPATH = "$PWD;$PWD\src;$PWD\RAFT\core"
cd src\detect
python detect.py
```

```bash
# Linux / macOS
export PYTHONPATH="$PWD:$PWD/src:$PWD/RAFT/core"
cd src/detect
python detect.py
```

## Not included

Gameplay videos (`resources/custom_test_img/**/*.mp4`), extracted frames, detection/tracking outputs (`resources/detection_sets/`) and the processed datasets (`resources/datasets/`) are not part of this repository because of their size and because the footage is copyrighted. Scripts that read them (`detect.py`, `stitch_map.py`, `process_data/create_dataset/*`) expect them at those paths.

## Dataset

The annotated dataset is hosted on Roboflow and is not included in this repository:

- Workspace: https://universe.roboflow.com/completecuphead
- Dataset used for training: [wally_warbles, version 8](https://universe.roboflow.com/completecuphead/wally_warbles/dataset/8) (CC BY 4.0)

Download it in YOLOv8 format and place it in `src/training/training_set/` (next to the included `data.yaml`).

## Model weights

The trained YOLOv8 models used by the code are included in `resources/trained_models/` (weights, training arguments, metrics and plots):

| Run | Base model | Dataset | Used in |
|---|---|---|---|
| `yolov8s_forest_folliesv142` | YOLOv8s | Forest Follies | `src/detect/detect.py` |
| `yolov8s_wally_warblesv8` | YOLOv8s | [wally_warbles v8](https://universe.roboflow.com/completecuphead/wally_warbles/dataset/8) | `src/process_data/create_dataset/detect_and_process.py` |
| `yolov8n_forest_folliesv132` | YOLOv8n | Forest Follies | `src/training/testing.py` |

Other weights are not included:

- **YOLOv8 base weights:** `yolov8n.pt` and `yolov8s.pt` are downloaded automatically by Ultralytics.
- **RAFT:** run `RAFT/download_models.sh` to download the pretrained weights into `RAFT/models/`.

## Generated maps

The stitched Forest Follies level maps produced by `src/map_generation/stitch_map.py` are included in `resources/custom_test_img/forest_follies/maps/`:

- `forest_follies_stitched.png`: stitched level image
- `forest_follies_stitched_map.png`: stitched map mask
- `kp_map.png`: keypoint visualisation
- `merged_keypoints.npz`: merged keypoints and descriptors

## Third-party code

`RAFT/` contains a modified copy of [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://github.com/princeton-vl/RAFT) by Zachary Teed and Jia Deng (Princeton Vision & Learning Lab), licensed under the BSD 3-Clause License (see `RAFT/LICENSE`). Changes were made to `core/__init__.py`, `core/datasets.py` and `train.py`, and package `__init__.py`/`setup.py` files were added so it can be imported from this project.

## Disclaimer

Cuphead is © StudioMDHR Entertainment Inc. This project is not affiliated with or endorsed by StudioMDHR. Game footage is used for non-commercial academic research only.
