#!/usr/bin/env python3
"""
Detect videos with YOLOv8 and turn the detections into datasets, for both
forest_follies and wally_warbles.

Each level is described by a LevelConfig; forest_follies uses
process_data_forest_follies.process(), while wally_warbles uses
process_data_wally_warbles.process(). All level-specific parameters are
set where the LevelConfig objects are built in main().
"""

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import process_data_forest_follies
import process_data_wally_warbles

# Make src/detect importable so we can reuse its detection logic instead of
# duplicating the YOLO tracking call here.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src/detect"))
import detect

PROJECT_ROOT = Path("../../../")


@dataclass
class LevelConfig:
    name: str
    video_dir: Path
    video_glob_patterns: List[str]
    detection_sets_dir: Path
    runs_dir: Path
    model_path: Path
    dataset_dir: Path
    process_fn: Callable[[str, str, str], object]
    conf: float = 0.5
    iou: float = 0.5
    imgsz: int = 640


def get_video_prefix(pattern: str) -> str:
    return pattern.split("*")[0]


def get_video_suffix(video_path: Path, config: LevelConfig) -> str:
    stem = video_path.stem
    for pattern in config.video_glob_patterns:
        prefix = get_video_prefix(pattern)
        if stem.lower().startswith(prefix.lower()):
            return stem[len(prefix):]
    return stem


def find_videos_needing_detection(config: LevelConfig, force_redetect=False):
    """Find videos that need detection.

    Args:
        config: level-specific paths/parameters
        force_redetect: If True, all videos are redetected (existing folders deleted)
    """
    videos = []
    for pattern in config.video_glob_patterns:
        videos += sorted(config.video_dir.glob(pattern))

    # Deduplicate by suffix using a dict
    videos_by_suffix = {}
    for video in videos:
        suffix = get_video_suffix(video, config)
        if suffix not in videos_by_suffix:
            videos_by_suffix[suffix] = video

    videos_needing_detection = []
    for suffix, video in sorted(videos_by_suffix.items()):
        detection_folder = config.detection_sets_dir / f"detect_{config.name}_{suffix}"

        if force_redetect and detection_folder.exists():
            print(f"[{config.name}] Removing existing detection folder: {detection_folder.name}")
            shutil.rmtree(detection_folder)

        if not detection_folder.exists():
            videos_needing_detection.append((video, suffix))
            print(f"[{config.name}] Video needs detection: {video.name} -> {detection_folder.name}")

    return videos_needing_detection


def run_detection_on_video(config: LevelConfig, video_path: Path, suffix: str):
    """Run YOLOv8 tracking on a single video"""
    print(f"\n{'='*60}")
    print(f"[{config.name}] Running detection on: {video_path.name}")
    print(f"{'='*60}")

    # Track folder is keyed by level name + suffix so forest_follies and
    # wally_warbles never collide when sharing the same runs_dir, and so one
    # video's labels can never get mixed with another's.
    run_folder = config.runs_dir / f"track_{config.name}_{suffix}"
    if run_folder.exists():
        shutil.rmtree(run_folder)

    detect.run_detection(
        source=video_path,
        model_path=config.model_path,
        conf=config.conf,
        iou=config.iou,
        imgsz=config.imgsz,
        project=config.runs_dir,
        name=f"track_{config.name}_{suffix}",
        exist_ok=True,
    )

    print(f"[{config.name}] Detection completed for: {video_path.name}")


def move_detection_results_to_detection_sets(config: LevelConfig, suffix: str):
    """Move the detect run folder (video + labels) from runs directory to detection_sets"""
    source_run_dir = config.runs_dir / f"track_{config.name}_{suffix}"
    target_detection_dir = config.detection_sets_dir / f"detect_{config.name}_{suffix}"

    print(f"\n[{config.name}] Moving detection results for suffix: {suffix}")
    print(f"From: {source_run_dir}")
    print(f"To: {target_detection_dir}")

    target_detection_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_detection_dir.exists():
        shutil.rmtree(target_detection_dir)

    # Copy the run folder's contents (video + labels/) into a "track"
    # subfolder so the on-disk layout under detection_sets stays consistent
    # regardless of the unique source folder name.
    if source_run_dir.exists():
        shutil.copytree(source_run_dir, target_detection_dir / "track", dirs_exist_ok=True)
        print(f"[{config.name}] Detection results moved successfully")
    else:
        print(f"[{config.name}] Warning: No detection results found at {source_run_dir}")


def process_detection_folder(config: LevelConfig, suffix: str):
    """Process a detection folder with the level's process_fn"""
    print(f"\n{'='*60}")
    print(f"[{config.name}] Processing detection folder for suffix: {suffix}")
    print(f"{'='*60}")

    detection_dir = config.detection_sets_dir / f"detect_{config.name}_{suffix}"

    video_path = None
    for pattern in config.video_glob_patterns:
        candidate = config.video_dir / f"{get_video_prefix(pattern)}{suffix}.mp4"
        if candidate.exists():
            video_path = candidate
            break

    if video_path is None:
        print(f"[{config.name}] Warning: Video not found for suffix {suffix}")
        return

    output_path = config.dataset_dir / f"{config.name}_{suffix}.pkl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()
        print(f"[{config.name}] Removed existing pickle: {output_path}")

    print(f"[{config.name}] Processing:")
    print(f"  Detection dir: {detection_dir}")
    print(f"  Video: {video_path}")
    print(f"  Output: {output_path}")

    try:
        config.process_fn(
            str(detection_dir / "track/labels"),
            str(video_path),
            str(output_path),
        )
        print(f"[{config.name}] Processing completed successfully")
    except Exception as e:
        print(f"[{config.name}] Error during processing: {e}")
        raise


def run_pipeline(config: LevelConfig, force_redetect=False):
    print(f"Starting {config.name} video detection and processing pipeline...")

    videos_needing_detection = find_videos_needing_detection(config, force_redetect=force_redetect)

    if not videos_needing_detection:
        print(f"[{config.name}] No videos need detection. All videos have corresponding detection folders.")
    else:
        print(f"[{config.name}] Found {len(videos_needing_detection)} video(s) needing detection")

        for video_path, suffix in videos_needing_detection:
            try:
                run_detection_on_video(config, video_path, suffix)
                move_detection_results_to_detection_sets(config, suffix)
                process_detection_folder(config, suffix)
                print(f"\n✓ [{config.name}] Successfully processed {video_path.name}")
            except Exception as e:
                print(f"\n✗ [{config.name}] Error processing {video_path.name}: {e}")
                raise

    # Also process existing detection folders that might not have pickles
    print(f"\n{'='*60}")
    print(f"[{config.name}] Checking for detection folders without corresponding pickles...")
    print(f"{'='*60}")

    all_detection_folders = sorted(config.detection_sets_dir.glob(f"detect_{config.name}_*"))
    for detection_folder in all_detection_folders:
        suffix = detection_folder.name.replace(f"detect_{config.name}_", "")
        output_path = config.dataset_dir / f"{config.name}_{suffix}.pkl"

        if not output_path.exists():
            print(f"\n[{config.name}] Processing {detection_folder.name} (no pickle found)")
            try:
                process_detection_folder(config, suffix)
                print(f"✓ [{config.name}] Successfully processed {suffix}")
            except Exception as e:
                print(f"✗ [{config.name}] Error processing {suffix}: {e}")

    print(f"\n{'='*60}")
    print(f"[{config.name}] Pipeline completed!")
    print(f"{'='*60}")


def main(force_redetect=False, levels=None):
    """Run the detect-and-process pipeline for forest_follies and wally_warbles.

    All level-specific parameters (paths, model, conf/iou/imgsz) live in the
    LevelConfig objects built here, so they can be tuned per level in one
    place. Pass `levels` to override which/how levels run.
    """

    # forest_follies_config = LevelConfig(
    #     name="forest_follies",
    #     video_dir=PROJECT_ROOT / "resources/custom_test_img/forest_follies",
    #     video_glob_patterns=["forest_follies_*.mp4", "Forest_Follies_*.mp4"],
    #     detection_sets_dir=PROJECT_ROOT / "resources/detection_sets",
    #     runs_dir=PROJECT_ROOT / "src/detect/runs/detect",
    #     model_path=PROJECT_ROOT / "resources/trained_models/yolov8s_forest_folliesv142/weights/best.pt",
    #     dataset_dir=PROJECT_ROOT / "resources/datasets/absolute",
    #     process_fn=process_data_forest_follies.process,
    #     conf=0.5,
    #     iou=0.5,
    #     imgsz=640,
    # )

    wally_warbles_config = LevelConfig(
        name="wally_warbles",
        video_dir=PROJECT_ROOT / "resources/custom_test_img/wally_warbles",
        video_glob_patterns=["wally_warbles_*.mp4", "Wally_Warbles_*.mp4"],
        detection_sets_dir=PROJECT_ROOT / "resources/detection_sets",
        runs_dir=PROJECT_ROOT / "src/detect/runs/detect",
        model_path=PROJECT_ROOT / "resources/trained_models/yolov8s_wally_warblesv8/weights/best.pt",
        dataset_dir=PROJECT_ROOT / "resources/datasets/absolute",
        process_fn=process_data_wally_warbles.process,
        conf=0.5,
        iou=0.5,
        imgsz=640,
    )

    if levels is None:
        levels = [wally_warbles_config]

    for config in levels:
        run_pipeline(config, force_redetect=force_redetect)


if __name__ == "__main__":
    main()
