"""
wally_warbles data-processing step.

Unlike forest_follies, the wally_warbles arena is a fixed screen that never
scrolls, so there is no need to estimate frame-to-frame camera motion and
stitch detections onto a level map: each frame's detection coordinates are
already absolute.
"""
import os
import pickle
from pathlib import Path

from my_utils.constants import constants_wally_warbles
from process_data.create_dataset.process_utils import VideoReader, coords_to_cv2


def debounce_has_start(raw_values, min_run=3):
    """
    Suppress single-frame flips of the "start" detection.

    A lone false-negative (missed `start` class mid title-screen) or false-positive
    (spurious `start` hit mid-gameplay) is enough to flip has_start for one frame,
    which the run-transition logic in process() would otherwise read as a real
    start<->gameplay transition and split/duplicate runs. A new state is only
    accepted once it has persisted for `min_run` consecutive frames.
    """
    if not raw_values:
        return []

    result = []
    stable = raw_values[0]
    i = 0
    n = len(raw_values)
    while i < n:
        j = i
        while j < n and raw_values[j] == raw_values[i]:
            j += 1
        run_len = j - i
        if raw_values[i] != stable and run_len >= min_run:
            stable = raw_values[i]
        result.extend([stable] * run_len)
        i = j
    return result


def process(data_path, video_path, save_path, start_debounce_frames=3):
    video = VideoReader(video_path=video_path)
    frame_height, frame_width = video.get_frame(0).shape[:2]

    runs = []

    # None means "no start screen has been observed yet", so a recording
    # that begins mid-gameplay (before any start screen was ever captured)
    # doesn't get counted as a run until a real start->gameplay transition
    # is seen.
    current_run = None
    was_start = False

    files = sorted(Path(data_path).glob('*.txt'), key=os.path.getmtime)
    file_lines = [file.read_text().splitlines() for file in files]
    raw_has_start = [
        any(line.split()[0] == constants_wally_warbles.start for line in lines)
        for lines in file_lines
    ]
    has_start_values = debounce_has_start(raw_has_start, min_run=start_debounce_frames)

    for file, lines, has_start in zip(files, file_lines, has_start_values):
        frame_of_file = int(file.name.split('_')[-1][:-4])

        # transition: start screen → gameplay
        if was_start and not has_start:
            if current_run and any(len(v) > 0 for v in current_run.values()):
                runs.append(current_run)

            current_run = {
                value: []
                for name, value in vars(constants_wally_warbles).items()
                if not name.startswith("__")
            }
        was_start = has_start

        if has_start or current_run is None:
            continue

        for line in lines:
            coord = line.split(' ')

            point_x, point_y = coords_to_cv2(coord[1], coord[2], frame_width, frame_height)
            accuracy = float(coord[5])
            if accuracy > 0.6:
                current_run[coord[0]].append(
                    (float(point_x), float(point_y), frame_of_file)
                )
    runs.append(current_run)
    with open(save_path, "wb") as f:
        pickle.dump(runs, f)

    return runs

if __name__ == "__main__":

    suffix = "pallytime"
    detection_dir = Path("../../../resources/detection_sets/detect_wally_warbles_pallytime")
    video_base = Path("../../../resources/custom_test_img/wally_warbles")
    output_base = Path("../../../resources/datasets/absolute")

    process(
        detection_dir / "track" / "labels",
        video_base / f"wally_warbles_{suffix}.mp4",
        output_base / f"wally_warbles_{suffix}.pkl",
    )