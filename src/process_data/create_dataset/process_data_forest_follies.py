import os
import pickle
from pathlib import Path
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

from my_utils.constants import constants_forest_follies

from src.my_utils.ml_feature_extract import extract_motion_layer
from process_utils import load_keypoints, should_stitch, coords_to_cv2, \
    filter_map_points, VideoReader, is_scattered

level_map = cv2.imread("../../../resources/custom_test_img/forest_follies/maps/forest_follies_stitched.png")
level_map_mask = cv2.imread("../../../resources/custom_test_img/forest_follies/maps/forest_follies_stitched_map.png")
kp_m, des_m = load_keypoints("../../../resources/custom_test_img/forest_follies/maps/merged_keypoints.npz")

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
ransac_x = RANSACRegressor(LinearRegression(),
                           min_samples=2)
ransac_y = RANSACRegressor(LinearRegression(),
                           min_samples=2)
map_height, map_width = level_map.shape[:2]


# @profile
def process(data_path, video_path, save_path):
    last_coords = (0, 0)
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
    # removes last frame, since feature extraction not possible there
    for file in files[:-1]:
        frame_of_file = int(file.name.split('_')[-1][:-4])
        frame_counter = frame_of_file

        lines = file.read_text().splitlines()

        has_start = any(
            line.split()[0] == constants_forest_follies.start
            for line in lines
        )

        # transition: start screen → gameplay
        if was_start and not has_start:
            if current_run and any(len(v) > 0 for v in current_run.values()):
                runs.append(current_run)

            current_run = {
                value: []
                for name, value in vars(constants_forest_follies).items()
                if not name.startswith("__")
            }
            # New run starts at a fresh spawn point, not where the
            # previous run ended, so the map localization search window
            # (filter_map_points/should_stitch) must not stay anchored
            # to the last run's final position.
            last_coords = (0, 0)
        was_start = has_start

        if has_start or current_run is None:
            continue

        frame1 = video.get_frame(frame_of_file - 1)
        frame2 = video.get_frame(frame_of_file)
        frame_x, frame_y = get_abs_frame_coords(frame1, frame2, last_coords, frame_height, frame_width)
        print("Frame counter: " + str(frame_counter))
        if frame_x is None or frame_y is None:
            continue
        last_coords = (float(frame_x), float(frame_y))
        for line in lines:
            coord = line.split(' ')

            rel_x, rel_y = coords_to_cv2(coord[1], coord[2], frame_width, frame_height)
            accuracy = float(coord[5])
            if accuracy > 0.6:
                point_x, point_y = frame_x + rel_x, frame_y + rel_y
                if point_x is not None and point_y is not None:
                    current_run[coord[0]].append(
                        (float(point_x), float(point_y), frame_counter)
                    )
    runs.append(current_run)
    with open(save_path, "wb") as f:
        pickle.dump(runs, f)

    return runs


def get_abs_frame_coords(img1, img2, last_coords, frame_height, frame_width):
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    mask = extract_motion_layer(frame1=img1, frame2=img2, beginning=True)
    if is_scattered(mask):
        print("skipped because scattered")
        return None, None

    # mask = np.ones(img1_grey.shape[:2], dtype=np.uint8) * 255
    kp1, des1 = sift.detectAndCompute(img1_grey, mask)
    kp1 = list(kp1)
    filtered_kp_m, filtered_des_m = filter_map_points(kp_m, des_m, last_coords, frame_height, frame_width)

    matches = bf.match(des1, filtered_des_m)
    good_matches = sorted(matches, key=lambda x: x.distance)[:50]
    if len(good_matches) < 3:
        return None, None

    # image_keypoints = cv2.drawKeypoints(level_map, filtered_kp_m, None)
    # cv2.imshow('Image 2 with Keypoints', image_keypoints)
    # cv2.waitKey(0)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts_m = np.float32([filtered_kp_m[m.trainIdx].pt for m in good_matches])

    # position of image in map coordinates (when points are overlapped)
    diff = pts_m - pts1
    # Separate x and y
    X = np.ones((len(diff), 1))  # dummy feature since translation is constant

    ransac_x.fit(X, diff[:, 0])
    ransac_y.fit(X, diff[:, 1])

    abs_x = int(round(ransac_x.predict([[1]])[0]))
    abs_y = int(round(ransac_y.predict([[1]])[0]))
    # print("dx, dy: ", dx, dy)

    if not should_stitch(mask, good_matches, abs_x, abs_y, map_height, map_width, last_coords):
        return None, None

    return abs_x, abs_y

if __name__ == "__main__":

    base_runs = Path("../../../resources/detection_sets")
    video_base = Path("../../../resources/custom_test_img/forest_follies")
    output_base = Path("../../../resources/datasets/absolute")

    for run_dir in base_runs.glob("detect_forest_follies_5"):
        suffix = run_dir.name.removeprefix("detect_forest_follies_")

        process(
            run_dir / "track" / "labels",
            video_base / f"forest_follies_{suffix}.mp4",
            output_base / f"forest_follies_{suffix}.pkl",
        )
