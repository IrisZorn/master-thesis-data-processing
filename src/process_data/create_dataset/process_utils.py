import math
import os

import cv2
import numpy as np
from natsort import natsorted
from skimage.measure import label, regionprops


class VideoReader:
    def __init__(self, video_path, target_width=1920):
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.target_width = target_width  # normalize all frames to this width

    def get_frame(self, frame_number):
        if frame_number > self.total_frames:
            frame_number = self.total_frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return None

        h, w = frame.shape[:2]
        scale = self.target_width / w
        frame_resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        return frame_resized

    def release(self):
        self.cap.release()
def filter_map_points(kp, des, last_coords, frame_height, frame_width):
    x_last, y_last = last_coords
    lower_bound, upper_bound = x_last - frame_width/2, x_last + frame_width*1.5
    filtered_kp = []
    filtered_des = []
    for i, kp in enumerate(kp):
        x, y = kp.pt[0], kp.pt[1]
        if lower_bound < x < upper_bound:
            filtered_kp.append(kp)
            filtered_des.append(des[i])

    return filtered_kp, np.stack(filtered_des)


def load_keypoints(filename):
    data = np.load(filename)
    kp_array = data['keypoints']
    descriptors = data['descriptors']

    keypoints = [
        cv2.KeyPoint(
            kp[0],  # x
            kp[1],  # y
            kp[2],  # size
            kp[3],  # angle
            kp[4],  # response
            int(kp[5]),  # octave
            int(kp[6])  # class_id
        )
        for kp in kp_array
    ]

    return keypoints, descriptors


def is_scattered(mask, area_ratio_threshold=0.001, min_components=200):

    labels = label(mask)
    regions = regionprops(labels)

    if not regions:
        return False  # empty mask is not scattered

    # Normalize each component area by total mask area
    norm_areas = [r.area / mask.size for r in regions]

    # Count small components
    small_count = sum(a < area_ratio_threshold for a in norm_areas)
    #print("small area count: " + str(small_count))

    # Consider mask scattered if enough small components exist
    return small_count >= min_components


def coords_to_cv2(rel_x, rel_y, frame_width, frame_height):
    return int(float(rel_x) * frame_width), int(float(rel_y) * frame_height)


def should_stitch(mask, matches, dx, dy, map_height, map_width, last_coords):
    mask_h, mask_w = mask.shape[:2]

    if matches[0].distance > 65:
        print("skipped because of distances")
        return False

    if abs(dx) > map_width or abs(dy) > map_height:
        print("skipped because difference too large")
        return False

    if dx > last_coords[0] + mask_w / 2 or dx < last_coords[0] - mask_w / 2:
        print("skipped because distance to last coord too high")
        return False

    if last_coords == (0,0) and (abs(dx) > mask_w/4 or abs(dy) > map_height/2):
        print("skipped because beginning too far away")
        return False
    return True
import math

def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx*dx + dy*dy)
