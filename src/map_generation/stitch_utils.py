import glob
import os

import cv2
import numpy as np
from natsort import natsorted
from skimage.measure import label, regionprops


def create_enemy_mask(frame, sprites):
    # Initialize SIFT detector

    sprite_mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
    overlay_layer = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
    for sprite, path, alpha_mask in sprites:
        res = cv2.matchTemplate(cv2.GaussianBlur(frame, (3, 3), 0), cv2.GaussianBlur(sprite, (3, 3), 0), cv2.TM_CCOEFF_NORMED, mask=alpha_mask)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= 0.65:

            top_left = max_loc
            h, w = sprite.shape

            sprite_mask_single = np.zeros((h, w), dtype=np.uint8)
            sprite_mask[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w] = cv2.bitwise_and(
                sprite_mask[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w],
                sprite_mask_single
            )

            x, y = top_left
            bottom_right = (top_left[0] + w, top_left[1] + h)
            overlay_layer[y:y + h, x:x + w, :] = cv2.cvtColor(sprite, cv2.COLOR_GRAY2BGRA)

            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(overlay_layer, top_left, bottom_right, color, thickness)

            accuracy = max_val * 100  # convert to percentage
            text = f"{accuracy:.2f}%"

            # Position text just to the right of the rectangle
            text_pos = (bottom_right[0] + 10, top_left[1] + 20)

            # Draw text
            cv2.putText(
                overlay_layer,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # font scale
                (0, 255, 0),  # text color
                2,  # thickness
                cv2.LINE_AA
            )

    cv2.imshow("Enemy Mask", cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA), 0.5, overlay_layer, 1 - 0.5, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return sprite_mask

def load_sprites():
    sprite_folder = "../../resources/custom_test_img/forest_follies/sprites_*/**/*.png"
    sprite_folder = "../../resources/custom_test_img/forest_follies/sprites_cuphead/**/*.png"

    image_paths = glob.glob(sprite_folder, recursive=True)
    sprites = []

    for path in image_paths:
        sprite = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        sprite = cv2.resize(sprite, (int(sprite.shape[1] * 0.9), int(sprite.shape[0] * 0.9)))
        b, g, r, a = cv2.split(sprite)
        _, alpha_mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY)
        # Normalize alpha to 0..1
        alpha = a.astype(float) / 255.0

        # Create black background (same shape as a channel)
        bg = np.zeros_like(a, dtype=float)

        # Blend each channel with black background
        b = (b.astype(float) * alpha + bg * (1 - alpha)).astype(np.uint8)
        g = (g.astype(float) * alpha + bg * (1 - alpha)).astype(np.uint8)
        r = (r.astype(float) * alpha + bg * (1 - alpha)).astype(np.uint8)

        # Merge back into 3-channel BGR
        sprite = cv2.merge([b, g, r])
        sprite = cv2.cvtColor(sprite, cv2.COLOR_BGR2GRAY)

        sprites.append((sprite, path, alpha_mask))
        # sprites.append((cv2.flip(sprite, 1), path))
    return sprites

def get_frames(from_files=False):
    frames = []
    if from_files:
        folder_path = "../../resources/custom_test_img/forest_follies/frames"
        frame_paths = natsorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)])
        for path in frame_paths:
            frames.append(cv2.imread(path))
        return frames

    video_path = "../../resources/custom_test_img/forest_follies/forest_follies_neikou.mp4"
    video = cv2.VideoCapture(video_path)
    _, frame = video.read()

    frames.append(frame)

    while True:
        ret, frame = video.read()
        frames.append(frame)
        if not ret:
            break
    return frames


def transform_keypoints(kp, des, dx, dy, mask):
    """
    Shift keypoints by (dx, dy) for pure translation.
    """
    filtered_kp = []
    filtered_des = []

    for i, kp in enumerate(kp):
        # Transform keypoint
        x_new = kp.pt[0] + dx
        y_new = kp.pt[1] + dy

        if mask is None:
            new_kp = cv2.KeyPoint(x_new, y_new,
                                  size=kp.size,
                                  angle=kp.angle,
                                  response=kp.response,
                                  octave=kp.octave,
                                  class_id=kp.class_id)
            filtered_kp.append(new_kp)
            filtered_des.append(des[i])

        # Check if inside mask bounds
        else:
            if 0 <= int(x_new) < mask.shape[1] and 0 <= int(y_new) < mask.shape[0]:
                if mask[int(y_new), int(x_new)] > 0:  # non-zero mask
                    # Keep keypoint
                    new_kp = cv2.KeyPoint(x_new, y_new,
                                          size=kp.size,
                                          angle=kp.angle,
                                          response=kp.response,
                                          octave=kp.octave,
                                          class_id=kp.class_id)
                    filtered_kp.append(new_kp)
                    filtered_des.append(des[i])

    if filtered_des:
        filtered_des = np.stack(filtered_des)
    else:
        # Return empty array with correct descriptor shape
        filtered_des = np.array([], dtype=des.dtype).reshape(0, des.shape[1])

    return filtered_kp, filtered_des


def is_scattered(mask, size_threshold=600):
    labels = label(mask)
    areas = [r.area for r in regionprops(labels)]
    print(np.mean(areas))
    return (np.mean(areas) < size_threshold)


def should_stitch(mask, matches, dx, dy, frame_width, frame_height, i):
    if is_scattered(mask):
        print("skipped because scattered")
        return False

    if matches[0].distance > 65:
        print("skipped because of distances")
        return False

    if abs(dx) > frame_width*1.5 or abs(dy) > frame_height*1.5:
        print("skipped because difference too large")
        return False

    if i < 300 and abs(dy) > 15:
        print("skipped because translation in y too large")
        return False

    return True


def save_keypoints(filename, keypoints, descriptors):
    # Convert keypoints to a NumPy-friendly format
    kp_array = np.array([(
        kp.pt[0], kp.pt[1], kp.size, kp.angle,
        kp.response, kp.octave, kp.class_id
    ) for kp in keypoints], dtype=np.float32)

    # Save keypoints and descriptors
    np.savez(filename, keypoints=kp_array, descriptors=descriptors)


