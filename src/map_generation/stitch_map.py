import cv2
import numpy as np
from utils.ml_feature_extract import extract_motion_layer
from stitch_utils import get_frames, transform_keypoints, should_stitch, save_keypoints

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

frame_counter = 0
step = 9

num_matches = 50
error_counter = 0
error_step = 1
error_back_step = 6

frames = get_frames()
frame_height, frame_width = frames[0].shape[:2]

img1 = frames[frame_counter]
img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

frame_counter += step
img2 = frames[frame_counter]

mask = extract_motion_layer(frame1=img1, frame2=img2, beginning=True, model="../../RAFT/models/raft-things.pth")
kp1, des1 = sift.detectAndCompute(img1_grey, mask)
kp1 = list(kp1)
kp_safe, des_safe = kp1, des1
#sprite_mask_1 = create_enemy_mask(img1_grey, sprites)
merged_img = img1
merged_img_grey = img1_grey
prev_img_mask = mask
combined_canvas_mask = mask

stop = len(frames) - 11
beginning = True
while frame_counter < stop:
    print("i: " + str(frame_counter))
    img2 = frames[frame_counter]
    img3 = frames[frame_counter + step]
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #sprite_mask_2 = create_enemy_mask(img2_grey, sprites)
    beginning = True
    # if frame_counter < 350 or (frame_counter > 1430 and frame_counter < 1750):
    #     beginning = True

    mask2 = extract_motion_layer(frame1=img2, frame2=img3, beginning=beginning, model="../../RAFT/models/raft-things.pth")

    kp2, des2 = sift.detectAndCompute(img2_grey, mask2)
    kp2 = list(kp2)

    #Display the images with keypoints
    # image1_keypoints = cv2.drawKeypoints(merged_img, kp1, None)
    # all_kps = cv2.drawKeypoints(merged_img, kp_safe, None)
    # image2_keypoints = cv2.drawKeypoints(img2, kp2, None)
    # cv2.imshow('Image 1 with all Keypoints', all_kps)
    # cv2.imshow('Image 1 with Keypoints', image1_keypoints)
    # cv2.imshow('Image 2 with Keypoints', image2_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    matches = bf.match(des1, des2)
    good_matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

    # Draw the matches
    # matched_img = cv2.drawMatches(
    #     merged_img_grey, kp1,
    #     img2_grey, kp2,
    #     good_matches,
    #     None,
    #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    # )

    # cv2.imshow('Matches between Image 1 and Image 2', matched_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    #difference of image points to map points
    diff = pts1 - pts2
    dx, dy = np.rint(np.median(diff, axis=0)).astype(int)

    print([match.distance for match in good_matches])

    merged_height, merged_width = merged_img.shape[:2]

    #get image offsets to 0
    offset_x_merged = abs(min(0, dx))
    offset_y_merged = abs(min(0, dy))
    offset_x2 = max(0, dx + offset_x_merged)
    offset_y2 = max(0, dy + offset_y_merged)

    # distance from current merged canvas edge
    dist_x = offset_x2 - (merged_width + offset_x_merged)
    dist_y = offset_y2 - (merged_height + offset_y_merged)

    print("dx, dy: ", dist_x, dist_y)

    if not should_stitch(mask2, good_matches, dist_x, dist_y, frame_width, frame_height, frame_counter):
        # back step is initiated to stay in approximate distance of last correct frame
        if error_counter > 10:
            frame_counter = frame_counter - (error_counter * error_step + error_back_step)
            error_counter = 0
            print("back step")
        else:
            frame_counter += error_step
            error_counter += 1
        continue
    error_counter = 0

    # Determine size of the canvas
    max_x = max(merged_width, dx + frame_width)
    max_y = max(merged_height, dy + frame_height)
    canvas_width = max_x + offset_x_merged
    canvas_height = max_y + offset_y_merged

    # Create canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=merged_img.dtype)
    mask_canvas = np.zeros((canvas_height, canvas_width), dtype=merged_img.dtype)
    combined_mask_canvas = np.zeros((canvas_height, canvas_width), dtype=merged_img.dtype)

    # Alpha blending for img2
    alpha = 0.5  # blending factor

    # Blend merged_img into canvas
    canvas[offset_y_merged:offset_y_merged + merged_height, offset_x_merged:offset_x_merged + merged_width] = merged_img

    # Blend img2 into canvas
    canvas[offset_y2:offset_y2 + frame_height, offset_x2:offset_x2 + frame_width] = (canvas[offset_y2:offset_y2 + frame_height, offset_x2:offset_x2 + frame_width] * alpha +
                                                                                     ((1 - alpha) * img2).astype(merged_img.dtype))

    mask_canvas[offset_y2:offset_y2 + frame_height, offset_x2:offset_x2 + frame_width] = mask2

    combined_mask_canvas[offset_y_merged:offset_y_merged + merged_height, offset_x_merged:offset_x_merged + merged_width] = combined_canvas_mask
    combined_mask_canvas[offset_y2:offset_y2 + frame_height, offset_x2:offset_x2 + frame_width] = (combined_mask_canvas[offset_y2:offset_y2 + frame_height, offset_x2:offset_x2 + frame_width] |
                                                                                     mask2)

    kp_safe, des_safe = transform_keypoints(kp_safe, des_safe, offset_x_merged, offset_y_merged, None)
    kp2_safe, des2_safe = transform_keypoints(kp2, des2, offset_x2, offset_y2, None)

    kp_safe, des_safe = kp_safe + kp2_safe, np.vstack((des_safe, des2_safe))

    kp1, des1 = transform_keypoints(kp1, des1, offset_x_merged, offset_y_merged, mask_canvas)
    kp2, des2 = transform_keypoints(kp2, des2, offset_x2, offset_y2, mask_canvas)

    kp1, des1 = kp1 + kp2, np.vstack((des1, des2))

    merged_img = canvas
    merged_img_grey = cv2.cvtColor(merged_img, cv2.COLOR_BGR2GRAY)
    img1 = img2
    img1_grey = img2_grey

    mask = mask_canvas
    prev_img_mask = mask2
    frame_counter += step
    combined_canvas_mask = combined_mask_canvas

# Save result
cv2.imwrite("forest_follies_stitched_map.png", combined_canvas_mask)
cv2.imwrite("forest_follies_stitched_end.png", merged_img)

save_keypoints("merged_keypoints.npz", kp_safe, des_safe)

