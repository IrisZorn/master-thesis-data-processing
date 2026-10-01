from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

from RAFT import RAFT
from RAFT.core.utils.utils import InputPadder
from src.map_generation.stitch_utils import get_frames

model_path = Path(__file__).parent / "../../RAFT/models/raft-things.pth"
model_path = model_path.resolve()  # absolute path
args = Namespace(
        model=model_path,  # path to your checkpoint
        small=False,  # use small model or not
        mixed_precision=False,  # or True, depending on how the model was saved
        alternate_corr=False  # whether to use alternate (memory-efficient) correlation
    )
model = RAFT(args)
device = torch.device("cuda")
# Wrap in DataParallel to match how the checkpoint was saved:
model = torch.nn.DataParallel(model)
state = torch.load(args.model, map_location=device)  # or "cpu" depending on your device
model.load_state_dict(state)

model = model.to(device)
kernel = np.ones((5, 5), np.uint8)

def flow_to_image(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    mag = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def extract_motion_layer(frame1, frame2, beginning, downscale_factor=0.5):

    h, w, _ = frame1.shape

    # --- Downscale frames for RAFT ---
    small_frame1 = cv2.resize(frame1, (int(w * downscale_factor), int(h * downscale_factor)))
    small_frame2 = cv2.resize(frame2, (int(w * downscale_factor), int(h * downscale_factor)))

    # Convert frames to GPU tensors efficiently
    current_frame = torch.as_tensor(small_frame1).permute(2, 0, 1).float().unsqueeze(0).to(device)
    next_frame = torch.as_tensor(small_frame2).permute(2, 0, 1).float().unsqueeze(0).to(device)

    padder = InputPadder(current_frame.shape)
    prev_frame_padded, next_frame_padded = padder.pad(current_frame, next_frame)

    with torch.inference_mode():
        flow_low, flow_up = model(prev_frame_padded, next_frame_padded, iters=12, test_mode=True)
    flow_up = padder.unpad(flow_up)
    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()

    # flow_image = flow_to_image(flow)
    # cv2.imshow('Optical Flow', flow_image)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    u = flow[...,0] #horizontal motion of each pixel
    v = flow[...,1] #vertical motion

    mag = np.hypot(u, v)
    weights = 1.0 / (mag + 1e-6)
    global_u = np.average(u, weights=weights)
    global_v = np.average(v, weights=weights)


    # Compute residual motion
    resid_u = u - global_u
    resid_v = v - global_v
    resid_mag = np.hypot(resid_u, resid_v)

    # Mask pixels with small residual motion (moving with scroll)
    if beginning:
        lower_q = 45
        upper_q = 90
    else:
        lower_q = 50
        upper_q = 74
    thr_bg = np.percentile(resid_mag, lower_q)
    thr_fg = np.percentile(resid_mag, upper_q)
    mask = ((resid_mag < thr_bg) | (resid_mag > thr_fg)).astype(np.uint8) * 255

    # --- Morphology on downscaled mask for speed ---
    if downscale_factor < 1.0:
        small_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        small_mask = cv2.morphologyEx(small_mask, cv2.MORPH_OPEN, kernel)
        full_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        full_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return cv2.bitwise_not(full_mask)


if __name__ == "__main__":
    frames = get_frames()
    h,w = frames[10].shape[:2]
    for i in range(0, len(frames)-2):
        beginning = True
        # if i < 350 or (i > 1430 and i < 1565):
        #     beginning = True
        resized_1 = cv2.resize(frames[i],(int(w*0.4), int(h*0.4)))
        resized_2 = cv2.resize(frames[i+1], (int(w*0.4), int(h*0.4)))
        mask = extract_motion_layer(resized_1, resized_2, beginning, 0.4)
        img = cv2.addWeighted(mask, 0.5, cv2.cvtColor(resized_1, cv2.COLOR_BGR2GRAY), 1 - 0.5, 0)
        cv2.imshow("masks", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit early
            break
    cv2.destroyAllWindows()