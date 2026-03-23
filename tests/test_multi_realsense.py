import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import json
import time
import numpy as np
from diffusion_policy.real_world.multi_realsense import MultiRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder

def test():
    configs = [
        json.load(open("diffusion_policy/real_world/realsense_config/455_front.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/435_side.json")),
        json.load(open("diffusion_policy/real_world/realsense_config/415_wrist.json"))
    ]

    def transform(data):
        color = data['color']
        h,w,_ = color.shape
        factor = 4
        color = cv2.resize(color, (w//factor,h//factor), interpolation=cv2.INTER_AREA)
        color = color[:,140:500]
        data['color'] = color
        return data

    from diffusion_policy.common.cv2_util import get_image_transform
    color_transform = get_image_transform(
        input_res=(640,480),
        output_res=(224,224), 
        bgr_to_rgb=False)
    def transform(data):
        data['color'] = color_transform(data['color'])
        return data

    # one thread per camera
    video_recorder = VideoRecorder.create_h264(
        fps=30,
        codec='h264',
        thread_type='FRAME'
    )

    with MultiRealsense(
        resolution=(640,480),
        capture_fps=30,
        record_fps=15,
        enable_color=True,
        serial_numbers=['215122255213', '832112070487', '746112060198'],
        advanced_mode_config=configs,
        transform=transform,
        # recording_transform=transform,
        # video_recorder=video_recorder,
        verbose=True
        ) as realsense:
        intr = realsense.get_intrinsics()
        print(intr)

        video_path = 'data/test'
        rec_start_time = time.time() + 1
        realsense.start_recording(video_path, start_time=rec_start_time)
        realsense.restart_put(rec_start_time)

        out = None
        vis_img = None
        open_windows = set()
        while True:
            out = realsense.get(out=out)

            # Visualize each camera image in 'out' using fixed window names
            current_windows = set()
            for cam_idx, cam_data in out.items():
                img = cam_data['color']
                win_name = f'Camera {cam_idx}'
                cv2.imshow(win_name, img)
                current_windows.add(win_name)
            # Close windows for cameras that are no longer present
            for win_name in open_windows - current_windows:
                cv2.destroyWindow(win_name)
            open_windows = current_windows

            # Wait for key press; close on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test()
