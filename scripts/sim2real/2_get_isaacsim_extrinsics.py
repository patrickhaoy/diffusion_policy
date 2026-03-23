import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def extract_pos_quat_from_extrinsics(T):
    cam_base_pos = T[:3, 3]
    cam_base_ori = T[:3, :3]

    # Convert OpenCV → Isaac Sim (OpenGL) coordinates
    camera_axis_correction = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])

    corrected_rot = cam_base_ori @ camera_axis_correction
    quat_xyzw = R.from_matrix(corrected_rot).as_quat()  # [x, y, z, w]
    quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])  # reorder

    return cam_base_pos, quat_wxyz

def load_and_process_calibration(json_path):
    with open(json_path, "r") as f:
        calib_data = json.load(f)[0]

    intrinsics_raw = np.array(calib_data["intrinsics_raw"])
    width = calib_data["intrinsics"]["width"]
    height = calib_data["intrinsics"]["height"]
    extrinsics_raw = np.array(calib_data["extrinsics_raw"])

    pos, quat = extract_pos_quat_from_extrinsics(extrinsics_raw)

    print("Isaac Sim Camera Pose (OpenGL):")
    print("Position (x, y, z):", pos)
    print("Quaternion [w, x, y, z]:", quat)

if __name__ == "__main__":
    calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perception/calibrations/")
    load_and_process_calibration(os.path.join(calib_path, "most_recent_calib.json"))