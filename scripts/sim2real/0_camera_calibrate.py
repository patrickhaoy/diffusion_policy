import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from perception.multi_camera_wrapper import MultiCameraWrapper
from perception.pcd_utils import *


if __name__ == "__main__":
    # Number of calibration rounds
    NUM_CALIBRATION_ROUNDS = 10

    # gather cameras
    multi_camera_wrapper = MultiCameraWrapper(rgb=True, depth=True, ir=False, high_res_rgb=False, align="rgb", type="realsense")
    num_cameras = multi_camera_wrapper.num_cameras
    print(f"Number of cameras: {num_cameras}")

    # calibrate using aruco tag
    pcds = []
    all_calib_dicts = []
    
    for round_idx in range(NUM_CALIBRATION_ROUNDS):
        print(f"\nPerforming calibration round {round_idx + 1}/{NUM_CALIBRATION_ROUNDS}")
        round_calib_dict = []
        
        for camera in multi_camera_wrapper._all_cameras:
            intrinsics = camera.calibration["intrinsics"]["rgb"]["cameraMatrix"]
            print(f"Intrinsics:\n{intrinsics}")

            marker_size = 0.15
            tvec, rotmat = multi_camera_wrapper._get_aruco_pose(
                camera, marker_size=marker_size, verbose=True
            )
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = rotmat
            extrinsics[:3, 3:] = tvec
            print(f"Extrinsics:\n{extrinsics}")
            # camera frame -> aruco frame
            extrinsics_inv = np.linalg.inv(extrinsics)

            frames = camera.read_camera()
            rgb = frames["rgb"]
            depth = frames["depth"]

            # TODO: insert aruco offset to base
            aruco_offset = np.array(
                [
                    0.24,
                    0.0,
                    0.0,
                ]
            )

            extrinsics_inv[:3, 3] += aruco_offset

            round_calib_dict.append(
                {
                    "camera_serial_number": camera._serial_number,
                    "intrinsics_raw": intrinsics.tolist(),
                    "extrinsics_raw": extrinsics_inv.tolist(),
                    "intrinsics": {
                        "fx": intrinsics[0, 0],
                        "fy": intrinsics[1, 1],
                        "ppx": intrinsics[0, 2],
                        "ppy": intrinsics[1, 2],
                        "height": rgb.shape[0],
                        "width": rgb.shape[1],
                        "fovy": camera._fovy,
                        "coeffs": camera.calibration["intrinsics"]["rgb"][
                            "distCoeffs"
                        ].tolist(),
                    },
                    "camera_base_ori": extrinsics_inv[:3, :3].tolist(),
                    "camera_base_pos": extrinsics_inv[:3, 3:].tolist(),
                }
            )

            if round_idx == 0:  # Only collect point cloud data in first round
                points = depth_to_points(depth, intrinsics, extrinsics_inv, depth_scale=1000.0)
                colors = rgb.reshape(-1, 3) / 255.0
                points, colors = crop_points(points, colors=colors, crop_min=-2*np.ones(3), crop_max=2*np.ones(3))
                pcds.append(points_to_pcd(points, colors=colors))
        
        all_calib_dicts.append(round_calib_dict)

    # Average the calibration results
    final_calib_dict = []
    for camera_idx in range(num_cameras):
        # Collect all measurements for this camera
        camera_measurements = [round_dict[camera_idx] for round_dict in all_calib_dicts]
        
        # Average the extrinsics
        avg_extrinsics_raw = np.mean([np.array(m["extrinsics_raw"]) for m in camera_measurements], axis=0)
        avg_camera_base_ori = np.mean([np.array(m["camera_base_ori"]) for m in camera_measurements], axis=0)
        avg_camera_base_pos = np.mean([np.array(m["camera_base_pos"]) for m in camera_measurements], axis=0)
        
        # Use the first measurement for intrinsics (these shouldn't change)
        first_measurement = camera_measurements[0]
        
        final_calib_dict.append({
            "camera_serial_number": first_measurement["camera_serial_number"],
            "intrinsics_raw": first_measurement["intrinsics_raw"],
            "extrinsics_raw": avg_extrinsics_raw.tolist(),
            "intrinsics": first_measurement["intrinsics"],
            "camera_base_ori": avg_camera_base_ori.tolist(),
            "camera_base_pos": avg_camera_base_pos.tolist(),
        })

    current_time_date = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perception/calibrations/")
    os.makedirs(calib_path, exist_ok=True)
    json.dump(
        final_calib_dict,
        open(os.path.join(calib_path,f"{current_time_date}.json"), "w"),
    )
    json.dump(
        final_calib_dict,
        open(os.path.join(calib_path,f"most_recent_calib.json"), "w"),
    )
    print(f"Saved calibration at {os.path.join(calib_path,f'perception/logs/aruco/most_recent_calib.json')}")

    x = np.zeros((1, 3))
    for d in np.arange(0, 1, 0.1):
        x[:, 0] = d
        pcds.append(points_to_pcd(x, colors=[[255.0, 0.0, 0.0]]))
        y = np.zeros((1, 3))
        y[:, 1] = d
        pcds.append(points_to_pcd(y, colors=[[0.0, 255.0, 0.0]]))
        z = np.zeros((1, 3))
        z[:, 2] = d
        pcds.append(points_to_pcd(z, colors=[[0.0, 0.0, 255.0]]))

    visualize_pcds(pcds)
