#!/usr/bin/env python3
"""
Demo script for teleoperation without cameras using RealEnv.
This script shows how to use the RealEnv with use_cameras=False for pure robot control.
"""

import time
import numpy as np
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.gello_teleop import create_gello_interface

def main():
    # Configuration
    output_dir = "./teleop_data"
    robot_ip = "192.168.1.100"  # Replace with your robot IP
    frequency = 10  # Control frequency in Hz
    
    # Create RealEnv with cameras disabled
    env = RealEnv(
        output_dir=output_dir,
        robot_ip=robot_ip,
        frequency=frequency,
        use_cameras=False,  # Disable cameras for pure teleoperation
        # Robot parameters
        max_pos_speed=0.25,
        max_rot_speed=0.6,
        tcp_offset=0.13,
        # Gripper parameters
        gripper_ip="192.168.1.2",
        gripper_port=63352,
    )
    
    try:
        # Start the environment
        print("Starting RealEnv...")
        env.start()
        
        # Wait for environment to be ready
        while not env.is_ready:
            print("Waiting for environment to be ready...")
            time.sleep(0.1)
        
        print("Environment ready!")
        
        # Optional: Create Gello teleop interface for manual control
        # gello = create_gello_interface(dummy=True)  # Use dummy=True for testing
        
        # Start recording an episode
        print("Starting episode...")
        env.start_episode()
        
        # Main control loop
        print("Entering control loop. Press Ctrl+C to stop.")
        step = 0
        while True:
            # Get current observation (robot state only, no camera data)
            obs = env.get_obs()
            
            # Print robot state
            if step % 10 == 0:  # Print every 10 steps
                print(f"Step {step}:")
                print(f"  Joint positions: {obs['arm_joint_pos'][-1]}")
                print(f"  End effector pose: {obs['end_effector_pose'][-1]}")
                print(f"  Timestamp: {obs['timestamp'][-1]:.3f}")
            
            # Example: Generate a simple action (move joints slightly)
            # In practice, you would get this from your teleop interface
            current_joints = obs['arm_joint_pos'][-1]
            
            # Create a small sinusoidal movement
            t = time.time()
            action_joints = current_joints + 0.01 * np.sin(t + np.arange(6))
            
            # Add gripper action (0 = closed, 1 = open)
            gripper_action = 1.0  # Keep gripper open
            action = np.concatenate([action_joints, [gripper_action]])
            
            # Create timestamps for the action
            timestamps = obs['timestamp'][-1] + np.array([0.1, 0.2, 0.3])  # 3-step action
            
            # Execute the action
            env.exec_actions(
                actions=np.tile(action, (3, 1)),  # Repeat action for 3 steps
                timestamps=timestamps
            )
            
            step += 1
            time.sleep(1.0 / frequency)  # Maintain control frequency
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # End the episode and stop the environment
        env.end_episode()
        env.stop()
        print("Environment stopped.")

if __name__ == "__main__":
    main() 