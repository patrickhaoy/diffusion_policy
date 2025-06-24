#!/usr/bin/env python3
import time
import sys
import math
from diffusion_policy.real_world.gello_teleop import GelloTeleopInterface, DummyGelloTeleopInterface

def test_gello_interface(use_dummy=False, port=None):
    """
    Test the GelloTeleopInterface by continuously reading and displaying data.
    
    Args:
        use_dummy: If True, use DummyGelloTeleopInterface for testing without hardware
        port: USB port path for Gello device (only used if not dummy)
    """
    try:
        if use_dummy:
            print("Using DummyGelloTeleopInterface for testing...")
            with DummyGelloTeleopInterface() as gello:
                print("Starting to read dummy data. Press Ctrl+C to stop.")
                while True:
                    values = gello.get_latest_values()
                    joints = values[:-1]  # First 7 values are joints
                    gripper = values[-1]  # Last value is gripper
                    
                    print(f"Joints (rad): {[f'{j:.4f}' for j in joints]}")
                    print(f"Joints (deg): {[f'{math.degrees(j):.2f}' for j in joints]}")
                    print(f"Gripper: {'Closed' if gripper < 0 else 'Open'}")
                    print("-" * 50)
                    
                    time.sleep(0.1)  # Update every 100ms
        else:
            print("Using GelloTeleopInterface with real hardware...")
            with GelloTeleopInterface(port=port) as gello:
                print("Starting to read Gello data. Press Ctrl+C to stop.")
                while True:
                    values = gello.get_latest_values()
                    joints = values[:-1]  # First 7 values are joints
                    gripper = values[-1]  # Last value is gripper
                    
                    print(f"Joints (rad): {[f'{j:.4f}' for j in joints]}")
                    print(f"Joints (deg): {[f'{math.degrees(j):.2f}' for j in joints]}")
                    print(f"Gripper: {'Closed' if gripper < 0 else 'Open'}")
                    print("-" * 50)
                    
                    time.sleep(0.1)  # Update every 100ms
                    
    except KeyboardInterrupt:
        print("\nStopping Gello interface...")
    except Exception as e:
        print(f"Error reading Gello data: {e}")

def main():
    # Check command line arguments
    use_dummy = False
    port = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--dummy':
            use_dummy = True
        elif sys.argv[i] == '--port' and i + 1 < len(sys.argv):
            port = sys.argv[i + 1]
            i += 1
        elif sys.argv[i] == '--help':
            print("Usage: python test_gello.py [options]")
            print("Options:")
            print("  --dummy     Use dummy interface (no hardware required)")
            print("  --port PORT Specify USB port for Gello device")
            print("  --help      Show this help message")
            return
        i += 1
    
    try:
        # Give the device time to initialize
        if not use_dummy:
            print("Initializing Gello device...")
            time.sleep(2)
        
        # Start reading data
        test_gello_interface(use_dummy=use_dummy, port=port)
        
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 