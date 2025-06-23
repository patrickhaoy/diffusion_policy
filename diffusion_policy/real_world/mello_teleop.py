"""Module to control robot using Mello device through serial communication."""

import serial
import threading
import ast
import math
import time

class DummyMelloTeleopInterface:
    """A dummy version of MelloTeleopInterface that returns fixed joint positions for testing."""
    def __init__(self, port=None, baudrate=None):
        """
        Initialize with fixed joint positions.
        port and baudrate are ignored, they're just here for API compatibility.
        """
        # Initialize with a reasonable "home" position in radians
        self.fixed_joints = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]
        # Concatenate joints with gripper value (1 for open, -1 for closed)
        self.latest_values = self.fixed_joints + [1]  # Gripper stays open
        print("Initialized DummyMelloTeleopInterface with fixed joint positions")
        print(f"Fixed joints (rad): {self.fixed_joints}")
        print(f"Fixed joints (deg): {[math.degrees(j) for j in self.fixed_joints]}")
        print(f"Latest values (joints + gripper): {self.latest_values}")

    def get_latest_values(self):
        """Get the fixed joint positions and gripper state as concatenated array."""
        return self.latest_values

    def cleanup(self):
        """Dummy cleanup method for API compatibility."""
        pass

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

class MelloTeleopInterface:
    def __init__(self, port='/dev/serial/by-id/usb-M5Stack_Technology_Co.__Ltd_M5Stack_UiFlow_2.0_24587ce945900000-if00', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self._setup_serial()
        # Initialize previous values
        self.prev_joints = [0] * 6
        self.prev_gripper = 0
        # Concatenate joints with gripper value (1 for open, -1 for closed)
        self.latest_values = self.prev_joints + [1]  # Start with gripper open
        self.running = True
        self._start_read_thread()

    def _setup_serial(self):
        """Set up serial connection to Mello device."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1  # 1 second timeout
            )
            print(f"Successfully connected to {self.port}")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            raise

    def _degrees_to_radians(self, degrees):
        """Convert a list of angles from degrees to radians."""
        return [math.radians(deg) for deg in degrees]

    def _read_thread(self):
        """Background thread to continuously read from serial port."""
        while self.running:
            try:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8').strip()
                    try:
                        data = ast.literal_eval(line)
                        joint_positions = data.get('joint_positions:', [0]*7)
                        
                        # Take first 6 values for joints (in degrees) and convert to radians
                        joints_deg = joint_positions[:6]
                        # Negate joint 2 (index 2)
                        joints_deg[2] = -1 * joints_deg[2]
                        joints_rad = self._degrees_to_radians(joints_deg)
                        
                        # Get gripper value (last value)
                        gripper_value = joint_positions[-1] if len(joint_positions) > 6 else self.prev_gripper
                        
                        # Update values if joints are not all zeros
                        if not all(j == 0 for j in joints_rad):
                            self.prev_joints = joints_rad
                        self.prev_gripper = gripper_value
                        
                        # Convert gripper value to 1 (open) or -1 (closed)
                        gripper_state = 1 if gripper_value >= 0 else -1
                        self.latest_values = self.prev_joints + [gripper_state]
                        
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing serial data: {e}")
            except Exception as e:
                print(f"Error reading serial data: {e}")
            time.sleep(0.01)  # Small sleep to prevent busy waiting

    def _start_read_thread(self):
        """Start the background reading thread."""
        self.read_thread = threading.Thread(target=self._read_thread)
        self.read_thread.daemon = True
        self.read_thread.start()

    def get_latest_values(self):
        """Get the most recent joint and gripper values as concatenated array."""
        return self.latest_values

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Serial port closed")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 