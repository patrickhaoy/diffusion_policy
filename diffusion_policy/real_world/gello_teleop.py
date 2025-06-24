"""Module to control robot using Gello device through serial communication."""

import glob
import math
import time
import threading
from typing import Optional, List

import numpy as np

# Import the existing GelloAgent from gello_software
try:
    from gello.agents.gello_agent import GelloAgent as GelloSoftwareAgent
    GELLO_AVAILABLE = True
except ImportError:
    print("Warning: gello_software not available, using dummy mode")
    GELLO_AVAILABLE = False


class DummyGelloTeleopInterface:
    """A dummy version of GelloTeleopInterface that returns fixed joint positions for testing."""
    def __init__(self, port=None, start_joints=None):
        """
        Initialize with fixed joint positions.
        port and start_joints are ignored, they're just here for API compatibility.
        """
        # Initialize with a reasonable "home" position in radians (7 DOF for Gello)
        if start_joints is not None:
            self.fixed_joints = start_joints
        else:
            self.fixed_joints = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0, 0]
        
        # Concatenate joints with gripper value (1 for open, -1 for closed)
        self.latest_values = self.fixed_joints + [1]  # Gripper stays open
        print("Initialized DummyGelloTeleopInterface with fixed joint positions")
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


class GelloTeleopInterface:
    """Interface for controlling robot using Gello device through serial communication."""
    
    def __init__(self, port: Optional[str] = None, start_joints: Optional[List[float]] = None):
        """
        Initialize Gello teleoperation interface.
        
        Args:
            port: USB port path for Gello device. If None, will auto-detect.
            start_joints: Initial joint positions in radians. If None, uses default home position.
        """
        self.port = port
        self.start_joints = start_joints
        self.gello_agent = None
        self.running = True
        # Initialize previous values
        self.prev_joints = [0] * 6
        self.prev_gripper = 0
        
        # Initialize with default home position if not provided
        if self.start_joints is None:
            self.start_joints = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        
        # Convert to numpy array if it's a list
        if isinstance(self.start_joints, list):
            self.start_joints = np.array(self.start_joints)
        
        # Concatenate joints with gripper value (1 for open, -1 for closed)
        self.latest_values = self.start_joints.tolist() + [1]  # Start with gripper open
        
        self._setup_gello()
        self._start_read_thread()

    def _setup_gello(self):
        """Set up Gello agent connection."""
        try:
            if not GELLO_AVAILABLE:
                raise ImportError("Gello software not available")
            
            # Auto-detect port if not provided
            if self.port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} USB ports")
                if len(usb_ports) > 0:
                    self.port = usb_ports[0]
                    print(f"Using auto-detected port: {self.port}")
                else:
                    raise ValueError("No Gello port found, please specify one or plug in Gello")
            
            # Initialize Gello agent using the existing implementation
            self.gello_agent = GelloSoftwareAgent(port=self.port, start_joints=self.start_joints)
            print(f"Successfully connected to Gello at {self.port}")
            
        except Exception as e:
            print(f"Error setting up Gello connection: {e}")
            raise

    def _read_thread(self):
        """Background thread to continuously read from Gello device."""
        while self.running:
            try:
                if self.gello_agent is not None:
                    # Get current joint positions from Gello using the existing interface
                    current_joints = self.gello_agent._robot.get_joint_state()
                     
                    if current_joints is not None and len(current_joints) >= 6:
                        # Update joint positions (first 7 values)
                        joint_positions = current_joints[:6]
                        
                        # Get gripper state (assuming last value or separate method)
                        # For now, we'll use a default open state
                        gripper_state = -1 if current_joints[6] > 0 else 1
                        
                        # Update latest values
                        self.latest_values = joint_positions.tolist() + [gripper_state]
                        
            except Exception as e:
                print(f"Error reading from Gello: {e}")
            
            time.sleep(0.01)  # Small sleep to prevent busy waiting

    def _start_read_thread(self):
        """Start the background reading thread."""
        self.read_thread = threading.Thread(target=self._read_thread)
        self.read_thread.daemon = True
        self.read_thread.start()

    def get_latest_values(self):
        """Get the most recent joint and gripper values as concatenated array."""
        return self.latest_values

    def get_current_joints(self):
        """Get current joint positions from Gello."""
        if self.gello_agent is not None:
            return self.gello_agent._robot.get_joint_state()
        return self.start_joints

    def send_joint_command(self, joint_positions: List[float]):
        """Send joint position command to Gello."""
        if self.gello_agent is not None:
            try:
                # Ensure we have 7 DOF
                if len(joint_positions) == 7:
                    self.gello_agent._robot.command_joint_state(np.array(joint_positions))
                else:
                    print(f"Expected 7 joint positions, got {len(joint_positions)}")
            except Exception as e:
                print(f"Error sending joint command: {e}")

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.gello_agent is not None:
            try:
                # Close any open connections
                if hasattr(self.gello_agent, '_robot') and hasattr(self.gello_agent._robot, 'cleanup'):
                    self.gello_agent._robot.cleanup()
                print("Gello connection closed")
            except Exception as e:
                print(f"Error during cleanup: {e}")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Convenience function to create interface
def create_gello_interface(port: Optional[str] = None, 
                          start_joints: Optional[List[float]] = None,
                          dummy: bool = False):
    """
    Create a Gello teleoperation interface.
    
    Args:
        port: USB port path for Gello device
        start_joints: Initial joint positions in radians
        dummy: If True, creates a dummy interface for testing
    
    Returns:
        GelloTeleopInterface or DummyGelloTeleopInterface
    """
    if dummy:
        return DummyGelloTeleopInterface(port=port, start_joints=start_joints)
    else:
        return GelloTeleopInterface(port=port, start_joints=start_joints) 