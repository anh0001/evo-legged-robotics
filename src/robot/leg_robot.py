import pybullet as p
import numpy as np
import os
import math


class LeggedRobot:
    """
    A multi-legged robot model that accurately matches the C++ ODE implementation.
    This includes 6 dummy legs (fixed) and 18 active leg segments (3 per leg).
    """
    
    def __init__(self, client=None, urdf_path="src/robot/urdf/legged_robot.urdf"):
        """
        Initialize the robot with parameters matching the C++ code.
        
        Args:
            client: PyBullet physics client ID
            urdf_path: Path to the URDF file
        """
        self.client = client if client is not None else p.connect(p.DIRECT)
        
        # Robot parameters matching C++ code exactly
        self.box_pos = [0.0, 0.0, 0.5]
        self.box_length = 1.0
        self.box_width = 0.4
        self.box_height = 0.2
        self.box_mass = 1.0
        
        # Leg parameters
        self.bar_length = 0.1
        self.bar_width = 0.2
        self.bar_height = 0.1
        self.bar_mass = 0.05
        self.bar_rest = 0.04
        
        # Structure matching C++ code
        self.dof = 3          # DOF per leg
        self.leg = 6          # Number of legs
        self.tleg = 18        # Total leg segments (TLEG)
        self.dleg = 6         # Dummy legs (DLEG)
        
        # Joint limits (in degrees, converted to radians when used)
        self.q_min = np.array([-45, 0, 0])
        self.q_range = np.array([90, 60, 60])
        self.q_init = np.array([0, 45, 45])
        
        # Current and target angles for 6 legs x 3 DOF
        self.qang = np.zeros((self.leg, self.dof))
        self.tang = np.zeros((self.leg, self.dof))
        
        # Velocity array for all 18 joints
        self.vel = np.zeros(self.tleg)
        
        # Control parameters
        self.gain = 5.0
        self.posz = 1  # Normal: 1, Overturn: -1
        
        # Load URDF
        self._load_urdf(urdf_path)
        
        # Map joints properly
        self._map_joints()
    
    def _load_urdf(self, urdf_path):
        """Load the robot from URDF."""
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        self.body_id = p.loadURDF(
            urdf_path,
            basePosition=self.box_pos,
            useFixedBase=False
        )
        
        self.num_joints = p.getNumJoints(self.body_id)
        print(f"Loaded robot with {self.num_joints} joints")
    
    def _map_joints(self):
        """
        Map joints to match C++ structure:
        - joint2[0-5]: dummy leg joints (fixed)
        - joint[0-17]: active leg joints
        """
        self.joint2 = []  # Dummy leg joints
        self.joint = []   # Active leg joints
        
        # Get all joint info
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.body_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Map based on joint names from URDF
            if joint_name.startswith('body_to_dummy_'):
                # Dummy leg joint (fixed)
                dummy_idx = int(joint_name.split('_')[-1])
                self.joint2.append(i)
            elif joint_name.startswith('joint_'):
                # Active leg joint
                joint_idx = int(joint_name.split('_')[-1])
                self.joint.append(i)
        
        print(f"Mapped {len(self.joint2)} dummy joints and {len(self.joint)} active joints")
        
        # Create mapping from leg index to joint indices
        self.leg_to_joints = {}
        for leg_idx in range(self.leg):
            # Each leg has 3 joints
            start_idx = leg_idx * 3
            self.leg_to_joints[leg_idx] = [
                self.joint[start_idx],
                self.joint[start_idx + 1],
                self.joint[start_idx + 2]
            ]
    
    def reset_posture(self):
        """Reset robot to initial posture matching C++ implementation."""
        # Set initial target angles
        for i in range(self.leg):
            for j in range(self.dof):
                if i < 3:  # Right side legs
                    self.tang[i][j] = -np.radians(self.q_init[j])
                else:      # Left side legs
                    self.tang[i][j] = np.radians(self.q_init[j])
        
        # Apply target angles
        self.apply_target_angles()
    
    def apply_target_angles(self):
        """
        Apply target angles using velocity control matching C++ implementation.
        This is called every 2 simulation steps in the main loop.
        """
        # Calculate velocities for all joints
        for i in range(self.leg):
            for j in range(self.dof):
                # Get joint index
                k = i * 3 + j  # Linear index for velocity array
                joint_idx = self.leg_to_joints[i][j]
                
                # Get current angle
                joint_state = p.getJointState(self.body_id, joint_idx)
                self.qang[i][j] = joint_state[0]
                
                # Calculate velocity
                self.vel[k] = self.gain * (self.tang[i][j] - self.qang[i][j])
        
        # Apply velocities to all joints
        for i in range(self.leg):
            for j in range(self.dof):
                k = i * 3 + j
                joint_idx = self.leg_to_joints[i][j]
                
                p.setJointMotorControl2(
                    bodyUniqueId=self.body_id,
                    jointIndex=joint_idx,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=self.vel[k],
                    force=20.0  # Matches dParamFMax from C++
                )
    
    def set_target_angles(self, angles):
        """
        Set target angles for all joints.
        
        Args:
            angles: Array of shape (leg, dof) with target angles in radians
        """
        # Apply posz for handling robot orientation
        self.tang = angles * self.posz
    
    def update_orientation(self):
        """Update posz based on robot orientation (upright vs flipped)."""
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        # Check if robot is upside down
        if rot_matrix[2, 2] < -0.7:  # z-component of z-axis
            self.posz = -1
        else:
            self.posz = 1
    
    def get_position(self):
        """Get current position of robot body."""
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos
    
    def get_orientation(self):
        """Get current orientation of robot body."""
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        return orn
    
    def get_state(self):
        """Get complete state of the robot."""
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        
        # Get leg positions for all active leg end segments
        leg_positions = []
        for i in range(self.leg):
            # Get position of the last segment of each leg
            last_joint_idx = self.leg_to_joints[i][2]
            link_state = p.getLinkState(self.body_id, last_joint_idx)
            leg_positions.append(link_state[0])  # World position
        
        state = {
            'position': pos,
            'orientation': orn,
            'rotation_matrix': rot_matrix,
            'joint_angles': self.qang.copy(),
            'leg_positions': leg_positions
        }
        return state