import pybullet as p
import numpy as np
import os
import math


class LeggedRobot:
    """
    A multi-legged robot model implemented in PyBullet using URDF.
    This is a port of the original ODE robot model from the C++ code.
    """
    
    def __init__(self, client=None, urdf_path="src/robot/urdf/legged_robot.urdf"):
        """
        Initialize the robot with default parameters.
        
        Args:
            client: PyBullet physics client ID
            urdf_path: Path to the URDF file
        """
        # Store physics client
        self.client = client if client is not None else p.connect(p.DIRECT)
        
        # Robot parameters (kept for reference and for control)
        self.box_pos = [0.0, 0.0, 0.5]
        self.box_length = 1.0
        self.box_width = 0.4
        self.box_height = 0.2
        
        # Robot leg parameters
        self.leg_count = 6
        self.total_legs = 18
        self.dummy_legs = 6
        self.dof = 3  # degrees of freedom per leg
        
        # Min and max joint angles (in degrees, will be converted to radians)
        self.q_min = [-45, 0, 0]
        self.q_range = [90, 60, 60]
        self.q_init = [0, 45, 45]
        
        # Current and target joint angles
        self.q_angle = np.zeros((self.leg_count, self.dof))
        self.t_angle = np.zeros((self.leg_count, self.dof))
        
        # Load URDF
        self._load_urdf(urdf_path)
        
        # Store joint information
        self._get_joint_info()
    
    def _load_urdf(self, urdf_path):
        """
        Load the robot from URDF.
        
        Args:
            urdf_path: Path to the URDF file
        """
        # Check if URDF file exists
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        # Load URDF
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=self.box_pos,
            useFixedBase=False
        )
        
        # Enable self-collision
        p.setCollisionFilterGroupMask(self.robot_id, -1, 0, 0)
        
        # Get number of joints
        self.num_joints = p.getNumJoints(self.robot_id)
        
        print(f"Loaded robot from {urdf_path} with {self.num_joints} joints")
    
    def _get_joint_info(self):
        """Get information about all joints."""
        # Initialize arrays for joint information
        self.joint_indices = []
        self.joint_names = []
        self.joint_types = []
        
        # Map from leg index to joint indices
        self.leg_joints = {}
        
        # Get information about each joint
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Store joint information
            self.joint_indices.append(i)
            self.joint_names.append(joint_name)
            self.joint_types.append(joint_type)
            
            # Map joints to legs based on naming convention from URDF
            if joint_name.startswith('joint_'):
                # Extract leg index from joint name
                try:
                    leg_index = int(joint_name[6:])
                    leg_group = leg_index // 3
                    leg_segment = leg_index % 3
                    
                    # Initialize the leg group if not already
                    if leg_group not in self.leg_joints:
                        self.leg_joints[leg_group] = []
                    
                    # Store joint index with its segment position
                    self.leg_joints[leg_group].append((i, leg_segment))
                except ValueError:
                    # Not a leg joint
                    pass
        
        # Sort joint indices within each leg by segment
        for leg_group in self.leg_joints:
            self.leg_joints[leg_group] = sorted(self.leg_joints[leg_group], key=lambda x: x[1])
            # Keep only joint indices
            self.leg_joints[leg_group] = [x[0] for x in self.leg_joints[leg_group]]
    
    def reset_posture(self):
        """Reset the robot to its initial posture."""
        for i in range(self.leg_count):
            for j in range(self.dof):
                if i < 3:  # Right side legs
                    self.t_angle[i][j] = -math.radians(self.q_init[j])
                else:      # Left side legs
                    self.t_angle[i][j] = math.radians(self.q_init[j])
        
        self.apply_target_angles()
    
    def apply_target_angles(self):
        """Apply the current target angles to the robot joints."""
        gain = 5.0  # Control gain
        
        # Apply to each leg
        for leg_group in range(self.leg_count):
            if leg_group in self.leg_joints:
                joints = self.leg_joints[leg_group]
                
                # Apply to each joint in this leg
                for j, joint_index in enumerate(joints):
                    if j < self.dof:
                        # Get current joint angle
                        joint_state = p.getJointState(self.robot_id, joint_index)
                        current_angle = joint_state[0]
                        self.q_angle[leg_group][j] = current_angle
                        
                        # Calculate velocity based on error
                        target_angle = self.t_angle[leg_group][j]
                        velocity = gain * (target_angle - current_angle)
                        
                        # Apply velocity to joint
                        p.setJointMotorControl2(
                            bodyUniqueId=self.robot_id,
                            jointIndex=joint_index,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=velocity,
                            force=20.0
                        )
    
    def set_target_angles(self, angles):
        """
        Set target angles for all joints.
        
        Args:
            angles: Array of shape (leg_count, dof) with target angles
        """
        self.t_angle = np.array(angles)
    
    def get_position(self):
        """Get current position of the robot body."""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return pos
    
    def get_orientation(self):
        """Get current orientation of the robot body."""
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        return orn
    
    def get_state(self):
        """Get complete state of the robot (position, orientation, joint angles)."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        
        state = {
            'position': pos,
            'orientation': orn,
            'rotation_matrix': rot_matrix,
            'joint_angles': self.q_angle.copy()
        }
        return state
    
    @property
    def body_id(self):
        """Return the robot ID for compatibility."""
        return self.robot_id