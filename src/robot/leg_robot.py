import pybullet as p
import numpy as np
import os
import math


class LeggedRobot:
    """
    Enhanced legged robot with improved motor control to prevent vibrations.
    Uses POSITION_CONTROL with proper PD gains instead of VELOCITY_CONTROL.
    """
    
    def __init__(self, client=None, urdf_path="src/robot/urdf/legged_robot.urdf"):
        """Initialize the robot with enhanced motor control parameters."""
        self.client = client if client is not None else p.connect(p.DIRECT)
        
        # Robot parameters matching C++ code exactly
        self.box_pos = [0.0, 0.0, 0.12]
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
        self.leg_count = 6    # Number of legs
        self.tleg = 18        # Total leg segments (TLEG)
        self.dleg = 6         # Dummy legs (DLEG)
        
        # Joint limits (in degrees, converted to radians when used)
        self.q_min = np.array([-45, 0, 0])
        self.q_range = np.array([90, 60, 60])
        self.q_init = np.array([0, 45, 45])
        
        # Current and target angles for 6 legs x 3 DOF
        self.qang = np.zeros((self.leg_count, self.dof))
        self.tang = np.zeros((self.leg_count, self.dof))
        
        # FIXED: Better motor control parameters to prevent oscillations
        self.kp = 8.0         # Increased from 10.0 for better response
        self.kd = 1.5         # Increased from 0.5 for better damping
        self.max_force = 15.0 # Increased from 20.0 but still conservative
        
        # Control parameters
        self.posz = 1  # Normal: 1, Overturn: -1
        
        # Load URDF
        self._load_urdf(urdf_path)
        
        # Map joints properly
        self._map_joints()
        
        # Apply enhanced dynamics to robot
        self._configure_robot_dynamics()
    
    def _load_urdf(self, urdf_path):
        """Load the robot from URDF."""
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        self.body_id = p.loadURDF(
            urdf_path,
            basePosition=self.box_pos,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION  # Enable self-collision detection
        )
        
        self.num_joints = p.getNumJoints(self.body_id)
        print(f"Loaded robot with {self.num_joints} joints")
    
    def _map_joints(self):
        """Map joints to match C++ structure."""
        self.joint2 = []  # Dummy leg joints
        self.joint = []   # Active leg joints
        self.leg_joints = {}  # Mapping from leg to joint indices
        
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
        for leg_idx in range(self.leg_count):
            # Each leg has 3 joints
            start_idx = leg_idx * 3
            self.leg_joints[leg_idx] = [
                self.joint[start_idx],
                self.joint[start_idx + 1], 
                self.joint[start_idx + 2]
            ]
    
    def _configure_robot_dynamics(self):
        """Configure robot dynamics to reduce vibrations."""
        # Configure base dynamics
        p.changeDynamics(
            self.body_id, -1,  # Base link
            lateralFriction=0.9,      # Increased friction for better grip
            spinningFriction=0.02,    # Slightly increased
            rollingFriction=0.002,    # Slightly increased
            restitution=0.05,         # Reduced bouncing
            contactDamping=80.0,      # Increased damping
            contactStiffness=4000.0,  # Increased stiffness
            linearDamping=0.15,       # Increased damping
            angularDamping=0.2        # Increased angular damping
        )
        
        # Configure joint dynamics
        for joint_idx in range(self.num_joints):
            joint_info = p.getJointInfo(self.body_id, joint_idx)
            joint_type = joint_info[2]
            
            if joint_type == p.JOINT_REVOLUTE:
                p.changeDynamics(
                    self.body_id, joint_idx,
                    lateralFriction=0.9,
                    spinningFriction=0.02,
                    rollingFriction=0.002,
                    restitution=0.05,
                    contactDamping=80.0,
                    contactStiffness=4000.0,
                    jointDamping=0.1,        # Increased joint damping
                    linearDamping=0.15,
                    angularDamping=0.2
                )
        
        # Disable default motor control for all active joints
        for joint_idx in self.joint:
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                force=0  # Disable default motors
            )
    
    def reset_posture(self, smooth=True):
        """Reset robot to initial posture with option for smooth transition."""
        # Set initial target angles
        for i in range(self.leg_count):
            for j in range(self.dof):
                if i < 3:  # Right side legs
                    self.tang[i][j] = -np.radians(self.q_init[j])
                else:      # Left side legs
                    self.tang[i][j] = np.radians(self.q_init[j])
        
        if smooth:
            # Smooth transition using motor control
            self._apply_smooth_position_control()
        else:
            # Instant reset (only for true initialization)
            self._apply_position_control_direct()
    
    def apply_target_angles(self):
        """
        Apply target angles using enhanced POSITION_CONTROL with PD gains.
        This replaces the vibration-prone VELOCITY_CONTROL approach.
        """
        # Update current angles
        for i in range(self.leg_count):
            for j in range(self.dof):
                joint_idx = self.leg_joints[i][j]
                joint_state = p.getJointState(self.body_id, joint_idx)
                self.qang[i][j] = joint_state[0]
        
        # Apply POSITION_CONTROL with proper PD gains
        target_positions = []
        joint_indices = []
        forces = []
        position_gains = []
        velocity_gains = []
        
        for i in range(self.leg_count):
            for j in range(self.dof):
                joint_idx = self.leg_joints[i][j]
                target_angle = self.tang[i][j] * self.posz
                
                joint_indices.append(joint_idx)
                target_positions.append(target_angle)
                forces.append(self.max_force)
                position_gains.append(self.kp)
                velocity_gains.append(self.kd)
        
        # Apply motor control using vectorized approach
        p.setJointMotorControlArray(
            bodyUniqueId=self.body_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=forces,
            positionGains=position_gains,
            velocityGains=velocity_gains
        )
    
    def _apply_position_control_direct(self):
        """Apply position control for immediate positioning (used in reset)."""
        for i in range(self.leg_count):
            for j in range(self.dof):
                joint_idx = self.leg_joints[i][j]
                target_angle = self.tang[i][j] * self.posz
                
                # Reset joint to target position
                p.resetJointState(self.body_id, joint_idx, target_angle)
    
    def _apply_smooth_position_control(self):
        """Apply position control for smooth transitions."""
        # Use reduced gains for smooth movement
        target_positions = []
        joint_indices = []
        forces = []
        position_gains = []
        velocity_gains = []
        
        for i in range(self.leg_count):
            for j in range(self.dof):
                joint_idx = self.leg_joints[i][j]
                target_angle = self.tang[i][j] * self.posz
                
                joint_indices.append(joint_idx)
                target_positions.append(target_angle)
                forces.append(self.max_force * 0.5)  # Reduced force for smooth movement
                position_gains.append(self.kp * 0.3)  # Reduced gain
                velocity_gains.append(self.kd * 2.0)  # Increased damping
        
        # Apply smooth motor control
        p.setJointMotorControlArray(
            bodyUniqueId=self.body_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=forces,
            positionGains=position_gains,
            velocityGains=velocity_gains
        )
    
    def set_target_angles(self, angles):
        """
        Set target angles for all joints.
        
        Args:
            angles: Array of shape (leg_count, dof) with target angles in radians
        """
        if angles.shape != (self.leg_count, self.dof):
            # Handle flattened input
            if len(angles.flatten()) == self.leg_count * self.dof:
                angles = angles.reshape(self.leg_count, self.dof)
            else:
                raise ValueError(f"Expected angles shape {(self.leg_count, self.dof)}, got {angles.shape}")
        
        self.tang = angles.copy()
    
    def update_orientation(self):
        """Update posz based on robot orientation (upright vs flipped)."""
        try:
            _, orn = p.getBasePositionAndOrientation(self.body_id)
            rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            
            # Check if robot is upside down
            if rot_matrix[2, 2] < -0.7:  # z-component of z-axis
                self.posz = -1
            else:
                self.posz = 1
        except:
            # Handle case where robot might be removed
            self.posz = 1
    
    def get_position(self):
        """Get current position of robot body."""
        try:
            pos, _ = p.getBasePositionAndOrientation(self.body_id)
            return pos
        except:
            return [0, 0, 0]
    
    def get_orientation(self):
        """Get current orientation of robot body."""
        try:
            _, orn = p.getBasePositionAndOrientation(self.body_id)
            return orn
        except:
            return [0, 0, 0, 1]
    
    def get_state(self):
        """Get complete state of the robot."""
        try:
            pos, orn = p.getBasePositionAndOrientation(self.body_id)
            rot_matrix = p.getMatrixFromQuaternion(orn)
            
            # Get leg positions for all active leg end segments
            leg_positions = []
            for i in range(self.leg_count):
                # Get position of the last segment of each leg
                last_joint_idx = self.leg_joints[i][2]
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
        except:
            # Return default state if robot is not available
            return {
                'position': [0, 0, 0],
                'orientation': [0, 0, 0, 1],
                'rotation_matrix': [1, 0, 0, 0, 1, 0, 0, 0, 1],
                'joint_angles': np.zeros((self.leg_count, self.dof)),
                'leg_positions': [[0, 0, 0]] * self.leg_count
            }
    
    def set_motor_gains(self, kp=None, kd=None, max_force=None):
        """
        Set motor control gains for fine-tuning.
        
        Args:
            kp: Position gain
            kd: Velocity gain  
            max_force: Maximum motor force
        """
        if kp is not None:
            self.kp = kp
        if kd is not None:
            self.kd = kd
        if max_force is not None:
            self.max_force = max_force
        
        print(f"Motor gains updated: kp={self.kp}, kd={self.kd}, max_force={self.max_force}")
    
    def check_stability(self):
        """
        Check robot stability and return stability metrics.
        
        Returns:
            Dictionary with stability information
        """
        try:
            pos, orn = p.getBasePositionAndOrientation(self.body_id)
            rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            
            # Vertical orientation (z-component of z-axis)
            vertical_stability = rot_matrix[2, 2]
            
            # Height above ground
            height = pos[2]
            
            # Angular velocity
            linear_vel, angular_vel = p.getBaseVelocity(self.body_id)
            angular_speed = np.linalg.norm(angular_vel)
            
            return {
                'vertical_stability': vertical_stability,
                'height': height,
                'angular_speed': angular_speed,
                'is_stable': vertical_stability > 0.6 and angular_speed < 3.0  # FIXED: More lenient thresholds
            }
        except:
            return {
                'vertical_stability': 0.0,
                'height': 0.0,
                'angular_speed': 10.0,
                'is_stable': False
            }