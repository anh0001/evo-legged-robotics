import pybullet as p
import numpy as np
import math


class LeggedRobot:
    """
    A multi-legged robot model implemented in PyBullet.
    This is a port of the original ODE robot model from the C++ code.
    """
    
    def __init__(self, client=None):
        """
        Initialize the robot with default parameters.
        
        Args:
            client: PyBullet physics client ID
        """
        # Store physics client
        self.client = client if client is not None else p.connect(p.DIRECT)
        
        # Robot body parameters
        self.box_pos = [0.0, 0.0, 0.5]
        self.box_length = 1.0
        self.box_width = 0.4
        self.box_height = 0.2
        self.box_mass = 1.0
        
        # Robot leg parameters
        self.leg_count = 6
        self.total_legs = 18
        self.dummy_legs = 6
        self.bar_length = 0.1
        self.bar_width = 0.2
        self.bar_height = 0.1
        self.bar_mass = 0.05
        self.bar_rest = 0.04
        
        # Joint parameters
        self.dof = 3  # degrees of freedom per leg
        
        # Min and max joint angles (in degrees, will be converted to radians)
        self.q_min = [-45, 0, 0]
        self.q_range = [90, 60, 60]
        self.q_init = [0, 45, 45]
        
        # IDs for bodies and joints in PyBullet
        self.body_id = None
        self.leg_ids = []
        self.dummy_leg_ids = []
        self.joint_ids = []
        self.dummy_joint_ids = []
        
        # Current and target joint angles
        self.q_angle = np.zeros((self.leg_count, self.dof))
        self.t_angle = np.zeros((self.leg_count, self.dof))
        
        # Leg positions
        self.bar_pos = self._calculate_leg_positions()
        self.bar_pos2 = self._calculate_dummy_leg_positions()
        
        # Build the robot
        self._build_robot()
    
    def _calculate_leg_positions(self):
        """Calculate positions for all leg segments."""
        positions = []
        
        # Front right legs
        positions.append([(self.box_length-self.bar_length)*0.5, -self.box_width*0.5-self.bar_width*0.5-(self.bar_rest+self.bar_width), self.box_pos[2]])
        positions.append([(self.box_length-self.bar_length)*0.5, -self.box_width*0.5-self.bar_width*0.5-(self.bar_rest+self.bar_width)*2, self.box_pos[2]])
        positions.append([(self.box_length-self.bar_length)*0.5, -self.box_width*0.5-self.bar_width*0.5-(self.bar_rest+self.bar_width)*3, self.box_pos[2]])
        
        # Middle right legs
        positions.append([0.0, -self.box_width*0.5-self.bar_width*0.5-(self.bar_rest+self.bar_width), self.box_pos[2]])
        positions.append([0.0, -self.box_width*0.5-self.bar_width*0.5-(self.bar_rest+self.bar_width)*2, self.box_pos[2]])
        positions.append([0.0, -self.box_width*0.5-self.bar_width*0.5-(self.bar_rest+self.bar_width)*3, self.box_pos[2]])
        
        # Back right legs
        positions.append([-(self.box_length-self.bar_length)*0.5, -self.box_width*0.5-self.bar_width*0.5-(self.bar_rest+self.bar_width), self.box_pos[2]])
        positions.append([-(self.box_length-self.bar_length)*0.5, -self.box_width*0.5-self.bar_width*0.5-(self.bar_rest+self.bar_width)*2, self.box_pos[2]])
        positions.append([-(self.box_length-self.bar_length)*0.5, -self.box_width*0.5-self.bar_width*0.5-(self.bar_rest+self.bar_width)*3, self.box_pos[2]])
        
        # Front left legs
        positions.append([(self.box_length-self.bar_length)*0.5, self.box_width*0.5+self.bar_width*0.5+(self.bar_rest+self.bar_width), self.box_pos[2]])
        positions.append([(self.box_length-self.bar_length)*0.5, self.box_width*0.5+self.bar_width*0.5+(self.bar_rest+self.bar_width)*2, self.box_pos[2]])
        positions.append([(self.box_length-self.bar_length)*0.5, self.box_width*0.5+self.bar_width*0.5+(self.bar_rest+self.bar_width)*3, self.box_pos[2]])
        
        # Middle left legs
        positions.append([0.0, self.box_width*0.5+self.bar_width*0.5+(self.bar_rest+self.bar_width), self.box_pos[2]])
        positions.append([0.0, self.box_width*0.5+self.bar_width*0.5+(self.bar_rest+self.bar_width)*2, self.box_pos[2]])
        positions.append([0.0, self.box_width*0.5+self.bar_width*0.5+(self.bar_rest+self.bar_width)*3, self.box_pos[2]])
        
        # Back left legs
        positions.append([-(self.box_length-self.bar_length)*0.5, self.box_width*0.5+self.bar_width*0.5+(self.bar_rest+self.bar_width), self.box_pos[2]])
        positions.append([-(self.box_length-self.bar_length)*0.5, self.box_width*0.5+self.bar_width*0.5+(self.bar_rest+self.bar_width)*2, self.box_pos[2]])
        positions.append([-(self.box_length-self.bar_length)*0.5, self.box_width*0.5+self.bar_width*0.5+(self.bar_rest+self.bar_width)*3, self.box_pos[2]])
        
        return positions
    
    def _calculate_dummy_leg_positions(self):
        """Calculate positions for dummy legs (attachment points)."""
        positions = []
        
        # Right side dummy legs
        positions.append([(self.box_length-self.bar_length)*0.5, -self.box_width*0.5-self.bar_width*0.5, self.box_pos[2]])
        positions.append([0.0, -self.box_width*0.5-self.bar_width*0.5, self.box_pos[2]])
        positions.append([-(self.box_length-self.bar_length)*0.5, -self.box_width*0.5-self.bar_width*0.5, self.box_pos[2]])
        
        # Left side dummy legs
        positions.append([(self.box_length-self.bar_length)*0.5, self.box_width*0.5+self.bar_width*0.5, self.box_pos[2]])
        positions.append([0.0, self.box_width*0.5+self.bar_width*0.5, self.box_pos[2]])
        positions.append([-(self.box_length-self.bar_length)*0.5, self.box_width*0.5+self.bar_width*0.5, self.box_pos[2]])
        
        return positions
    
    def _build_robot(self):
        """Build the robot in PyBullet."""
        # Create main body
        base_col_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.box_length/2, self.box_width/2, self.box_height/2]
        )
        base_vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.box_length/2, self.box_width/2, self.box_height/2],
            rgbaColor=[0.5, 0.5, 1.0, 1.0]
        )
        self.body_id = p.createMultiBody(
            baseMass=self.box_mass,
            baseCollisionShapeIndex=base_col_id,
            baseVisualShapeIndex=base_vis_id,
            basePosition=self.box_pos
        )
        
        # Create dummy legs (attachments to body)
        for i in range(self.dummy_legs):
            dummy_col_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self.bar_length/2, self.bar_width/2, self.bar_height/2]
            )
            dummy_vis_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.bar_length/2, self.bar_width/2, self.bar_height/2],
                rgbaColor=[0.6, 0.6, 1.0, 1.0]
            )
            dummy_leg = p.createMultiBody(
                baseMass=self.bar_mass,
                baseCollisionShapeIndex=dummy_col_id,
                baseVisualShapeIndex=dummy_vis_id,
                basePosition=self.bar_pos2[i]
            )
            self.dummy_leg_ids.append(dummy_leg)
            
            # Create constraint between body and dummy leg
            joint_id = p.createConstraint(
                parentBodyUniqueId=self.body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=dummy_leg,
                childLinkIndex=-1,
                jointType=p.JOINT_HINGE,
                jointAxis=[0, 1, 0],
                parentFramePosition=[
                    self.bar_pos2[i][0] - self.box_pos[0],
                    self.bar_pos2[i][1] - self.box_pos[1],
                    self.bar_pos2[i][2] - self.box_pos[2]
                ],
                childFramePosition=[0, 0, 0]
            )
            p.changeConstraint(joint_id, maxForce=10.0)
            self.dummy_joint_ids.append(joint_id)
        
        # Create actual leg segments and joints
        leg_col_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.bar_length/2, self.bar_width/2, self.bar_height/2]
        )
        leg_vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.bar_length/2, self.bar_width/2, self.bar_height/2],
            rgbaColor=[0.6, 0.6, 1.0, 1.0]
        )
        
        # For each leg position, create a leg and joint
        for i in range(self.total_legs):
            # Create leg segment
            leg = p.createMultiBody(
                baseMass=self.bar_mass,
                baseCollisionShapeIndex=leg_col_id,
                baseVisualShapeIndex=leg_vis_id,
                basePosition=self.bar_pos[i]
            )
            self.leg_ids.append(leg)
            
            # Create appropriate joint based on leg index
            if i % 3 == 0:  # First segment connects to dummy leg
                dummy_index = i // 3 if i < 9 else (i - 9) // 3 + 3
                
                # First joint connects to dummy leg
                joint_id = p.createConstraint(
                    parentBodyUniqueId=self.dummy_leg_ids[dummy_index],
                    parentLinkIndex=-1,
                    childBodyUniqueId=leg,
                    childLinkIndex=-1,
                    jointType=p.JOINT_HINGE,
                    jointAxis=[0, 1, 0],
                    parentFramePosition=[
                        self.bar_pos[i][0] - self.bar_pos2[dummy_index][0],
                        self.bar_pos[i][1] - self.bar_pos2[dummy_index][1],
                        self.bar_pos[i][2] - self.bar_pos2[dummy_index][2]
                    ],
                    childFramePosition=[0, 0, 0]
                )
            else:  # Other segments connect to previous leg segment
                joint_id = p.createConstraint(
                    parentBodyUniqueId=self.leg_ids[i-1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=leg,
                    childLinkIndex=-1,
                    jointType=p.JOINT_HINGE,
                    jointAxis=[1, 0, 0],  # Different axis for subsequent leg segments
                    parentFramePosition=[
                        self.bar_pos[i][0] - self.bar_pos[i-1][0],
                        self.bar_pos[i][1] - self.bar_pos[i-1][1],
                        self.bar_pos[i][2] - self.bar_pos[i-1][2]
                    ],
                    childFramePosition=[0, 0, 0]
                )
            
            # Set joint limits and other parameters
            p.changeConstraint(
                joint_id,
                maxForce=20.0,
                gearRatio=1,
                erp=0.2,
                cfm=0.00001
            )
            
            # Set joint limits
            lower_limit = -math.pi/2
            upper_limit = math.pi/2
            p.setJointMotorControl2(
                bodyUniqueId=self.leg_ids[i] if i % 3 != 0 else self.dummy_leg_ids[i//3],
                jointIndex=0,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )
            self.joint_ids.append(joint_id)
            
        # Initialize with default posture
        self.reset_posture()
    
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
        
        for i in range(self.leg_count):
            for j in range(self.dof):
                joint_index = i * self.dof + j
                current_angle = p.getJointState(self.leg_ids[joint_index], 0)[0]
                self.q_angle[i][j] = current_angle
                
                # Calculate velocity based on error
                velocity = gain * (self.t_angle[i][j] - current_angle)
                
                # Apply velocity to joint
                p.setJointMotorControl2(
                    bodyUniqueId=self.leg_ids[joint_index],
                    jointIndex=0,
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
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos
    
    def get_orientation(self):
        """Get current orientation of the robot body."""
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        return orn
    
    def get_state(self):
        """Get complete state of the robot (position, orientation, joint angles)."""
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        
        state = {
            'position': pos,
            'orientation': orn,
            'rotation_matrix': rot_matrix,
            'joint_angles': self.q_angle.copy()
        }
        return state