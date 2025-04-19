import numpy as np
import math
import pickle
import os


class LocomotionGenerator:
    """
    Generator for legged robot locomotion patterns.
    This is a port of the locomotion generation logic from the original C++ code.
    """
    
    def __init__(self, robot, time_step=0.01):
        """
        Initialize the locomotion generator.
        
        Args:
            robot: Robot instance to control
            time_step: Simulation time step
        """
        self.robot = robot
        self.time_step = time_step
        
        # Robot parameters
        self.dof = 3  # degrees of freedom per leg
        self.leg_count = 6  # number of legs
        
        # Motion sequence parameters
        self.num_phases = 0
        self.max_phases = 10
        self.current_phase = 0
        self.steps_per_phase = 20
        self.current_step = 0
        
        # Angle limits
        self.q_min = np.array([-45, 0, 0])
        self.q_range = np.array([90, 60, 60])
        self.q_init = np.array([0, 45, 45])
        
        # Phases for different types of motion
        self.phase_angles = np.zeros((self.max_phases, 2, self.dof))
        
        # Direction control
        self.direction = 0  # 0: forward, 1: left turn, 2: right turn
        
        # Robot state tracking
        self.pos = np.zeros(3)
        self.prev_pos = np.zeros(3)
        self.orientation = 0
        self.prev_orientation = 0
        
        # Default gait
        self.define_tripod_gait()
    
    def define_tripod_gait(self):
        """
        Define a basic tripod gait (alternating sets of legs).
        This is a common gait for hexapod robots.
        """
        # Reset phases
        self.num_phases = 2
        self.phase_angles = np.zeros((self.max_phases, 2, self.dof))
        
        # Phase 1: Legs 0, 2, 4 up and forward, others down and back
        self.phase_angles[0, 0, 0] = -30  # Right side, phase 0, DOF 0 (leg angle)
        self.phase_angles[0, 0, 1] = 30   # Right side, phase 0, DOF 1 (middle joint)
        self.phase_angles[0, 0, 2] = 60   # Right side, phase 0, DOF 2 (end joint)
        
        self.phase_angles[0, 1, 0] = 30   # Right side, phase 1, DOF 0
        self.phase_angles[0, 1, 1] = 45   # Right side, phase 1, DOF 1
        self.phase_angles[0, 1, 2] = 30   # Right side, phase 1, DOF 2
        
        # Phase 2: Opposite of phase 1
        self.phase_angles[1, 0, 0] = 30   # Right side, phase 0, DOF 0
        self.phase_angles[1, 0, 1] = 45   # Right side, phase 0, DOF 1
        self.phase_angles[1, 0, 2] = 30   # Right side, phase 0, DOF 2
        
        self.phase_angles[1, 1, 0] = -30  # Right side, phase 1, DOF 0
        self.phase_angles[1, 1, 1] = 30   # Right side, phase 1, DOF 1
        self.phase_angles[1, 1, 2] = 60   # Right side, phase 1, DOF 2
    
    def define_wave_gait(self):
        """
        Define a wave gait (legs move in sequence from back to front).
        This is good for stability on uneven terrain.
        """
        # Reset phases
        self.num_phases = 6  # One phase for each leg
        self.phase_angles = np.zeros((self.max_phases, 2, self.dof))
        
        # For each phase, one leg is lifted and moved forward
        for phase in range(self.num_phases):
            # All legs in default position
            for leg_phase in range(2):
                self.phase_angles[phase, leg_phase, 0] = 0   # Default leg angle
                self.phase_angles[phase, leg_phase, 1] = 45  # Default middle joint
                self.phase_angles[phase, leg_phase, 2] = 45  # Default end joint
            
            # Determine which leg to move in this phase
            leg = phase  # Each phase moves a different leg
            leg_phase = leg % 2  # 0 for even legs, 1 for odd legs
            
            # Move this leg forward and up
            self.phase_angles[phase, leg_phase, 0] = -30  # Forward
            self.phase_angles[phase, leg_phase, 1] = 30   # Up (middle joint)
            self.phase_angles[phase, leg_phase, 2] = 60   # End joint compensates
    
    def define_ripple_gait(self):
        """
        Define a ripple gait (pairs of legs move together).
        This is a compromise between speed and stability.
        """
        # Reset phases
        self.num_phases = 3  # Three phases for pairs of legs
        self.phase_angles = np.zeros((self.max_phases, 2, self.dof))
        
        # Phase 1: Legs 0 and 3 move (front legs)
        self.phase_angles[0, 0, 0] = -30  # Leg 0 forward
        self.phase_angles[0, 0, 1] = 30   # Leg 0 up
        self.phase_angles[0, 0, 2] = 60   # Leg 0 end joint
        
        self.phase_angles[0, 1, 0] = 0    # Other legs normal
        self.phase_angles[0, 1, 1] = 45
        self.phase_angles[0, 1, 2] = 45
        
        # Phase 2: Legs 2 and 5 move (middle legs)
        self.phase_angles[1, 0, 0] = 0    # Leg 0 normal
        self.phase_angles[1, 0, 1] = 45
        self.phase_angles[1, 0, 2] = 45
        
        self.phase_angles[1, 1, 0] = -30  # Leg 1 forward
        self.phase_angles[1, 1, 1] = 30
        self.phase_angles[1, 1, 2] = 60
        
        # Phase 3: Legs 1 and 4 move (rear legs)
        self.phase_angles[2, 0, 0] = 30   # Leg 0 back
        self.phase_angles[2, 0, 1] = 45
        self.phase_angles[2, 0, 2] = 30
        
        self.phase_angles[2, 1, 0] = 0    # Leg 1 normal
        self.phase_angles[2, 1, 1] = 45
        self.phase_angles[2, 1, 2] = 45
    
    def define_turn_left_gait(self):
        """Define a gait for turning left."""
        # Reset phases
        self.num_phases = 2
        self.phase_angles = np.zeros((self.max_phases, 2, self.dof))
        
        # Phase 1: Right legs forward, left legs backward
        self.phase_angles[0, 0, 0] = -30  # Right side forward
        self.phase_angles[0, 0, 1] = 30
        self.phase_angles[0, 0, 2] = 60
        
        self.phase_angles[0, 1, 0] = 30   # Left side backward
        self.phase_angles[0, 1, 1] = 45
        self.phase_angles[0, 1, 2] = 30
        
        # Phase 2: Right legs backward, left legs forward
        self.phase_angles[1, 0, 0] = 30   # Right side backward
        self.phase_angles[1, 0, 1] = 45
        self.phase_angles[1, 0, 2] = 30
        
        self.phase_angles[1, 1, 0] = -30  # Left side forward
        self.phase_angles[1, 1, 1] = 30
        self.phase_angles[1, 1, 2] = 60
    
    def define_turn_right_gait(self):
        """Define a gait for turning right."""
        # Reset phases
        self.num_phases = 2
        self.phase_angles = np.zeros((self.max_phases, 2, self.dof))
        
        # Phase 1: Right legs backward, left legs forward
        self.phase_angles[0, 0, 0] = 30   # Right side backward
        self.phase_angles[0, 0, 1] = 45
        self.phase_angles[0, 0, 2] = 30
        
        self.phase_angles[0, 1, 0] = -30  # Left side forward
        self.phase_angles[0, 1, 1] = 30
        self.phase_angles[0, 1, 2] = 60
        
        # Phase 2: Right legs forward, left legs backward
        self.phase_angles[1, 0, 0] = -30  # Right side forward
        self.phase_angles[1, 0, 1] = 30
        self.phase_angles[1, 0, 2] = 60
        
        self.phase_angles[1, 1, 0] = 30   # Left side backward
        self.phase_angles[1, 1, 1] = 45
        self.phase_angles[1, 1, 2] = 30
    
    def set_sequence_controller(self, controller):
        """
        Set a controller with pre-evolved sequences.
        
        Args:
            controller: Controller with sequences
        """
        # Extract sequences from controller
        if isinstance(controller, dict) and 'sequences' in controller:
            self.num_phases = controller['sequence_length']
            self.phase_angles = np.zeros((self.max_phases, 2, self.dof))
            
            # Copy sequences
            for i in range(min(self.num_phases, self.max_phases)):
                for j in range(2):
                    for k in range(self.dof):
                        self.phase_angles[i, j, k] = controller['sequences'][i, j, k]
    
    def get_next_angles(self):
        """
        Get the next set of target angles for the robot.
        
        Returns:
            Array of shape (leg_count, dof) with target angles
        """
        # Increment step counter
        self.current_step += 1
        
        # Move to next phase if needed
        if self.current_step >= self.steps_per_phase:
            self.current_step = 0
            self.current_phase = (self.current_phase + 1) % self.num_phases
        
        # Get current phase
        phase = self.current_phase
        
        # Create target angles array
        angles = np.zeros((self.leg_count, self.dof))
        # Clamp end-joint angles (DOF index 2) to safe range [0, 45°]
        for leg in range(self.leg_count):
            angles[leg, 2] = np.clip(angles[leg, 2], 0, math.radians(45))
        
        # Set target angles for each leg
        for leg in range(self.leg_count):
            for dof in range(self.dof):
                leg_phase = 0 if leg % 2 == 0 else 1  # Even legs use phase 0, odd use phase 1
                
                # Apply signs correctly based on DOF and leg position
                if dof == 0:  # First DOF (leg angle)
                    # No negation for DOF 0, just use the raw angle value
                    angles[leg, dof] = np.radians(self.phase_angles[phase, leg_phase, dof])
                else:  # Other DOFs (middle and end joints)
                    if leg < 3:  # Right side legs
                        # Negate angles for right-side legs for DOFs other than 0
                        angles[leg, dof] = -np.radians(self.phase_angles[phase, leg_phase, dof])
                    else:  # Left side legs
                        # Don't negate for left-side legs
                        angles[leg, dof] = np.radians(self.phase_angles[phase, leg_phase, dof])
        
        return angles
    
    def update_robot_state(self):
        """Update the internal state of the robot."""
        # Get current position and orientation
        pos = np.array(self.robot.get_position())
        state = self.robot.get_state()
        rot_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
        
        # Calculate orientation (angle around z-axis)
        if rot_matrix[0, 0] == 0 and rot_matrix[1, 0] == 0:
            orientation = 0
        else:
            orientation = math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
        
        # Store previous values
        self.prev_pos = self.pos
        self.prev_orientation = self.orientation
        
        # Update current values
        self.pos = pos
        self.orientation = orientation
    
    def calculate_fitness(self):
        """
        Calculate fitness metrics for the current motion.
        
        Returns:
            Dictionary of fitness metrics
        """
        # Make sure state is updated
        self.update_robot_state()
        
        # Calculate displacement
        displacement = self.pos - self.prev_pos
        distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
        
        # Calculate orientation change
        angle_change = self.orientation - self.prev_orientation
        
        # Adjust angle to range [-pi, pi]
        if angle_change > math.pi:
            angle_change -= 2 * math.pi
        elif angle_change < -math.pi:
            angle_change += 2 * math.pi
        
        # Calculate direction vector
        direction = np.array([math.cos(self.orientation), math.sin(self.orientation), 0])
        
        # Calculate alignment (dot product between displacement and direction)
        alignment = 0
        if distance > 0:
            alignment = (direction[0] * displacement[0] + direction[1] * displacement[1]) / distance
        
        # Calculate different fitness metrics
        fitness = {
            # Forward motion fitness
            'forward': math.exp(-angle_change**2) + distance * 10 + alignment,
            
            # Left turn fitness
            'left_turn': math.exp(-(angle_change + math.pi/2)**2) + math.exp(-distance**2),
            
            # Right turn fitness
            'right_turn': math.exp(-(angle_change - math.pi/2)**2) + math.exp(-distance**2),
        }
        
        return fitness
    
    def select_gait_for_direction(self, direction):
        """
        Select an appropriate gait for the desired direction.
        
        Args:
            direction: Direction to move (0: forward, 1: left, 2: right)
        """
        self.direction = direction
        
        if direction == 0:
            # Forward motion
            self.define_tripod_gait()
        elif direction == 1:
            # Turn left
            self.define_turn_left_gait()
        elif direction == 2:
            # Turn right
            self.define_turn_right_gait()
        else:
            # Default to forward motion
            self.define_tripod_gait()
    
    def save(self, filename):
        """
        Save the locomotion generator to a file.
        
        Args:
            filename: Path to save the generator
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create data to save
        data = {
            'num_phases': self.num_phases,
            'phase_angles': self.phase_angles,
            'direction': self.direction,
            'steps_per_phase': self.steps_per_phase
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filename, robot):
        """
        Load a locomotion generator from a file.
        
        Args:
            filename: Path to the saved generator
            robot: Robot instance
            
        Returns:
            Loaded locomotion generator
        """
        # Load data from file
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Create a new generator
        generator = cls(robot)
        
        # Set properties
        generator.num_phases = data['num_phases']
        generator.phase_angles = data['phase_angles']
        generator.direction = data['direction']
        generator.steps_per_phase = data['steps_per_phase']
        
        return generator