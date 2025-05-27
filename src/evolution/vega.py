import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
import math
import logging
import json
from datetime import datetime
import pandas as pd


class VEGA:
    """
    Enhanced Virus-Host coEvolutionary Genetic Algorithm (VEGA) with 
    comprehensive fitness function to prevent leg vibrations and improve locomotion.
    """
    
    def __init__(self, population_size=30, chromosome_length=10, generations=500):
        """Initialize the enhanced VEGA algorithm."""
        # Setup experiment logging
        self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join('logs', 'evolution', self.experiment_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self._setup_logging()
        
        # Algorithm parameters
        self.gan = population_size    # Host population size
        self.gav = 20                 # Virus population size  
        self.gal = chromosome_length  # Chromosome length
        
        # Robot parameters
        self.dof = 3                  # Degree of freedom
        self.leg = 6                  # Number of legs
        
        # Angle limits (in degrees)
        self.q_min = np.array([-45, 0, 0])
        self.q_range = np.array([90, 60, 60])
        self.q_init = np.array([0, 45, 45])
        
        # Populations and fitness arrays
        self.hosts = np.zeros((self.gan, self.gal, 2, self.dof))
        self.virus = np.zeros((self.gav, self.dof))
        
        self.host_lengths = np.zeros(self.gan, dtype=int)
        # Enhanced fitness with multiple objectives
        self.fitness = np.zeros((self.gan, 6))  # Increased from 3 to 6 objectives
        self.fitv = np.zeros((self.gav, 6))
        
        # For VEGA ranking 
        self.gac = np.full(self.gan, -1)
        
        # Enhanced fitness tracking
        self.iterations = generations
        self.bfith  = np.zeros((self.iterations + 1, 6))  # Best fitness history
        self.cfith  = np.zeros((self.iterations + 1, 6))  # Current fitness history
        self.bhostl = np.zeros((self.iterations + 1, 6), dtype=int)
        self.chostl = np.zeros(self.iterations + 1, dtype=int)
        
        # Fitness component weights (tunable)
        self.fitness_weights = {
            'forward_motion': 1.0,
            'stability': 2.0,        # High weight for stability
            'energy_efficiency': 0.5,
            'smoothness': 1.5,       # High weight for smooth motion
            'direction_control': 1.0,
            'foot_contact': 0.8
        }
        
        # Current individual and sequence indices
        self.gai = 0
        self.gaj = 0
        self.iteration = 0
        
        # Enhanced tracking for stability
        self.stability_history = []
        self.prev_robot_state = None
        self.motion_history = []
        
        # Initialize populations
        self.initialize_populations()
        
        # Create necessary directories
        for subdir in ['models', 'checkpoints', 'data', 'plots']:
            os.makedirs(os.path.join(self.log_dir, subdir), exist_ok=True)
        
        self._save_config()
        
        self.logger.info(f"Enhanced VEGA initialized with comprehensive fitness function")
        
    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger('enhanced_vega')
        self.logger.setLevel(logging.INFO)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'evolution.log'))
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _save_config(self):
        """Save experiment configuration."""
        config = {
            'population_size': self.gan,
            'virus_population': self.gav,
            'chromosome_length': self.gal,
            'fitness_weights': self.fitness_weights,
            'q_min': self.q_min.tolist(),
            'q_range': self.q_range.tolist(),
            'q_init': self.q_init.tolist(),
            'experiment_id': self.experiment_id,
            'timestamp': time.time(),
            'date': datetime.now().isoformat()
        }
        
        config_file = os.path.join(self.log_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def initialize_populations(self):
        """Initialize populations with better diversity."""
        # Initialize host population
        for i in range(self.gan):
            # Random sequence length between 2 and 4
            self.host_lengths[i] = 2 + int(np.random.random() * 3)
            
            # Initialize each position with better constraints
            for m in range(self.host_lengths[i]):
                for phase in range(2):
                    for j in range(self.dof):
                        # Use smaller initial variations to reduce extreme movements
                        center = self.q_init[j] if j > 0 else 0  # Keep first DOF near 0
                        variation = self.q_range[j] * 0.3  # Reduce initial variation
                        
                        angle = center + (np.random.random() - 0.5) * variation
                        
                        # Ensure within bounds
                        angle = max(self.q_min[j], min(self.q_min[j] + self.q_range[j], angle))
                        self.hosts[i, m, phase, j] = angle
        
        # Initialize virus population
        for i in range(self.gav):
            for j in range(self.dof):
                self.virus[i, j] = self.q_min[j] + self.q_range[j] * np.random.random()
        
        # Initialize fitness arrays
        self.fitness = np.zeros((self.gan, 6))
        self.fitv = np.zeros((self.gav, 6))
        
        self.logger.info("Populations initialized with enhanced diversity control")
    
    def evaluate_fitness(self, robot, prev_pos, curr_pos, prev_rot, curr_rot):
        """
        Enhanced fitness evaluation with multiple objectives to prevent vibrations.
        
        Args:
            robot: Robot instance
            prev_pos: Previous robot position
            curr_pos: Current robot position  
            prev_rot: Previous rotation matrix
            curr_rot: Current rotation matrix
            
        Returns:
            Updated fitness array for current individual
        """
        # Get robot state for comprehensive evaluation
        robot_state = robot.get_state()
        stability_metrics = robot.check_stability()
        
        # Calculate basic movement metrics
        displacement = np.array(curr_pos) - np.array(prev_pos)
        distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
        
        # Calculate rotation change
        if curr_rot[0, 0] == 0 and curr_rot[1, 0] == 0:
            ra = 0
        else:
            ra = math.atan2(curr_rot[1, 0], curr_rot[0, 0])
            
        if prev_rot[0, 0] == 0 and prev_rot[1, 0] == 0:
            rap = 0
        else:
            rap = math.atan2(prev_rot[1, 0], prev_rot[0, 0])
            
        angle_change = ra - rap
        if angle_change > math.pi:
            angle_change -= 2 * math.pi
        elif angle_change < -math.pi:
            angle_change += 2 * math.pi
        
        # Calculate direction alignment
        rr = np.array([curr_rot[0, 0], curr_rot[1, 0]])  # Current direction
        v = displacement[:2]  # Movement vector
        
        alignment = 0
        if distance > 0:
            alignment = np.dot(rr, v) / distance
        
        # Enhanced Fitness Components
        
        # 1. Forward Motion (basic locomotion)
        forward_fitness = math.exp(-angle_change**2) + distance * 10 + alignment
        
        # 2. Stability (CRITICAL for preventing vibrations)
        stability_fitness = self._calculate_stability_fitness(
            stability_metrics, curr_rot, robot_state
        )
        
        # 3. Energy Efficiency (prevents excessive joint movements)
        energy_fitness = self._calculate_energy_fitness(robot_state, distance)
        
        # 4. Smoothness (prevents jerky movements and vibrations)
        smoothness_fitness = self._calculate_smoothness_fitness(
            robot, robot_state, displacement
        )
        
        # 5. Direction Control (left/right turning ability)
        left_turn_fitness = math.exp(-(angle_change - math.pi * 0.5)**2) * 20 + math.exp(-distance**2)
        right_turn_fitness = math.exp(-(angle_change + math.pi * 0.5)**2) * 20 + math.exp(-distance**2)
        direction_fitness = (left_turn_fitness + right_turn_fitness) / 2
        
        # 6. Foot Contact Quality (stable ground contact)
        contact_fitness = self._calculate_contact_fitness(robot_state)
        
        # Apply weights and combine fitness components
        weighted_fitness = [
            forward_fitness * self.fitness_weights['forward_motion'],
            stability_fitness * self.fitness_weights['stability'],
            energy_fitness * self.fitness_weights['energy_efficiency'],
            smoothness_fitness * self.fitness_weights['smoothness'],
            direction_fitness * self.fitness_weights['direction_control'],
            contact_fitness * self.fitness_weights['foot_contact']
        ]
        
        # Update fitness for current individual
        self.fitness[self.gai] = weighted_fitness
        
        # Track stability for monitoring
        self.stability_history.append(stability_metrics['vertical_stability'])
        
        # Update fitness history
        for i in range(6):
            self.cfith[self.iteration, i] = self.fitness[self.gai, i]
        self.chostl[self.iteration] = self.host_lengths[self.gai]
        
        # Log comprehensive fitness metrics
        self.logger.info(
            f"Fitness - Forward: {forward_fitness:.3f}, Stability: {stability_fitness:.3f}, "
            f"Energy: {energy_fitness:.3f}, Smoothness: {smoothness_fitness:.3f}, "
            f"Direction: {direction_fitness:.3f}, Contact: {contact_fitness:.3f}"
        )
        self.logger.info(
            f"Robot metrics - Distance: {distance:.3f}, Vertical stability: {stability_metrics['vertical_stability']:.3f}, "
            f"Angular speed: {stability_metrics['angular_speed']:.3f}"
        )
        
        # Find best fitness for each objective
        if self.iteration < self.gan:
            h = self.iteration + 1
        else:
            h = self.gan
            
        for j in range(6):
            k = 0
            for i in range(h):
                if self.fitness[i, j] > self.fitness[k, j]:
                    k = i
            self.bfith[self.iteration, j] = self.fitness[k, j]
            self.bhostl[self.iteration, j] = self.host_lengths[k]
        
        # Apply penalties for poor stability (critical for preventing vibrations)
        if stability_metrics['vertical_stability'] < 0.5:
            self.logger.info(f"Stability penalty applied - vertical: {stability_metrics['vertical_stability']:.3f}")
            # Severely penalize unstable gaits
            for i in range(6):
                self.fitness[self.gai, i] *= 0.1
        
        # Apply penalties for excessive vibrations
        if stability_metrics['angular_speed'] > 5.0:
            self.logger.info(f"Vibration penalty applied - angular speed: {stability_metrics['angular_speed']:.3f}")
            for i in range(6):
                self.fitness[self.gai, i] *= 0.5
        
        # Reverse gait if moving backward
        if alignment < -0.5:
            self.logger.info(f"Reversing gait due to backward movement - alignment: {alignment:.3f}")
            self.reverse(self.gai)
        
        # Store current state for next evaluation
        self.prev_robot_state = robot_state
        
        return self.fitness[self.gai]
    
    def _calculate_stability_fitness(self, stability_metrics, curr_rot, robot_state):
        """Calculate stability-based fitness to prevent vibrations."""
        # Vertical orientation component (most important)
        vertical_component = stability_metrics['vertical_stability']
        
        # Height stability (penalize bouncing)
        height_component = 1.0 / (1.0 + abs(robot_state['position'][2] - 0.5))
        
        # Angular velocity penalty (prevent spinning/vibrating)
        angular_penalty = math.exp(-stability_metrics['angular_speed'])
        
        # Body roll/pitch stability
        rot_matrix = np.array(curr_rot).reshape(3, 3)
        roll_pitch_stability = math.exp(-(
            math.asin(max(-1, min(1, -rot_matrix[2, 1])))**2 +  # Roll
            math.asin(max(-1, min(1, rot_matrix[2, 0])))**2      # Pitch
        ))
        
        # Combine stability components
        stability_fitness = (
            vertical_component * 2.0 +
            height_component * 1.0 +
            angular_penalty * 1.5 +
            roll_pitch_stability * 1.0
        ) / 5.5
        
        return max(0, stability_fitness * 100)  # Scale to reasonable range
    
    def _calculate_energy_fitness(self, robot_state, distance):
        """Calculate energy efficiency to prevent excessive movements."""
        # Simple energy model based on joint velocity variance
        joint_angles = robot_state['joint_angles']
        
        if self.prev_robot_state is not None:
            prev_angles = self.prev_robot_state['joint_angles']
            joint_velocities = np.abs(joint_angles - prev_angles)
            
            # Penalize high joint velocities
            energy_cost = np.sum(joint_velocities**2)
            
            # Reward distance per energy cost
            if energy_cost > 0:
                efficiency = distance / (energy_cost + 0.01)
            else:
                efficiency = distance
        else:
            efficiency = distance
        
        return max(0, efficiency * 50)  # Scale appropriately
    
    def _calculate_smoothness_fitness(self, robot, robot_state, displacement):
        """Calculate smoothness to prevent jerky movements and vibrations."""
        smoothness_score = 0
        
        # Track motion history for smoothness calculation
        self.motion_history.append({
            'position': robot_state['position'],
            'displacement': displacement,
            'joint_angles': robot_state['joint_angles']
        })
        
        # Keep only recent history
        if len(self.motion_history) > 10:
            self.motion_history.pop(0)
        
        if len(self.motion_history) >= 3:
            # Calculate acceleration (change in velocity)
            recent_displacements = [m['displacement'] for m in self.motion_history[-3:]]
            
            # Velocity smoothness
            vel_changes = []
            for i in range(1, len(recent_displacements)):
                vel_change = np.linalg.norm(recent_displacements[i] - recent_displacements[i-1])
                vel_changes.append(vel_change)
            
            # Penalize large velocity changes (jerky motion)
            if vel_changes:
                avg_vel_change = np.mean(vel_changes)
                smoothness_score = math.exp(-avg_vel_change * 10)
            
            # Joint angle smoothness
            recent_angles = [m['joint_angles'] for m in self.motion_history[-3:]]
            angle_changes = []
            for i in range(1, len(recent_angles)):
                angle_change = np.linalg.norm(recent_angles[i] - recent_angles[i-1])
                angle_changes.append(angle_change)
            
            if angle_changes:
                avg_angle_change = np.mean(angle_changes)
                joint_smoothness = math.exp(-avg_angle_change * 2)
                smoothness_score = (smoothness_score + joint_smoothness) / 2
        
        return max(0, smoothness_score * 100)
    
    def _calculate_contact_fitness(self, robot_state):
        """Calculate foot contact quality for stable locomotion."""
        # This is a simplified version - in full implementation would use
        # p.getContactPoints() to analyze actual ground contacts
        
        leg_positions = robot_state['leg_positions']
        contact_quality = 0
        
        # Check if feet are at reasonable heights
        for pos in leg_positions:
            if pos[2] < 0.1:  # Close to ground
                contact_quality += 1
        
        # Normalize by number of legs
        contact_quality /= len(leg_positions)
        
        return contact_quality * 50
    
    def rank(self):
        """Enhanced ranking for multi-objective optimization."""
        # Reset categories
        self.gac = np.full(self.gan, -1)
        
        # Assign individuals to objective categories (now 6 objectives)
        for j in range(self.gan):
            h = j % 6  # Cycle through 6 objectives instead of 3
            
            # Find first unassigned individual
            k = 0
            while self.gac[k] != -1:
                k += 1
            
            # Find best unassigned individual for this objective
            for i in range(k+1, self.gan):
                if self.gac[i] == -1 and self.fitness[i, h] > self.fitness[k, h]:
                    k = i
            
            self.gac[k] = h
        
        # Log enhanced rankings
        rank_str = "\nEnhanced Rankings (6 objectives):\n"
        obj_names = ["Forward", "Stability", "Energy", "Smoothness", "Direction", "Contact"]
        for i in range(self.gan):
            h = self.gac[i]
            if h >= 0 and h < len(obj_names):
                rank_str += f"r[{i}]:{obj_names[h]}, {self.fitness[i, h]:.2f}\n"
        
        self.logger.info(rank_str)
    
    def evolve(self):
        """Enhanced evolution with better stability focus."""
        self.rank()
        
        # Focus on stability-related objectives more often
        if self.iteration % 6 in [1, 3]:  # Focus on stability 2/6 times
            h = 1  # Stability objective
        else:
            h = self.iteration % 6
            
        obj_names = ["Forward", "Stability", "Energy", "Smoothness", "Direction", "Contact"]
        
        # Find worst and best individuals for this objective
        g1 = 0  # Worst
        while self.gac[g1] != h:
            g1 += 1
        g2 = g1  # Best
        
        for i in range(g1+1, self.gan):
            if self.gac[i] == h:
                if self.fitness[i, h] < self.fitness[g1, h]:
                    g1 = i
                elif self.fitness[i, h] > self.fitness[g2, h]:
                    g2 = i
        
        self.logger.info(f"Evolving for {obj_names[h]} objective")
        
        # Apply evolution with enhanced mutation rates for stability
        g3 = int(self.gan * np.random.random())
        r = np.random.random() * 0.4  # Slightly reduced crossover rate
        
        # Copy sequence length
        self.host_lengths[g1] = self.host_lengths[g2]
        
        # Enhanced crossover and mutation
        for m in range(self.host_lengths[g1]):
            for i in range(2):
                for j in range(self.dof):
                    if (np.random.random() < r) and (m < self.host_lengths[g3]):
                        # Crossover with reduced mutation for stability
                        mutation_factor = 0.1 if h == 1 else 0.2  # Less mutation for stability
                        self.hosts[g1, m, i, j] = (
                            self.hosts[g3, m, i, j] + 
                            self.randn() * self.q_range[j] * mutation_factor
                        )
                    else:
                        # Best individual with small mutation
                        mutation_factor = 0.05 if h == 1 else 0.1
                        self.hosts[g1, m, i, j] = (
                            self.hosts[g2, m, i, j] + 
                            self.randn() * self.q_range[j] * mutation_factor
                        )
                    
                    # Enforce bounds
                    self.hosts[g1, m, i, j] = np.clip(
                        self.hosts[g1, m, i, j],
                        self.q_min[j],
                        self.q_min[j] + self.q_range[j]
                    )
        
        # Apply specialized mutations with reduced probability for stability
        mutation_prob = 0.1 if h == 1 else 0.15
        
        # Insertion mutation
        if (self.host_lengths[g1] < self.gal - 1 and np.random.random() < mutation_prob):
            self.logger.info("-- insertion mutation --")
            self._apply_insertion_mutation(g1)
        
        # Deletion mutation  
        elif (self.host_lengths[g1] > 2 and np.random.random() < mutation_prob):
            self.logger.info("-- deletion mutation --")
            self._apply_deletion_mutation(g1)
        
        # Phase exchange (reduced probability for stability)
        if np.random.random() < mutation_prob * 0.5:
            self.logger.info("-- phase exchange mutation --")
            self._apply_phase_exchange_mutation(g1)
        
        # Order exchange
        elif np.random.random() < mutation_prob:
            self.logger.info("-- order exchange mutation --")
            self._apply_order_exchange_mutation(g1)
        
        self.gai = g1
        self.logger.info(f"Individual {self.gai} selected for {obj_names[h]}")
    
    def _apply_insertion_mutation(self, individual):
        """Apply insertion mutation."""
        k = int(self.host_lengths[individual] * np.random.random())
        
        if k < self.host_lengths[individual]:
            # Shift positions
            for m in range(self.host_lengths[individual], k, -1):
                for i in range(2):
                    for j in range(self.dof):
                        self.hosts[individual, m, i, j] = self.hosts[individual, m-1, i, j]
            
            # Insert new posture with conservative values
            for i in range(2):
                for j in range(self.dof):
                    center = self.q_init[j] if j > 0 else 0
                    variation = self.q_range[j] * 0.2  # Conservative variation
                    self.hosts[individual, k, i, j] = center + (np.random.random() - 0.5) * variation
        
        self.host_lengths[individual] += 1
    
    def _apply_deletion_mutation(self, individual):
        """Apply deletion mutation.""" 
        self.host_lengths[individual] -= 1
        k = int(self.host_lengths[individual] * np.random.random())
        
        if k < self.host_lengths[individual] - 1:
            # Shift positions
            for m in range(k, self.host_lengths[individual]):
                for i in range(2):
                    for j in range(self.dof):
                        self.hosts[individual, m, i, j] = self.hosts[individual, m+1, i, j]
    
    def _apply_phase_exchange_mutation(self, individual):
        """Apply phase exchange mutation."""
        m = int(self.host_lengths[individual] * np.random.random())
        
        # Swap phases
        for j in range(self.dof):
            temp = self.hosts[individual, m, 0, j]
            self.hosts[individual, m, 0, j] = self.hosts[individual, m, 1, j]
            self.hosts[individual, m, 1, j] = temp
    
    def _apply_order_exchange_mutation(self, individual):
        """Apply order exchange mutation."""
        k = int(self.host_lengths[individual] * np.random.random())
        m = int(self.host_lengths[individual] * np.random.random())
        
        if k != m:
            # Swap positions
            for i in range(2):
                for j in range(self.dof):
                    temp = self.hosts[individual, k, i, j]
                    self.hosts[individual, k, i, j] = self.hosts[individual, m, i, j]
                    self.hosts[individual, m, i, j] = temp
    
    def reverse(self, n):
        """Reverse motion sequence for backward movement."""
        self.logger.info(f"Reversing motion sequence for individual {n}")
        
        for m in range(self.host_lengths[n]):
            for i in range(2):
                for j in range(0, self.dof, 3):  # Only first DOF
                    self.hosts[n, m, i, j] = -self.hosts[n, m, i, j]
    
    def get_target_angles(self):
        """Get target angles with enhanced bounds checking."""
        gaj = self.gaj % self.host_lengths[self.gai]
        angles = np.zeros((6, 3))
        
        for i in range(6):
            for j in range(3):
                if j == 0:  # First DOF
                    phase = 0 if i % 2 == 0 else 1
                    angle_deg = self.hosts[self.gai, gaj, phase, j]
                    # Clamp to reasonable range to prevent extreme movements
                    angle_deg = np.clip(angle_deg, -30, 30)  # More conservative range
                    angles[i, j] = np.radians(angle_deg)
                else:  # DOF 1 and 2
                    phase = 0 if i % 2 == 0 else 1
                    angle_deg = self.hosts[self.gai, gaj, phase, j]
                    # Ensure reasonable joint limits
                    angle_deg = np.clip(angle_deg, self.q_min[j], self.q_min[j] + self.q_range[j])
                    
                    if i < 3:  # Right side
                        angles[i, j] = -np.radians(angle_deg)
                    else:  # Left side
                        angles[i, j] = np.radians(angle_deg)
        
        return angles
    
    def plot_enhanced_fitness_history(self):
        """Plot comprehensive fitness history for all 6 objectives."""
        plt.figure(figsize=(15, 20))
        
        objectives = ["Forward", "Stability", "Energy", "Smoothness", "Direction", "Contact"]
        
        for i in range(6):
            plt.subplot(7, 1, i+1)
            plt.plot(self.bfith[:self.iteration+1, i], 'b-', label=f'Best {objectives[i]}')
            plt.plot(self.cfith[:self.iteration+1, i], 'r--', label=f'Current {objectives[i]}')
            plt.legend()
            plt.grid(True)
            plt.ylabel('Fitness')
            plt.title(f'{objectives[i]} Fitness Evolution')
        
        # Plot stability history
        plt.subplot(7, 1, 7)
        if self.stability_history:
            plt.plot(self.stability_history, 'g-', label='Vertical Stability')
            plt.axhline(y=0.7, color='r', linestyle='--', label='Stability Threshold')
            plt.legend()
            plt.grid(True)
            plt.ylabel('Stability')
            plt.xlabel('Evaluation')
            plt.title('Robot Stability Over Time')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.log_dir, 'plots', 'enhanced_fitness_history.png')
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Enhanced fitness history plot saved to {plot_path}")
        return plot_path
    
    @staticmethod
    def randn():
        """Generate random number from normal distribution."""
        return (np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() - 6.0)
    
    def save_fitness_data(self):
        """Save enhanced fitness data with proper array length handling."""
        # Ensure we have valid iteration count
        if self.iteration < 0:
            self.iteration = 0
        
        # Create base data with consistent lengths
        num_iterations = self.iteration + 1
        
        data = {
            'iteration': range(num_iterations),
            'best_forward': self.bfith[:num_iterations, 0],
            'best_stability': self.bfith[:num_iterations, 1], 
            'best_energy': self.bfith[:num_iterations, 2],
            'best_smoothness': self.bfith[:num_iterations, 3],
            'best_direction': self.bfith[:num_iterations, 4],
            'best_contact': self.bfith[:num_iterations, 5],
            'current_forward': self.cfith[:num_iterations, 0],
            'current_stability': self.cfith[:num_iterations, 1],
            'current_energy': self.cfith[:num_iterations, 2],
            'current_smoothness': self.cfith[:num_iterations, 3],
            'current_direction': self.cfith[:num_iterations, 4],
            'current_contact': self.cfith[:num_iterations, 5]
        }
        
        # Create DataFrame from base data first
        df = pd.DataFrame(data)
        
        # Handle stability_history separately - it may have different length
        if self.stability_history:
            # Truncate or pad stability_history to match iteration count
            if len(self.stability_history) >= num_iterations:
                # Take the last num_iterations values
                stability_data = self.stability_history[-num_iterations:]
            else:
                # Pad with the last value or zeros
                stability_data = list(self.stability_history)
                last_value = stability_data[-1] if stability_data else 0.0
                while len(stability_data) < num_iterations:
                    stability_data.append(last_value)
            
            df['stability_history'] = stability_data
        else:
            # If no stability history, fill with zeros
            df['stability_history'] = [0.0] * num_iterations
        
        # Save to CSV
        csv_filename = os.path.join(self.log_dir, 'data', f"enhanced_evolution_data_{self.iteration:06d}.csv")
        df.to_csv(csv_filename, index=False)
        
        self.logger.info(f"Enhanced fitness data saved to {csv_filename}")
        return csv_filename
    
    def save_best_controller(self, filename=None):
        """
        Save the best controller (chromosome) found during evolution.
        
        Args:
            filename: Optional filename to save to. If None, auto-generates.
            
        Returns:
            Path to saved controller file
        """
        if filename is None:
            filename = os.path.join(self.log_dir, 'models', f'best_controller_{self.iteration:06d}.pkl')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Find best individual across all objectives
        best_overall_idx = 0
        best_overall_score = 0
        
        # Calculate weighted sum of all objectives for each individual
        for i in range(self.gan):
            weighted_score = (
                self.fitness[i, 0] * self.fitness_weights['forward_motion'] +
                self.fitness[i, 1] * self.fitness_weights['stability'] +
                self.fitness[i, 2] * self.fitness_weights['energy_efficiency'] +
                self.fitness[i, 3] * self.fitness_weights['smoothness'] +
                self.fitness[i, 4] * self.fitness_weights['direction_control'] +
                self.fitness[i, 5] * self.fitness_weights['foot_contact']
            )
            
            if weighted_score > best_overall_score:
                best_overall_score = weighted_score
                best_overall_idx = i
        
        # Save best controller data
        controller_data = {
            'individual_index': best_overall_idx,
            'sequence_length': self.host_lengths[best_overall_idx],
            'sequences': self.hosts[best_overall_idx, :self.host_lengths[best_overall_idx], :, :].copy(),
            'fitness_values': self.fitness[best_overall_idx].copy(),
            'weighted_score': best_overall_score,
            'fitness_weights': self.fitness_weights.copy(),
            'iteration_found': self.iteration,
            'chromosome_length': self.gal,
            'dof': self.dof,
            'q_min': self.q_min.copy(),
            'q_range': self.q_range.copy(),
            'q_init': self.q_init.copy(),
            'experiment_id': self.experiment_id,
            'timestamp': time.time()
        }
        
        # Save using pickle
        with open(filename, 'wb') as f:
            pickle.dump(controller_data, f)
        
        self.logger.info(f"Best controller saved to {filename}")
        self.logger.info(f"Best individual: {best_overall_idx}, Sequence length: {self.host_lengths[best_overall_idx]}")
        self.logger.info(f"Fitness values: {self.fitness[best_overall_idx]}")
        self.logger.info(f"Weighted score: {best_overall_score:.3f}")
        
        return filename
    
    def save_summary(self, filename=None):
        """
        Save a summary of the evolution results.
        
        Args:
            filename: Optional filename to save to. If None, auto-generates.
            
        Returns:
            Path to saved summary file
        """
        if filename is None:
            filename = os.path.join(self.log_dir, f'evolution_summary_{self.iteration:06d}.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Calculate final statistics
        final_stats = {}
        
        for i, obj_name in enumerate(['forward_motion', 'stability', 'energy_efficiency', 
                                     'smoothness', 'direction_control', 'foot_contact']):
            final_stats[obj_name] = {
                'best_fitness': float(np.max(self.fitness[:, i])),
                'mean_fitness': float(np.mean(self.fitness[:, i])),
                'std_fitness': float(np.std(self.fitness[:, i])),
                'final_best': float(self.bfith[self.iteration, i]) if self.iteration < len(self.bfith) else 0.0
            }
        
        # Overall summary
        summary = {
            'experiment_id': self.experiment_id,
            'completion_time': time.time(),
            'total_iterations': self.iteration,
            'population_size': self.gan,
            'chromosome_length_range': [int(np.min(self.host_lengths)), int(np.max(self.host_lengths))],
            'fitness_weights': self.fitness_weights,
            'final_statistics': final_stats,
            'convergence_achieved': len(self.stability_history) > 100 and np.mean(self.stability_history[-50:]) > 0.8,
            'stability_failures': sum(1 for s in self.stability_history if s < 0.5) if self.stability_history else 0,
            'avg_stability': float(np.mean(self.stability_history)) if self.stability_history else 0.0
        }
        
        # Save summary
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Evolution summary saved to {filename}")
        return filename