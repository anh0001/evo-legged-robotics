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
import pybullet as p

from .pareto import crowding_distance


class VEGA:
    """
    Enhanced Virus-Host coEvolutionary Genetic Algorithm (VEGA) with 
    comprehensive fitness function to prevent leg vibrations and improve locomotion.
    """
    
    def __init__(self, population_size=30, chromosome_length=10, generations=500,
                 elite_fraction=0.05, crossover_rate=0.8, mutation_prob=0.18,
                 mutation_factor_donor=0.3, mutation_factor_parent=0.2,
                 infection_factor=0.2, virus_mutation_factor=0.2):
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
        # Elite configuration
        self.elite_fraction = elite_fraction
        self.n_elite = max(1, int(self.gan * self.elite_fraction))

        # Tunable evolutionary parameters
        self.crossover_rate = crossover_rate
        self.mutation_prob = mutation_prob
        self.mutation_factor_donor = mutation_factor_donor
        self.mutation_factor_parent = mutation_factor_parent
        self.infection_factor = infection_factor
        self.virus_mutation_factor = virus_mutation_factor
        
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
        
        # Pareto ranking results
        self.parents = []
        self.pareto_front = []
        self.pareto_archive = []
        self.elite_archive = []
        
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
            'direction_control': 1.5,
            'foot_contact': 2.0
        }

        # Penalty factors for stability and angular velocity (values in [0,1))
        self.penalty_factors = {
            'stability_high': 0.5,
            'stability_low': 0.2,
            'angular_high': 0.6,
            'angular_low': 0.3,
            # Scaling of penalties per fitness objective
            'forward_weight': 0.3,
            'stability_weight': 1.0,
            'energy_weight': 0.2,
            'smoothness_weight': 0.2,
            'direction_weight': 0.2,
            'contact_weight': 0.1
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
            'penalty_factors': self.penalty_factors,
            'q_min': self.q_min.tolist(),
            'q_range': self.q_range.tolist(),
            'q_init': self.q_init.tolist(),
            'crossover_rate': self.crossover_rate,
            'mutation_prob': self.mutation_prob,
            'mutation_factor_donor': self.mutation_factor_donor,
            'mutation_factor_parent': self.mutation_factor_parent,
            'infection_factor': self.infection_factor,
            'virus_mutation_factor': self.virus_mutation_factor,
            'experiment_id': self.experiment_id,
            'timestamp': time.time(),
            'date': datetime.now().isoformat()
        }
        
        config_file = os.path.join(self.log_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def clear_motion_history(self):
        """Reset the stored motion history used for smoothness calculation."""
        self.motion_history = []
    
    def initialize_populations(self):
        """Initialize populations with better diversity."""
        # Initialize host population
        for i in range(self.gan):
            # Random sequence length between 2 and 4
            self.host_lengths[i] = 2 + int(np.random.random() * 3)
            assert 2 <= self.host_lengths[i] <= self.gal, "Invalid initial chromosome length"
            
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

        # Ensure all initial lengths are within valid bounds
        assert np.all((self.host_lengths >= 2) & (self.host_lengths <= self.gal)), "Host lengths out of bounds"
        
        self.logger.info("Populations initialized with enhanced diversity control")
    
    def evaluate_fitness(self, robot, prev_pos, curr_pos, prev_rot, curr_rot, ground_id):
        """
        Enhanced fitness evaluation with multiple objectives to prevent vibrations.
        
        Args:
            robot: Robot instance
            prev_pos: Previous robot position
            curr_pos: Current robot position  
            prev_rot: Previous rotation matrix
            curr_rot: Current rotation matrix
            ground_id: PyBullet ID of the ground body
            
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
            
        yaw = ra
        prev_yaw = rap
        direction_error = yaw - prev_yaw
        if direction_error > math.pi:
            direction_error -= 2 * math.pi
        elif direction_error < -math.pi:
            direction_error += 2 * math.pi

        angle_change = direction_error
        
        # Calculate direction alignment
        rr = np.array([curr_rot[0, 0], curr_rot[1, 0]])  # Current direction
        v = displacement[:2]  # Movement vector
        
        alignment = 0
        if distance > 0:
            alignment = np.dot(rr, v) / distance

        # FIXED: Improved Fitness Components with proper scaling
        
        # 1. Forward Motion (normalized to [0, 1])
        forward_base = math.exp(-angle_change**2) + distance * 10 + alignment
        forward_fitness = np.clip(forward_base / 20.0, 0, 1)  # Normalize to [0,1]
        
        # 2. Stability (normalized to [0, 1])
        stability_fitness = self._calculate_normalized_stability_fitness(
            stability_metrics, curr_rot, robot_state
        )
        
        # 3. Energy Efficiency (normalized to [0, 1]) 
        energy_fitness = self._calculate_normalized_energy_fitness(robot_state, distance)
        
        # 4. Smoothness (FIXED: better scaling)
        smoothness_fitness = self._calculate_normalized_smoothness_fitness(
            robot, robot_state, displacement
        )
        closure_penalty = self._cycle_closure_penalty(self.gai)
        smoothness_fitness *= math.exp(-closure_penalty * 2.0)
        smoothness_fitness = float(np.clip(smoothness_fitness, 0, 1))
        
        # 5. Direction Control (normalized to [0, 1])
        direction_fitness = math.exp(-direction_error**2)  # Already in [0,1]
        
        # 6. Foot Contact Quality (normalized to [0, 1])
        contact_fitness = self._calculate_normalized_contact_fitness(robot, ground_id)
        
        # FIXED: Weighted combination in normalized space [0, 1]
        normalized_fitness = [
            forward_fitness,
            stability_fitness, 
            energy_fitness,
            smoothness_fitness,
            direction_fitness,
            contact_fitness
        ]
        
        # Apply weights AFTER normalization
        weighted_fitness = [
            normalized_fitness[0] * self.fitness_weights['forward_motion'],
            normalized_fitness[1] * self.fitness_weights['stability'],
            normalized_fitness[2] * self.fitness_weights['energy_efficiency'], 
            normalized_fitness[3] * self.fitness_weights['smoothness'],
            normalized_fitness[4] * self.fitness_weights['direction_control'],
            normalized_fitness[5] * self.fitness_weights['foot_contact']
        ]

        # Multiplicative penalties to scale fitness components
        stability_penalty = 0.0
        if stability_metrics['vertical_stability'] < 0.3:
            stability_penalty = self.penalty_factors['stability_high']
        elif stability_metrics['vertical_stability'] < 0.5:
            stability_penalty = self.penalty_factors['stability_low']

        angular_penalty = 0.0
        if stability_metrics['angular_speed'] > 6.0:
            angular_penalty = self.penalty_factors['angular_high']
        elif stability_metrics['angular_speed'] > 4.0:
            angular_penalty = self.penalty_factors['angular_low']

        final_fitness = [
            weighted_fitness[0]
            * (1 - stability_penalty * self.penalty_factors['forward_weight']),
            weighted_fitness[1]
            * (1 - stability_penalty * self.penalty_factors['stability_weight']),
            weighted_fitness[2]
            * (1 - angular_penalty * self.penalty_factors['energy_weight']),
            weighted_fitness[3]
            * (1 - angular_penalty * self.penalty_factors['smoothness_weight']),
            weighted_fitness[4]
            * (1 - stability_penalty * self.penalty_factors['direction_weight']),
            weighted_fitness[5]
            * (1 - stability_penalty * self.penalty_factors['contact_weight']),
        ]
        final_fitness = [max(0, f) for f in final_fitness]
        
        self.fitness[self.gai] = final_fitness

        # Track stability for monitoring
        self.stability_history.append(stability_metrics['vertical_stability'])

        # Update fitness history
        for i in range(6):
            self.cfith[self.iteration, i] = self.fitness[self.gai, i]
        self.chostl[self.iteration] = self.host_lengths[self.gai]

        # Log fitness metrics
        self.logger.info(
            f"Fitness - Forward: {self.fitness[self.gai, 0]:.3f}, Stability: {self.fitness[self.gai, 1]:.3f}, "
            f"Energy: {self.fitness[self.gai, 2]:.3f}, Smoothness: {self.fitness[self.gai, 3]:.3f}, "
            f"Direction: {self.fitness[self.gai, 4]:.3f}, Contact: {self.fitness[self.gai, 5]:.3f}"
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
        
        # Reverse gait if moving backward
        if alignment < -0.5:
            self.logger.info(f"Reversing gait due to backward movement - alignment: {alignment:.3f}")
            self.reverse(self.gai)
        
        # Store current state for next evaluation
        self.prev_robot_state = robot_state
        
        return self.fitness[self.gai]
    
    def _calculate_normalized_stability_fitness(self, stability_metrics, curr_rot, robot_state):
        """FIXED: Normalized stability fitness in [0, 1]."""
        vertical_component = max(0, stability_metrics['vertical_stability'])
        height_component = max(0, 1.0 / (1.0 + abs(robot_state['position'][2] - 0.35)))
        
        # More lenient angular penalty
        angular_penalty = max(0, 1.0 / (1.0 + stability_metrics['angular_speed'] / 3.0))
        
        rot_matrix = np.array(curr_rot).reshape(3, 3)
        try:
            roll = math.asin(max(-0.99, min(0.99, -rot_matrix[2, 1])))
            pitch = math.asin(max(-0.99, min(0.99, rot_matrix[2, 0])))
            roll_pitch_stability = math.exp(-(roll**2 + pitch**2) / 2.0)  # More lenient
        except:
            roll_pitch_stability = 0.5
        
        # Weighted combination normalized to [0, 1]
        stability_fitness = (
            vertical_component * 0.5 +
            height_component * 0.2 +
            angular_penalty * 0.2 +
            roll_pitch_stability * 0.1
        )
        
        return np.clip(stability_fitness, 0, 1)
    
    def _calculate_normalized_energy_fitness(self, robot_state, distance):
        """FIXED: Normalized energy fitness in [0, 1]."""
        joint_angles = robot_state['joint_angles']
        
        if self.prev_robot_state is not None:
            prev_angles = self.prev_robot_state['joint_angles']
            joint_velocities = np.abs(joint_angles - prev_angles)
            
            # Energy cost
            energy_cost = np.mean(joint_velocities**2)  # Use mean instead of sum
            
            # Efficiency = distance per unit energy cost
            if energy_cost > 0:
                efficiency = distance / (energy_cost + 0.01)
            else:
                efficiency = distance
        else:
            efficiency = distance
        
        # Normalize to [0, 1] - assume max efficiency around 5.0
        return np.clip(efficiency / 5.0, 0, 1)
    
    def _calculate_normalized_smoothness_fitness(self, robot, robot_state, displacement):
        """FIXED: Normalized smoothness fitness in [0, 1] with better scaling."""
        smoothness_score = 0
        
        # Track motion history for smoothness calculation
        self.motion_history.append({
            'position': robot_state['position'],
            'displacement': displacement,
            'joint_angles': robot_state['joint_angles']
        })
        
        # Keep only recent history
        if len(self.motion_history) > 5:  # Shorter history for responsiveness
            self.motion_history.pop(0)
        
        if len(self.motion_history) >= 3:
            # Calculate velocity smoothness
            recent_displacements = [m['displacement'] for m in self.motion_history[-3:]]
            
            vel_changes = []
            for i in range(1, len(recent_displacements)):
                vel_change = np.linalg.norm(recent_displacements[i] - recent_displacements[i-1])
                vel_changes.append(vel_change)
            
            # FIXED: More lenient exponential decay
            if vel_changes:
                avg_vel_change = np.mean(vel_changes)
                smoothness_score = math.exp(-avg_vel_change * 2)  # Reduced from 10
            
            # Joint angle smoothness
            recent_angles = [m['joint_angles'] for m in self.motion_history[-3:]]
            angle_changes = []
            for i in range(1, len(recent_angles)):
                angle_change = np.mean(np.abs(recent_angles[i] - recent_angles[i-1]))  # Mean instead of norm
                angle_changes.append(angle_change)
            
            if angle_changes:
                avg_angle_change = np.mean(angle_changes)
                joint_smoothness = math.exp(-avg_angle_change * 1)  # Reduced from 2
                smoothness_score = (smoothness_score + joint_smoothness) / 2
        
        return np.clip(smoothness_score, 0, 1)
    
    def _calculate_normalized_contact_fitness(self, robot, ground_id):
        """FIXED: Normalized contact fitness in [0, 1]."""
        contacts = 0
        for i in range(robot.leg_count):
            foot_link = robot.leg_joints[i][2]
            pts = p.getContactPoints(bodyA=robot.body_id, linkIndexA=foot_link, bodyB=ground_id)
            if len(pts) > 0:
                contacts += 1

        # Already normalized to [0, 1]
        return contacts / robot.leg_count

    def _cycle_closure_penalty(self, idx):
        """Penalty for discontinuity between first and last poses."""
        length = int(self.host_lengths[idx])
        if length < 2:
            return 0.0

        start_pose = self.hosts[idx, 0]
        end_pose = self.hosts[idx, length - 1]

        diff = np.abs(end_pose - start_pose)
        norm_diff = diff / self.q_range
        penalty = float(np.mean(norm_diff))
        return np.clip(penalty, 0.0, 1.0)

    def infect_hosts(self):
        """Apply viruses to each host chromosome."""
        for host_idx in range(self.gan):
            virus_idx = int(np.random.random() * self.gav)
            for m in range(self.host_lengths[host_idx]):
                for phase in range(2):
                    for j in range(self.dof):
                        perturb = (
                            self.virus[virus_idx, j] +
                            self.randn() * self.q_range[j] * self.infection_factor
                        )
                        self.hosts[host_idx, m, phase, j] += perturb
                        self.hosts[host_idx, m, phase, j] = np.clip(
                            self.hosts[host_idx, m, phase, j],
                            self.q_min[j],
                            self.q_min[j] + self.q_range[j]
                        )
            self.logger.info(
                f"Host {host_idx} infected by virus {virus_idx}")

    def mutate_viruses(self):
        """Evolve virus population by small random changes."""
        for v in range(self.gav):
            for j in range(self.dof):
                if np.random.random() < 0.3:
                    self.virus[v, j] += self.randn() * self.q_range[j] * self.virus_mutation_factor
                    self.virus[v, j] = np.clip(
                        self.virus[v, j],
                        -self.q_range[j],
                        self.q_range[j]
                    )
        self.logger.info("Virus population mutated")
    
    def rank(self):
        """Rank population using non-dominated sorting and crowding distance."""
        from .pareto import non_dominated_sort, crowding_distance

        # Compute Pareto fronts
        self.fronts = non_dominated_sort(self.fitness)
        self.pareto_front = self.fronts[0] if self.fronts else []

        # Compute crowding distance for the first front
        distances = crowding_distance(self.fitness, self.pareto_front)

        # Sort individuals in the first front by crowding distance
        self.parents = sorted(self.pareto_front, key=lambda i: distances.get(i, 0), reverse=True)

        # Update external archive of Pareto-optimal individuals
        for idx in self.pareto_front:
            individual = {
                'length': int(self.host_lengths[idx]),
                'sequence': self.hosts[idx, :self.host_lengths[idx]].copy(),
                'fitness': self.fitness[idx].copy(),
            }
            self.pareto_archive.append(individual)

        self.logger.info(f"Pareto front size: {len(self.pareto_front)}")
    
    def evolve(self):
        """Enhanced evolution with better stability focus."""
        # Rank population before applying any virus operations
        self.rank()

        # Identify elite individuals based on Pareto ranking prior to mutation
        elite_count = max(1, int(self.gan * self.elite_fraction))
        elite_indices = self.parents[:elite_count]
        elite_hosts = self.hosts[elite_indices].copy()
        elite_lengths = self.host_lengths[elite_indices].copy()
        elite_fitness = self.fitness[elite_indices].copy()

        # Viruses mutate and infect hosts after ranking so elites are preserved
        self.mutate_viruses()
        self.infect_hosts()

        # Archive elites for reference
        self.elite_archive = [
            {
                'length': int(self.host_lengths[i]),
                'sequence': self.hosts[i, :self.host_lengths[i]].copy(),
                'fitness': self.fitness[i].copy(),
            }
            for i in elite_indices
        ]

        if len(self.parents) < 1:
            return

        # Tournament selection among Pareto front based on crowding distance
        distances = crowding_distance(self.fitness, self.parents)
        tournament_size = min(3, len(self.parents))
        candidates = np.random.choice(self.parents, size=tournament_size, replace=False)
        parent1 = max(candidates, key=lambda idx: distances.get(idx, 0))

        donor = int(self.gan * np.random.random())

        non_elites = [i for i in range(self.gan) if i not in elite_indices]
        if non_elites:
            target = non_elites[int(np.argmin(np.sum(self.fitness[non_elites], axis=1)))]
        else:
            target = int(np.argmin(np.sum(self.fitness, axis=1)))

        # Apply evolution using Pareto parents
        r = np.random.random() * self.crossover_rate

        self.host_lengths[target] = self.host_lengths[parent1]
        
        # FIXED: Enhanced mutation with stability focus
        for m in range(self.host_lengths[target]):
            for i in range(2):
                for j in range(self.dof):
                    if (np.random.random() < r) and (m < self.host_lengths[donor]):
                        self.hosts[target, m, i, j] = (
                            self.hosts[donor, m, i, j] +
                            self.randn() * self.q_range[j] * self.mutation_factor_donor
                        )
                    else:
                        self.hosts[target, m, i, j] = (
                            self.hosts[parent1, m, i, j] +
                            self.randn() * self.q_range[j] * self.mutation_factor_parent
                        )

                    self.hosts[target, m, i, j] = np.clip(
                        self.hosts[target, m, i, j],
                        self.q_min[j] + 0.1,
                        self.q_min[j] + self.q_range[j] - 0.1
                    )
        
        # FIXED: Reduced structural mutation probabilities  
        mutation_prob = self.mutation_prob
        
        # Insertion mutation
        if (self.host_lengths[target] < self.gal - 1 and np.random.random() < mutation_prob):
            self.logger.info("-- insertion mutation --")
            self._apply_insertion_mutation(target)
        
        # Deletion mutation  
        elif (self.host_lengths[target] > 2 and np.random.random() < mutation_prob):
            self.logger.info("-- deletion mutation --")
            self._apply_deletion_mutation(target)
        
        # Phase exchange (reduced probability for stability)
        if np.random.random() < mutation_prob * 0.5:
            self.logger.info("-- phase exchange mutation --")
            self._apply_phase_exchange_mutation(target)
        
        # Order exchange
        elif np.random.random() < mutation_prob:
            self.logger.info("-- order exchange mutation --")
            self._apply_order_exchange_mutation(target)
        
        self.gai = target
        self.clear_motion_history()

        # Reinsert elites unchanged
        for idx, e_idx in enumerate(elite_indices):
            self.hosts[e_idx] = elite_hosts[idx]
            self.host_lengths[e_idx] = elite_lengths[idx]
            self.fitness[e_idx] = elite_fitness[idx]

        # Propagate fitness history so elites persist across iterations
        if self.iteration > 0 and self.iteration < len(self.bfith):
            self.bfith[self.iteration] = self.bfith[self.iteration - 1]
            self.cfith[self.iteration] = self.cfith[self.iteration - 1]
            self.bhostl[self.iteration] = self.bhostl[self.iteration - 1]
            self.chostl[self.iteration] = self.chostl[self.iteration - 1]

        self.logger.info("Individual %d evolved from Pareto parents", self.gai)

        # Verify chromosome lengths remain within valid bounds
        assert np.all((self.host_lengths >= 2) & (self.host_lengths <= self.gal)), "Chromosome length out of bounds"
    
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
        self.clamp_chromosome_length(individual)
    
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

        self.clamp_chromosome_length(individual)
    
    def _apply_phase_exchange_mutation(self, individual):
        """Apply phase exchange mutation."""
        m = int(self.host_lengths[individual] * np.random.random())
        
        # Swap phases
        for j in range(self.dof):
            temp = self.hosts[individual, m, 0, j]
            self.hosts[individual, m, 0, j] = self.hosts[individual, m, 1, j]
            self.hosts[individual, m, 1, j] = temp

        self.clamp_chromosome_length(individual)
    
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

        self.clamp_chromosome_length(individual)

    def clamp_chromosome_length(self, individual):
        """Ensure chromosome length stays within [2, self.gal]."""
        self.host_lengths[individual] = int(np.clip(self.host_lengths[individual], 2, self.gal))
    
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
                    # FIXED: Even more conservative range
                    angle_deg = np.clip(angle_deg, -20, 20)  # Reduced from -30,30
                    angles[i, j] = np.radians(angle_deg)
                else:  # DOF 1 and 2
                    phase = 0 if i % 2 == 0 else 1
                    angle_deg = self.hosts[self.gai, gaj, phase, j]
                    # FIXED: Conservative joint limits with safety margins
                    min_angle = self.q_min[j] + 2.0  # Safety margin
                    max_angle = self.q_min[j] + self.q_range[j] - 2.0
                    angle_deg = np.clip(angle_deg, min_angle, max_angle)
                    
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
        
        # FIXED: Ensure all values are JSON serializable
        summary = {
            'experiment_id': str(self.experiment_id),
            'completion_time': float(time.time()),
            'total_iterations': int(self.iteration),
            'population_size': int(self.gan),
            'chromosome_length_range': [int(np.min(self.host_lengths)), int(np.max(self.host_lengths))],
            'fitness_weights': dict(self.fitness_weights),  # Ensure it's a regular dict
            'final_statistics': final_stats,
            'convergence_achieved': bool(len(self.stability_history) > 100 and np.mean(self.stability_history[-50:]) > 0.8),
            'stability_failures': int(sum(1 for s in self.stability_history if s < 0.5) if self.stability_history else 0),
            'avg_stability': float(np.mean(self.stability_history)) if self.stability_history else 0.0
        }
        
        # FIXED: Custom JSON encoder to handle any remaining numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        # Save summary with custom encoder
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"Evolution summary saved to {filename}")
        return filename