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
    Virus-Host coEvolutionary Genetic Algorithm (VEGA) implementation
    for multi-objective optimization of robot locomotion patterns.
    This is a direct port of the C++ VEGA implementation from the original ODE codebase,
    updated with modern Python logging practices.
    """
    
    def __init__(self, population_size=30, chromosome_length=10, generations=500):
        """
        Initialize the VEGA algorithm with parameters matching the C++ implementation.
        
        Args:
            population_size: Number of individuals in the population (GAN in C++)
            chromosome_length: Maximum length of locomotion sequences (GAL in C++)
            generations: Maximum number of generations to evolve
        """
        # Setup experiment logging
        self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join('logs', 'evolution', self.experiment_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self._setup_logging()
        
        # Match parameters from C++ implementation
        self.gan = population_size    # Host population size (GAN=30)
        self.gav = 20                 # Virus population size (GAV=20)
        self.gal = chromosome_length  # Chromosome length (GAL=10)
        
        # Robot parameters
        self.dof = 3                  # Degree of freedom (DOF=3)
        self.leg = 6                  # Number of legs (LEG=6)
        
        # Angle limits and ranges (in degrees, matching C++)
        self.q_min = np.array([-45, 0, 0])     # Min angle for target motion
        self.q_range = np.array([90, 60, 60])  # Range for target motion
        self.q_init = np.array([0, 45, 45])    # Init angle for target motion
        
        # Populations and fitness arrays (matching C++ structure)
        # Host population: sequence of postures
        self.hosts = np.zeros((self.gan, self.gal, 2, self.dof))
        # Virus population: individual joint angles
        self.virus = np.zeros((self.gav, self.dof))
        
        self.host_lengths = np.zeros(self.gan, dtype=int)  # Length of each sequence
        self.fitness = np.zeros((self.gan, 3))             # Host fitness for 3 objectives
        self.fitv = np.zeros((self.gav, 3))               # Virus fitness for 3 objectives
        
        # For VEGA ranking
        self.gac = np.full(self.gan, -1)               # Category assignments
        
        # Best fitness tracking
        self.iterations = 500
        self.bfith = np.zeros((self.iterations, 3))     # Best fitness history
        self.cfith = np.zeros((self.iterations, 3))     # Current fitness history
        self.bhostl = np.zeros((self.iterations, 3), dtype=int)  # Best host length history
        self.chostl = np.zeros(self.iterations, dtype=int)  # Current host length history
        
        # Current individual and sequence indices
        self.gai = 0      # Current host ID for simulation
        self.gaj = 0      # Current sequence ID
        self.iteration = 0
        
        # Evolutionary mode
        self.ERmode = 0   # 0: all, 1:Forward, 2:Right Forward, 3:Right
        
        # Initialize populations
        self.initialize_populations()
        
        # For current objective selection
        self.current_objective = None
        
        # Create necessary directories for experiment data
        os.makedirs(os.path.join(self.log_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'plots'), exist_ok=True)
        
        # Save experiment configuration
        self._save_config()
        
        self.logger.info(f"VEGA initialized with {self.gan} hosts, {self.gav} viruses, max sequence length {self.gal}")
    
    def _setup_logging(self):
        """Set up proper logging configuration."""
        self.logger = logging.getLogger('vega_evolution')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'evolution.log'))
        file_handler.setLevel(logging.INFO)
        
        # Console handler for basic info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _save_config(self):
        """Save experiment configuration to JSON file."""
        config = {
            'population_size': self.gan,
            'virus_population': self.gav,
            'chromosome_length': self.gal,
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
        
        self.logger.info(f"Configuration saved to {config_file}")
    
    def initialize_populations(self):
        """Initialize the populations of hosts and viruses."""
        # Initialize host population
        for i in range(self.gan):
            # Random sequence length between 2 and 4 (matching C++ implementation)
            self.host_lengths[i] = 2 + int(np.random.random() * 3)
            
            # Initialize each position in the sequence
            for m in range(self.host_lengths[i]):
                for phase in range(2):
                    for j in range(self.dof):
                        # Random angle within range
                        self.hosts[i, m, phase, j] = self.q_min[j] + self.q_range[j] * np.random.random()
        
        # Initialize virus population
        for i in range(self.gav):
            for j in range(self.dof):
                self.virus[i, j] = self.q_min[j] + self.q_range[j] * np.random.random()
        
        # Initialize fitness to zeros
        self.fitness = np.zeros((self.gan, 3))
        self.fitv = np.zeros((self.gav, 3))
        
        self.logger.info(f"Populations initialized with varying sequence lengths (2-4)")
    
    def rank(self):
        """
        Rank individuals based on fitness for multi-objective optimization.
        Direct port of VEGA_rank() from C++ implementation.
        """
        # Reset categories
        self.gac = np.full(self.gan, -1)
        
        # Assign individuals to objective categories
        for j in range(self.gan):
            # Determine objective based on index (cycling through objectives)
            h = j % 3
            
            # Find first unassigned individual
            k = 0
            while self.gac[k] != -1:
                k += 1
            
            # Find best unassigned individual for this objective
            for i in range(k+1, self.gan):
                if self.gac[i] == -1 and self.fitness[i, h] > self.fitness[k, h]:
                    k = i
            
            # Assign category
            self.gac[k] = h
        
        # Log rankings
        rank_str = "\nRankings:\n"
        for i in range(self.gan):
            rank_str += f"r[{i}]:{self.gac[i]}, {self.fitness[i, self.gac[i]]:.2f}\n"
        rank_str += "\n"
        
        self.logger.info(rank_str)
    
    def reverse(self, n):
        """
        Reverse the motion sequence for an individual.
        Direct port of VEGA_reverse() from C++ implementation.
        
        Args:
            n: Individual index to reverse
        """
        self.logger.info(f"Reverse motion sequence for individual {n}")
        
        # Reverse yaw (first DOF - leg angle) for each position in sequence
        for m in range(self.host_lengths[n]):
            for i in range(2):  # For both phases
                for j in range(0, self.dof, 3):  # Only modify first DOF (leg angle)
                    self.hosts[n, m, i, j] = -self.hosts[n, m, i, j]
    
    def exchange_lr(self, n):
        """
        Exchange left and right phases.
        Direct port of VEGA_LR() from C++ implementation.
        
        Args:
            n: Individual index to modify
        """
        self.logger.info(f"Phase exchange for individual {n}")
        
        # Exchange phases for each position in sequence
        for m in range(self.host_lengths[n]):
            for j in range(0, self.dof, 3):  # Only for first DOF (leg angle)
                # Swap phase 0 and phase 1
                d = self.hosts[n, m, 0, j]
                self.hosts[n, m, 0, j] = self.hosts[n, m, 1, j]
                self.hosts[n, m, 1, j] = d
    
    def evolve(self):
        """
        Perform one generation of evolution.
        Direct port of VEGA_main() from C++ implementation.
        """
        # First rank the population
        self.rank()
        
        # Determine which objective to focus on
        if self.ERmode == 0:
            h = self.iteration % 3      # Cycle through objectives
        else:
            h = self.ERmode - 1         # Use specific mode
            
        obj_names = ["Forward", "Right Forward", "Right Turn"]
        
        # Find worst and best individual for this objective
        g1 = 0  # Worst individual (to be replaced)
        while self.gac[g1] != h:
            g1 += 1
        g2 = g1  # Best individual (to copy from)
        
        for i in range(g1+1, self.gan):
            if self.gac[i] == h:
                if self.fitness[i, h] < self.fitness[g1, h]:   # Find worst
                    g1 = i  
                elif self.fitness[i, h] > self.fitness[g2, h]:  # Find best
                    g2 = i
        
        # Early generations: search for best solution
        if self.iteration < 100:
            self.logger.info(f"Search for {obj_names[h]}")
            
            # Random individual for crossover
            g3 = int(self.gan * np.random.random())
            r = np.random.random() * 0.5
            
            # Copy sequence length from best to worst
            self.host_lengths[g1] = self.host_lengths[g2]
            
            # Apply crossover and mutation
            for m in range(self.host_lengths[g1]):
                for i in range(2):  # Two phases
                    for j in range(self.dof):
                        if (np.random.random() < r) and (m < self.host_lengths[g3]):
                            # Crossover with random individual + mutation
                            self.hosts[g1, m, i, j] = (
                                self.hosts[g3, m, i, j] + 
                                self.randn() * self.q_range[j] * 0.2
                            )
                        else:
                            # Crossover with best individual + mutation
                            self.hosts[g1, m, i, j] = (
                                self.hosts[g2, m, i, j] + 
                                self.randn() * self.q_range[j] * 0.1
                            )
                        
                        # Enforce bounds
                        if self.hosts[g1, m, i, j] < self.q_min[j]:
                            self.hosts[g1, m, i, j] = self.q_min[j] + np.random.random() * 0.01
                        elif self.hosts[g1, m, i, j] > self.q_min[j] + self.q_range[j]:
                            self.hosts[g1, m, i, j] = self.q_min[j] + self.q_range[j] - np.random.random() * 0.01
            
            # Apply specialized mutations with some probability
            
            # 1. Insertion mutation (15% chance)
            if (self.host_lengths[g1] < self.gal - 1 and np.random.random() < 0.15):
                self.logger.info("-- insertion mutation --")
                k = int(self.host_lengths[g1] * np.random.random())
                
                if k < self.host_lengths[g1]:
                    # Shift all positions after insertion point
                    for m in range(self.host_lengths[g1], k, -1):
                        for i in range(2):
                            for j in range(self.dof):
                                self.hosts[g1, m, i, j] = self.hosts[g1, m-1, i, j]
                    
                    # Insert random posture
                    for i in range(2):
                        for j in range(self.dof):
                            self.hosts[g1, k, i, j] = self.q_min[j] + self.q_range[j] * np.random.random()
                
                self.host_lengths[g1] += 1
            
            # 2. Deletion mutation (15% chance)
            elif (self.host_lengths[g1] > 2 and np.random.random() < 0.15):
                self.logger.info("-- deletion mutation --")
                self.host_lengths[g1] -= 1
                k = int(self.host_lengths[g1] * np.random.random())
                
                if k < self.host_lengths[g1] - 1:
                    # Shift all positions after deletion point
                    for m in range(k, self.host_lengths[g1]):
                        for i in range(2):
                            for j in range(self.dof):
                                self.hosts[g1, m, i, j] = self.hosts[g1, m+1, i, j]
            
            # 3. Phase exchange mutation (10% chance)
            if np.random.random() < 0.1:
                self.logger.info("-- phase exchange mutation --")
                m = int(self.host_lengths[g1] * np.random.random())
                
                # Swap phases
                for j in range(self.dof):
                    d = self.hosts[g1, m, 0, j]
                    self.hosts[g1, m, 0, j] = self.hosts[g1, m, 1, j]
                    self.hosts[g1, m, 1, j] = d
            
            # 4. Order exchange mutation (10% chance)
            elif np.random.random() < 0.1:
                k = int(self.host_lengths[g1] * np.random.random())
                m = int(self.host_lengths[g1] * np.random.random())
                
                if k != m:
                    self.logger.info("-- order exchange mutation --")
                    # Swap positions in sequence
                    for i in range(2):
                        for j in range(self.dof):
                            d = self.hosts[g1, m, i, j]
                            self.hosts[g1, m, i, j] = self.hosts[g1, k, i, j]
                            self.hosts[g1, k, i, j] = d
            
            # Set current individual to evolved offspring
            self.gai = g1
            
        # Later generations: use best solution for given objective
        else:
            self.logger.info(f"Best Locomotion of {obj_names[h]}")
            self.gai = g2
        
        self.logger.info(f"Iterations: {self.iteration}, host: {self.gai}")
        
        # Save checkpoint occasionally
        if self.iteration % 10 == 0:
            self.save_checkpoint()
    
    def evaluate_fitness(self, robot, prev_pos, curr_pos, prev_rot, curr_rot):
        """
        Calculate fitness values for the current motion.
        Directly follows the fitness calculation in loco_main() from the C++ code.
        
        Args:
            robot: Robot instance
            prev_pos: Previous robot position
            curr_pos: Current robot position
            prev_rot: Previous rotation matrix
            curr_rot: Current rotation matrix
            
        Returns:
            Updated fitness array for the current individual
        """
        # Extract rotation angle (around z-axis) from rotation matrices
        if curr_rot[0, 0] == 0 and curr_rot[1, 0] == 0:
            ra = 0
        else:
            ra = math.atan2(curr_rot[1, 0], curr_rot[0, 0])
            
        # Previous angle
        if prev_rot[0, 0] == 0 and prev_rot[1, 0] == 0:
            rap = 0
        else:
            rap = math.atan2(prev_rot[1, 0], prev_rot[0, 0])
            
        # Calculate angle change
        a = ra - rap
        if a > math.pi:
            a -= 2 * math.pi
        elif a < -math.pi:
            a += 2 * math.pi
            
        # Calculate fitness metrics for each objective
        # 1. Forward motion - reward going straight
        f0 = math.exp(-a * a)
        # 2. Left turn - reward turning left
        f1 = math.exp(-(a - math.pi * 0.5) * (a - math.pi * 0.5))
        # 3. Right turn - reward turning right
        f2 = math.exp(-(a + math.pi * 0.5) * (a + math.pi * 0.5))
        
        # Get robot orientation (z-direction)
        posz = 1  # Default upright
        if curr_rot[2, 2] < -0.7:  # Check if robot is upside down
            posz = -1  # Flipped over
            
        # Calculate movement direction
        rr = np.array([curr_rot[0, 0], curr_rot[1, 0]])  # Current direction vector (x-axis)
        v = np.array([curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]])  # Movement vector
        
        # Calculate distance moved
        d = np.sqrt(np.sum(v * v))
        
        # Calculate alignment between direction and movement
        q = 0
        if d != 0:
            q = np.dot(rr, v) / d  # cosine of angle between direction and movement
            
        # Update fitness for the current individual for each objective
        # Forward motion: maximize distance and alignment
        self.fitness[self.gai, 0] = f0 + d * 10 + q
        # Left turn: reward turning left (note: identical to right turn in C++ code)
        self.fitness[self.gai, 1] = f1 * 20 + math.exp(-d * d)
        # Right turn: reward turning right
        self.fitness[self.gai, 2] = f2 * 20 + math.exp(-d * d)
        
        # Update fitness history
        for i in range(3):
            self.cfith[self.iteration, i] = self.fitness[self.gai, i]
            
        self.chostl[self.iteration] = self.host_lengths[self.gai]
        
        # Log fitness metrics
        self.logger.info(f"Walking distance: {d:.3f}, posture change: {q:.3f}, moving dir: {a:.3f}")
        self.logger.info(f"Current fit[0,F]: {self.fitness[self.gai, 0]:.3f}/{f0:.3f}, "
              f"fit[1,L]: {self.fitness[self.gai, 1]:.3f}/{f1:.3f}, "
              f"fit[2,R]: {self.fitness[self.gai, 2]:.3f}/{f2:.3f}, pos-z: {curr_rot[2, 2]:.2f}")
        
        # Find best individual for each objective
        if self.iteration < self.gan:
            h = self.iteration + 1
        else:
            h = self.gan
            
        for j in range(3):
            k = 0
            for i in range(h):
                if self.fitness[i, j] > self.fitness[k, j]:
                    k = i
            self.bfith[self.iteration, j] = self.fitness[k, j]
            self.bhostl[self.iteration, j] = self.host_lengths[k]
            
        self.logger.info(f"Best fit[0,F]: {self.bfith[self.iteration, 0]:.3f}, "
              f"fit[1,L]: {self.bfith[self.iteration, 1]:.3f}, "
              f"fit[2,R]: {self.bfith[self.iteration, 2]:.3f}")
        
        # Check if alignment is negative (moving backward) - if so, reverse the sequence
        if q < 0:
            self.logger.info(f"\n\n[{self.gai}] Reverse: InnerP: {q}, angle: {a}\n\n")
            self.reverse(self.gai)
            
        # Optionally, exchange left-right phases if angle is negative
        # This is commented out in the original C++ code
        # elif a < 0:
        #     self.logger.info(f"\n\n[{self.gai}] ExchangeLR InnerP: {q}, angle: {a}\n\n")
        #     self.exchange_lr(self.gai)
        
        # Save data occasionally
        if self.iteration % 10 == 0:
            self.save_fitness_data()
            
        return self.fitness[self.gai]
    
    def get_target_angles(self):
        """
        Get target angles for the robot's current sequence position.
        This follows the logic in loco_main() from the C++ code.
        
        Returns:
            Array of target angles for all legs
        """
        # Get current sequence position
        gaj = self.gaj % self.host_lengths[self.gai]
        
        # Create angles array (6 legs x 3 DOF)
        angles = np.zeros((6, 3))
        
        # Set target angles from current sequence position
        for i in range(6):  # 6 legs
            for j in range(3):  # 3 DOF
                # Determine which phase to use based on leg index
                if j == 0:  # First DOF (leg angle)
                    phase = 0 if i % 2 == 0 else 1  # Alternating phases for even/odd legs
                    angles[i, j] = np.radians(self.hosts[self.gai, gaj, phase, j])
                else:  # Other DOFs (middle and end joints)
                    phase = 0 if i % 2 == 0 else 1
                    # Apply signs based on which side of the robot
                    if i < 3:  # Right side
                        angles[i, j] = -np.radians(self.hosts[self.gai, gaj, phase, j])
                    else:  # Left side
                        angles[i, j] = np.radians(self.hosts[self.gai, gaj, phase, j])
        
        # Apply posz to handle flipped robot (if it flips over)
        posz = 1  # Upright by default
        # In a real implementation, this would be determined from the robot's orientation
        if posz != 1:
            angles = angles * posz
            
        return angles
    
    def create_controller(self):
        """
        Create a controller from the current individual.
        
        Returns:
            Controller dictionary with sequence information
        """
        controller = {
            'type': 'sequence_controller',
            'sequence_length': self.host_lengths[self.gai],
            'sequences': self.hosts[self.gai, :self.host_lengths[self.gai]].copy()
        }
        return controller
    
    def save_best_controller(self):
        """
        Save the best evolved controller for deployment.
        
        Returns:
            Path to the saved controller file
        """
        # Find best individual for forward movement
        best_idx = np.argmax(self.fitness[:, 0])
        
        controller = {
            'type': 'locomotion_controller',
            'sequence_length': self.host_lengths[best_idx],
            'sequences': self.hosts[best_idx, :self.host_lengths[best_idx]].copy(),
            'fitness': self.fitness[best_idx].copy(),
            'creation_date': time.strftime("%Y-%m-%d-%H:%M:%S"),
            'parameters': {
                'dof': self.dof,
                'q_min': self.q_min.tolist(),
                'q_range': self.q_range.tolist()
            },
            'experiment_id': self.experiment_id
        }
        
        # Save to file
        filename = os.path.join(self.log_dir, 'models', f"evolved_controller_{time.strftime('%Y%m%d_%H%M%S')}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(controller, f)
        
        self.logger.info(f"Best controller saved to {filename} for deployment")
        return filename
    
    def save_fitness_data(self):
        """
        Save fitness data to CSV file.
        
        Returns:
            Path to the saved CSV file and plot
        """
        # Create DataFrame with all relevant data
        data = {
            'iteration': range(self.iteration + 1),
            'best_forward': self.bfith[:self.iteration + 1, 0],
            'current_forward': self.cfith[:self.iteration + 1, 0],
            'best_right_forward': self.bfith[:self.iteration + 1, 1],
            'current_right_forward': self.cfith[:self.iteration + 1, 1],
            'best_right_turn': self.bfith[:self.iteration + 1, 2],
            'current_right_turn': self.cfith[:self.iteration + 1, 2],
            'best_host_length_forward': self.bhostl[:self.iteration + 1, 0],
            'best_host_length_right_forward': self.bhostl[:self.iteration + 1, 1],
            'best_host_length_right_turn': self.bhostl[:self.iteration + 1, 2],
            'current_host_length': self.chostl[:self.iteration + 1]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_filename = os.path.join(self.log_dir, 'data', f"evolution_data_{self.iteration:06d}.csv")
        df.to_csv(csv_filename, index=False)
        self.logger.info(f"Fitness data saved to {csv_filename}")
        
        # Generate plots
        plot_path = self._generate_fitness_plots(df)
        
        return csv_filename, plot_path
    
    def _generate_fitness_plots(self, df):
        """
        Generate fitness plots from the DataFrame.
        
        Args:
            df: DataFrame with fitness data
            
        Returns:
            Path to saved plot file
        """
        try:
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            
            # Forward fitness
            axs[0].plot(df['iteration'], df['best_forward'], 'b-', label='Best')
            axs[0].plot(df['iteration'], df['current_forward'], 'r--', label='Current')
            axs[0].set_title('Forward Fitness')
            axs[0].set_xlabel('Iteration')
            axs[0].set_ylabel('Fitness')
            axs[0].legend()
            axs[0].grid(True)
            
            # Right forward fitness
            axs[1].plot(df['iteration'], df['best_right_forward'], 'b-', label='Best')
            axs[1].plot(df['iteration'], df['current_right_forward'], 'r--', label='Current')
            axs[1].set_title('Right Forward Fitness')
            axs[1].set_xlabel('Iteration')
            axs[1].set_ylabel('Fitness')
            axs[1].legend()
            axs[1].grid(True)
            
            # Right turn fitness
            axs[2].plot(df['iteration'], df['best_right_turn'], 'b-', label='Best')
            axs[2].plot(df['iteration'], df['current_right_turn'], 'r--', label='Current')
            axs[2].set_title('Right Turn Fitness')
            axs[2].set_xlabel('Iteration')
            axs[2].set_ylabel('Fitness')
            axs[2].legend()
            axs[2].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.log_dir, 'plots', f"fitness_{self.iteration:06d}.png")
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Fitness plots saved to {plot_path}")
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
            return None
    
    def save_checkpoint(self):
        """
        Save a checkpoint of the algorithm's current state for resuming later.
        
        Returns:
            Path to the checkpoint file
        """
        checkpoint = {
            'iteration': self.iteration,
            'gan': self.gan,
            'gav': self.gav,
            'gal': self.gal,
            'hosts': self.hosts.copy(),
            'virus': self.virus.copy(),
            'host_lengths': self.host_lengths.copy(),
            'fitness': self.fitness.copy(),
            'fitv': self.fitv.copy(),
            'best_fitness': self.bfith.copy(),
            'current_fitness': self.cfith.copy(),
            'best_host_lengths': self.bhostl.copy(),
            'current_host_lengths': self.chostl.copy(),
            'gai': self.gai,
            'gaj': self.gaj,
            'ERmode': self.ERmode,
            'q_min': self.q_min,
            'q_range': self.q_range,
            'q_init': self.q_init,
            'timestamp': time.time(),
            'experiment_id': self.experiment_id
        }
        
        filename = os.path.join(self.log_dir, 'checkpoints', f"vega_checkpoint_{self.iteration:06d}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.logger.info(f"Checkpoint saved to {filename}")
        return filename
    
    @classmethod
    def load_checkpoint(cls, filename):
        """
        Load a saved checkpoint to resume evolution.
        
        Args:
            filename: Path to the checkpoint file
            
        Returns:
            VEGA instance loaded from checkpoint
        """
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create a new instance
        vega = cls(
            population_size=checkpoint['gan'],
            chromosome_length=checkpoint['gal'],
            generations=len(checkpoint['best_fitness'])
        )
        
        # Override log directory with the one from the checkpoint
        if 'experiment_id' in checkpoint:
            vega.experiment_id = checkpoint['experiment_id']
            vega.log_dir = os.path.join('logs', 'evolution', vega.experiment_id)
            vega._setup_logging()
        
        # Restore state from checkpoint
        vega.iteration = checkpoint['iteration']
        vega.hosts = checkpoint['hosts']
        if 'virus' in checkpoint:
            vega.virus = checkpoint['virus']
        vega.host_lengths = checkpoint['host_lengths']
        vega.fitness = checkpoint['fitness']
        if 'fitv' in checkpoint:
            vega.fitv = checkpoint['fitv']
        vega.bfith = checkpoint['best_fitness']
        vega.cfith = checkpoint['current_fitness']
        vega.bhostl = checkpoint['best_host_lengths']
        vega.chostl = checkpoint['current_host_lengths']
        vega.gai = checkpoint['gai']
        vega.gaj = checkpoint['gaj']
        if 'ERmode' in checkpoint:
            vega.ERmode = checkpoint['ERmode']
        
        vega.logger.info(f"Loaded checkpoint from iteration {vega.iteration}")
        return vega
    
    @staticmethod
    def load_controller(filename):
        """
        Load a previously saved controller for deployment.
        
        Args:
            filename: Path to the controller file
            
        Returns:
            Loaded controller
        """
        with open(filename, 'rb') as f:
            controller = pickle.load(f)
        
        print(f"Loaded controller from {filename}")
        print(f"Sequence length: {controller['sequence_length']}")
        if 'fitness' in controller:
            print(f"Fitness values: {controller['fitness']}")
        
        return controller
    
    def plot_fitness_history(self):
        """
        Plot the complete fitness history of all objectives.
        
        Returns:
            Path to the saved plot
        """
        # Create figure
        plt.figure(figsize=(12, 15))
        
        # Plot all objectives
        objectives = ["Forward", "Right Forward", "Right Turn"]
        for i in range(3):
            plt.subplot(4, 1, i+1)
            plt.plot(self.bfith[:self.iteration+1, i], 'b-', label=f'Best {objectives[i]}')
            plt.plot(self.cfith[:self.iteration+1, i], 'r--', label=f'Current {objectives[i]}')
            plt.legend()
            plt.grid(True)
            plt.ylabel('Fitness')
            plt.title(f'{objectives[i]} Fitness')
        
        # Plot sequence lengths
        plt.subplot(4, 1, 4)
        for i in range(3):
            plt.plot(self.bhostl[:self.iteration+1, i], '-', label=f'Best {objectives[i]} Length')
        plt.plot(self.chostl[:self.iteration+1], 'k--', label='Current Length')
        plt.legend()
        plt.grid(True)
        plt.ylabel('Sequence Length')
        plt.xlabel('Generation')
        plt.title('Evolution of Sequence Lengths')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.log_dir, 'plots', f"complete_fitness_history.png")
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Complete fitness history plot saved to {plot_path}")
        return plot_path
    
    def save_summary(self):
        """Save a summary of the evolution run."""
        summary = {
            'experiment_id': self.experiment_id,
            'total_iterations': self.iteration,
            'best_fitness_forward': float(np.max(self.bfith[:self.iteration+1, 0])),
            'best_fitness_right_forward': float(np.max(self.bfith[:self.iteration+1, 1])),
            'best_fitness_right_turn': float(np.max(self.bfith[:self.iteration+1, 2])),
            'best_individual_forward': int(np.argmax(self.fitness[:, 0])),
            'best_individual_right_forward': int(np.argmax(self.fitness[:, 1])),
            'best_individual_right_turn': int(np.argmax(self.fitness[:, 2])),
            'end_time': datetime.now().isoformat(),
            'total_runtime_seconds': time.time() - os.path.getctime(os.path.join(self.log_dir, 'config.json'))
        }
        
        # Save summary
        summary_path = os.path.join(self.log_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Evolution summary saved to {summary_path}")
        return summary_path
    
    @staticmethod
    def randn():
        """
        Generate a random number from normal distribution.
        Direct port of rndn() from the C++ code.
        """
        return (np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() - 6.0)