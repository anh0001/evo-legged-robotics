import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import pickle
import os
import time


class VEGA:
    """
    Vector Evaluated Genetic Algorithm (VEGA) implementation
    for multi-objective optimization of robot locomotion patterns.
    
    This is a Python port of the C++ VEGA implementation from
    the original ODE codebase.
    """
    
    def __init__(self, population_size=30, chromosome_length=10, generations=500,
                 mutation_rate=0.1, crossover_rate=0.8, 
                 fitness_objectives=None, max_leg_sequence_length=10):
        """
        Initialize the VEGA algorithm.
        
        Args:
            population_size: Number of individuals in the population
            chromosome_length: Initial length of locomotion sequences
            generations: Maximum number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            crossover_rate: Probability of crossover between parents
            fitness_objectives: List of fitness objectives to optimize
            max_leg_sequence_length: Maximum allowed length for leg sequences
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_leg_sequence_length = max_leg_sequence_length
        
        # Set default fitness objectives if none provided
        if fitness_objectives is None:
            self.fitness_objectives = ["forward_speed", "turning_left", "turning_right"]
        else:
            self.fitness_objectives = fitness_objectives
        
        self.num_objectives = len(self.fitness_objectives)
        
        # Population initialization
        self.hosts = None  # Will store the population
        self.host_lengths = np.zeros(population_size, dtype=int)  # Length of each sequence
        self.fitness = np.zeros((population_size, self.num_objectives))
        
        # Tracking best individuals and history
        self.best_fitness = np.zeros((generations, self.num_objectives))
        self.current_fitness = np.zeros((generations, self.num_objectives))
        self.best_host_lengths = np.zeros((generations, self.num_objectives), dtype=int)
        self.current_host_lengths = np.zeros(generations, dtype=int)
        
        # Rankings for multi-objective optimization
        self.ranking = np.zeros((population_size, self.num_objectives))
        self.category = np.full(population_size, -1)
        
        # Current generation and evaluation counters
        self.generation = 0
        self.current_individual = 0
        self.current_sequence = 0
        
        # Initialize population
        self.initialize_population()
    
    def initialize_population(self):
        """
        Initialize the population with random locomotion sequences.
        Each individual has a sequence of postures for leg control.
        """
        # Create a structured array for hosts
        # Each host has a locomotion sequence of variable length
        # Each sequence has postures for both leg phases and DOF angles
        
        # First dimension: population size
        # Second dimension: max sequence length
        # Third dimension: 2 phases (left/right)
        # Fourth dimension: degrees of freedom per leg (typically 3)
        self.hosts = np.zeros((self.population_size, self.max_leg_sequence_length, 2, 3))
        
        # Initialize q_min and q_range (min angles and range for each DOF)
        q_min = np.array([-45, 0, 0])
        q_range = np.array([90, 60, 60])
        
        # For each individual in the population
        for i in range(self.population_size):
            # Randomly determine sequence length (2-5 initially)
            self.host_lengths[i] = 2 + int(np.random.random() * 4)
            
            # For each position in the sequence
            for j in range(self.host_lengths[i]):
                # For each phase (0=right, 1=left)
                for phase in range(2):
                    # For each degree of freedom
                    for dof in range(3):
                        # Random angle within range
                        self.hosts[i, j, phase, dof] = q_min[dof] + q_range[dof] * np.random.random()
        
        # Initialize fitness to zeros
        self.fitness = np.zeros((self.population_size, self.num_objectives))
    
    def rank(self):
        """
        Rank individuals based on each objective separately.
        This implements the multi-objective ranking for VEGA.
        """
        # Reset categories
        self.category = np.full(self.population_size, -1)
        
        # For each population member, assign to objective category
        for i in range(self.population_size):
            # Assign each individual to an objective in a round-robin fashion
            obj_index = i % self.num_objectives
            
            # Find the best unassigned individual for this objective
            best_idx = -1
            best_fitness = -float('inf')
            
            for j in range(self.population_size):
                if self.category[j] == -1 and self.fitness[j, obj_index] > best_fitness:
                    best_idx = j
                    best_fitness = self.fitness[j, obj_index]
            
            # Assign category
            self.category[best_idx] = obj_index
        
        # Print rankings (for debugging)
        print("\nRankings:")
        for i in range(self.population_size):
            obj_idx = self.category[i]
            if obj_idx >= 0:
                print(f"r[{i}]:{obj_idx}, {self.fitness[i, obj_idx]:.2f}")
        print()
    
    def evolve(self):
        """
        Perform one generation of evolution using VEGA.
        """
        # First rank the population
        self.rank()
        
        # Determine which objective to focus on for this generation
        objective_idx = self.generation % self.num_objectives
        objective_name = self.fitness_objectives[objective_idx]
        
        # Find worst and best individuals for this objective
        worst_idx = -1
        best_idx = -1
        
        # Find initial indices
        for i in range(self.population_size):
            if self.category[i] == objective_idx:
                worst_idx = i
                best_idx = i
                break
        
        # Find actual worst and best
        for i in range(self.population_size):
            if self.category[i] == objective_idx:
                if self.fitness[i, objective_idx] < self.fitness[worst_idx, objective_idx]:
                    worst_idx = i
                elif self.fitness[i, objective_idx] > self.fitness[best_idx, objective_idx]:
                    best_idx = i
        
        # Early generations: search for best solution
        if self.generation < 100:
            print(f"Search for {objective_name}")
            
            # Pick a random individual for crossover
            random_idx = int(self.population_size * np.random.random())
            crossover_rate = np.random.random() * 0.5
            
            # Copy sequence length from best individual to worst
            self.host_lengths[worst_idx] = self.host_lengths[best_idx]
            
            # For each position in the sequence
            for pos in range(self.host_lengths[worst_idx]):
                # For each phase
                for phase in range(2):
                    # For each DOF
                    for dof in range(3):
                        # Crossover and mutation
                        if np.random.random() < crossover_rate and pos < self.host_lengths[random_idx]:
                            # Crossover with random individual + mutation
                            self.hosts[worst_idx, pos, phase, dof] = (
                                self.hosts[random_idx, pos, phase, dof] + 
                                self.randn() * 0.2 * 90  # Using q_range[0] as estimate
                            )
                        else:
                            # Crossover with best individual + mutation
                            self.hosts[worst_idx, pos, phase, dof] = (
                                self.hosts[best_idx, pos, phase, dof] + 
                                self.randn() * 0.1 * 90
                            )
                        
                        # Enforce bounds
                        if self.hosts[worst_idx, pos, phase, dof] < -45:
                            self.hosts[worst_idx, pos, phase, dof] = -45 + np.random.random() * 0.01
                        elif self.hosts[worst_idx, pos, phase, dof] > -45 + 90:
                            self.hosts[worst_idx, pos, phase, dof] = -45 + 90 - np.random.random() * 0.01
            
            # Apply specialized mutations with some probability
            
            # 1. Insertion mutation (15% chance)
            if (self.host_lengths[worst_idx] < self.max_leg_sequence_length - 1 and 
                np.random.random() < 0.15):
                print("-- insertion mutation --")
                pos = int(self.host_lengths[worst_idx] * np.random.random())
                
                if pos < self.host_lengths[worst_idx]:
                    # Shift all positions after insertion point
                    for i in range(self.host_lengths[worst_idx], pos, -1):
                        for phase in range(2):
                            for dof in range(3):
                                self.hosts[worst_idx, i, phase, dof] = self.hosts[worst_idx, i-1, phase, dof]
                    
                    # Insert random posture
                    for phase in range(2):
                        for dof in range(3):
                            self.hosts[worst_idx, pos, phase, dof] = -45 + 90 * np.random.random()
                
                self.host_lengths[worst_idx] += 1
            
            # 2. Deletion mutation (15% chance)
            elif (self.host_lengths[worst_idx] > 2 and 
                 np.random.random() < 0.15):
                print("-- deletion mutation --")
                self.host_lengths[worst_idx] -= 1
                pos = int(self.host_lengths[worst_idx] * np.random.random())
                
                if pos < self.host_lengths[worst_idx] - 1:
                    # Shift all positions after deletion point
                    for i in range(pos, self.host_lengths[worst_idx]):
                        for phase in range(2):
                            for dof in range(3):
                                self.hosts[worst_idx, i, phase, dof] = self.hosts[worst_idx, i+1, phase, dof]
            
            # 3. Phase exchange mutation (10% chance)
            if np.random.random() < 0.1:
                print("-- phase exchange mutation --")
                pos = int(self.host_lengths[worst_idx] * np.random.random())
                
                # Swap phases
                for dof in range(3):
                    temp = self.hosts[worst_idx, pos, 0, dof]
                    self.hosts[worst_idx, pos, 0, dof] = self.hosts[worst_idx, pos, 1, dof]
                    self.hosts[worst_idx, pos, 1, dof] = temp
            
            # 4. Order exchange mutation (10% chance)
            elif np.random.random() < 0.1:
                pos1 = int(self.host_lengths[worst_idx] * np.random.random())
                pos2 = int(self.host_lengths[worst_idx] * np.random.random())
                
                if pos1 != pos2:
                    print("-- order exchange mutation --")
                    # Swap positions in sequence
                    for phase in range(2):
                        for dof in range(3):
                            temp = self.hosts[worst_idx, pos1, phase, dof]
                            self.hosts[worst_idx, pos1, phase, dof] = self.hosts[worst_idx, pos2, phase, dof]
                            self.hosts[worst_idx, pos2, phase, dof] = temp
            
            # Set current individual to evolved offspring
            self.current_individual = worst_idx
        
        # Later generations: use best solution for given objective
        else:
            print(f"Best Locomotion for {objective_name}")
            self.current_individual = best_idx
    
    def evaluate_fitness(self, robot, environment, individual_idx):
        """
        Evaluate the fitness of an individual using the simulator.
        
        Args:
            robot: Robot instance to be controlled
            environment: Simulation environment
            individual_idx: Index of individual to evaluate
            
        Returns:
            Fitness values array for the individual
        """
        # Reset the environment and robot
        environment.reset()
        robot.reset_posture()
        
        # Initial position and orientation
        initial_pos = np.array(robot.get_position())
        initial_orientation = robot.get_orientation()
        prev_pos = initial_pos.copy()
        prev_direction = np.array([1, 0, 0])  # Initial direction vector (x-axis)
        
        # For each motion sequence in the individual's genome
        sequence_idx = 0
        max_sequences = 20  # Number of times to repeat the motion sequence
        
        # Run simulation for fixed number of steps
        for step in range(max_sequences * 50):  # 50 steps per sequence
            # Get current sequence position
            current_pos = sequence_idx % self.host_lengths[individual_idx]
            
            # Apply leg positions from genome
            angles = np.zeros((6, 3))  # 6 legs, 3 DOF each
            
            # Set target angles for legs based on current genome position
            for leg in range(6):
                for dof in range(3):
                    # Right side legs (first 3)
                    if leg < 3:
                        if leg % 2 == 0:  # Even legs use phase 0
                            angles[leg, dof] = -np.radians(self.hosts[individual_idx, current_pos, 0, dof])
                        else:  # Odd legs use phase 1
                            angles[leg, dof] = -np.radians(self.hosts[individual_idx, current_pos, 1, dof])
                    # Left side legs (last 3)
                    else:
                        if leg % 2 == 0:  # Even legs use phase 0
                            angles[leg, dof] = np.radians(self.hosts[individual_idx, current_pos, 0, dof])
                        else:  # Odd legs use phase 1
                            angles[leg, dof] = np.radians(self.hosts[individual_idx, current_pos, 1, dof])
            
            # Set target angles on robot
            robot.set_target_angles(angles)
            
            # Step simulation
            for _ in range(5):  # Run multiple physics steps per control step
                environment.step()
            
            # Move to next sequence position after a while
            if step % 50 == 49:
                sequence_idx += 1
            
            # If we've completed all sequences, evaluate fitness
            if sequence_idx >= max_sequences:
                break
        
        # Calculate fitness based on final position and orientation
        final_pos = np.array(robot.get_position())
        state = robot.get_state()
        rotation_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
        
        # Extract rotation angle around z-axis (yaw)
        current_direction = rotation_matrix[:3, 0]  # First column is x-axis direction
        
        # Calculate angle between initial and final direction
        angle_change = np.arctan2(current_direction[1], current_direction[0])
        
        # Adjust angle to range [-pi, pi]
        if angle_change > np.pi:
            angle_change -= 2 * np.pi
        elif angle_change < -np.pi:
            angle_change += 2 * np.pi
        
        # Calculate displacement
        displacement = final_pos - initial_pos
        distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
        
        # Calculate direction alignment (dot product between displacement and direction)
        direction_alignment = 0
        if distance != 0:
            direction_alignment = (current_direction[0] * displacement[0] + 
                                   current_direction[1] * displacement[1]) / distance
        
        # Calculate fitness values for each objective
        fitness = np.zeros(self.num_objectives)
        
        # 1. Forward speed fitness: reward distance and alignment
        fitness[0] = np.exp(-angle_change**2) + distance * 10 + direction_alignment
        
        # 2. Left turn fitness: reward left turning
        fitness[1] = np.exp(-(angle_change + np.pi/2)**2) + np.exp(-distance**2)
        
        # 3. Right turn fitness: reward right turning
        fitness[2] = np.exp(-(angle_change - np.pi/2)**2) + np.exp(-distance**2)
        
        return fitness
    
    def evaluate_population(self, robot, environment, parallel=False):
        """
        Evaluate fitness for the entire population.
        
        Args:
            robot: Robot instance to be controlled
            environment: Simulation environment
            parallel: Whether to use parallel processing for evaluation
        """
        if parallel:
            # Using parallel processing for faster evaluation
            with ProcessPoolExecutor() as executor:
                # Create a list of futures
                futures = [
                    executor.submit(self.evaluate_fitness, robot, environment, i)
                    for i in range(self.population_size)
                ]
                
                # Get results as they complete
                for i, future in enumerate(futures):
                    self.fitness[i] = future.result()
        else:
            # Sequential evaluation
            for i in range(self.population_size):
                self.fitness[i] = self.evaluate_fitness(robot, environment, i)
                print(f"Individual {i}: Fitness = {self.fitness[i]}")
    
    def train(self, robot, environment, parallel=False):
        """
        Main training loop for the evolutionary algorithm.
        
        Args:
            robot: Robot instance to be controlled
            environment: Simulation environment
            parallel: Whether to use parallel processing for evaluation
            
        Returns:
            Best controller from the final generation
        """
        start_time = time.time()
        
        for generation in range(self.generations):
            self.generation = generation
            
            # Evaluate fitness of entire population
            self.evaluate_population(robot, environment, parallel)
            
            # Store current fitness
            obj_idx = generation % self.num_objectives
            self.current_fitness[generation, obj_idx] = self.fitness[self.current_individual, obj_idx]
            self.current_host_lengths[generation] = self.host_lengths[self.current_individual]
            
            # Find best fitness for each objective
            for obj in range(self.num_objectives):
                best_idx = np.argmax(self.fitness[:, obj])
                self.best_fitness[generation, obj] = self.fitness[best_idx, obj]
                self.best_host_lengths[generation, obj] = self.host_lengths[best_idx]
            
            # Print progress
            print(f"Generation {generation}:")
            print(f"Current fitness: {self.current_fitness[generation]}")
            print(f"Best fitness: {self.best_fitness[generation]}")
            
            # Save progress every 10 generations
            if generation % 10 == 0:
                self.save_checkpoint(f"checkpoint_gen_{generation}.pkl")
            
            # Stop if we've reached the target
            if self.check_termination():
                print("Termination criteria met. Stopping evolution.")
                break
            
            # Evolve to next generation
            self.evolve()
        
        # Final evaluation
        self.evaluate_population(robot, environment, parallel)
        
        # Calculate total training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Return best controller
        return self.create_controller()
    
    def check_termination(self):
        """Check if termination criteria are met."""
        # For now, just run for all generations
        return False
    
    def create_controller(self):
        """Create a controller from the best individual."""
        # Find best individual
        best_idx = np.argmax(np.mean(self.fitness, axis=1))
        
        # Create a controller
        # This is simplified - in practice you might create a more complex controller
        controller = {
            'type': 'sequence_controller',
            'sequence_length': self.host_lengths[best_idx],
            'sequences': self.hosts[best_idx, :self.host_lengths[best_idx]].copy()
        }
        
        return controller
    
    def save_checkpoint(self, filename):
        """Save the current state of the evolution."""
        checkpoint = {
            'generation': self.generation,
            'hosts': self.hosts.copy(),
            'host_lengths': self.host_lengths.copy(),
            'fitness': self.fitness.copy(),
            'best_fitness': self.best_fitness.copy(),
            'current_fitness': self.current_fitness.copy(),
            'best_host_lengths': self.best_host_lengths.copy(),
            'current_host_lengths': self.current_host_lengths.copy()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved to {filename}")
    
    @staticmethod
    def load_checkpoint(filename):
        """Load a saved checkpoint."""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create a new VEGA instance
        vega = VEGA(
            population_size=checkpoint['hosts'].shape[0],
            chromosome_length=checkpoint['host_lengths'][0],
            generations=checkpoint['best_fitness'].shape[0]
        )
        
        # Restore state
        vega.generation = checkpoint['generation']
        vega.hosts = checkpoint['hosts']
        vega.host_lengths = checkpoint['host_lengths']
        vega.fitness = checkpoint['fitness']
        vega.best_fitness = checkpoint['best_fitness']
        vega.current_fitness = checkpoint['current_fitness']
        vega.best_host_lengths = checkpoint['best_host_lengths']
        vega.current_host_lengths = checkpoint['current_host_lengths']
        
        return vega
    
    def plot_fitness_history(self):
        """Plot the fitness history."""
        plt.figure(figsize=(12, 8))
        
        # Plot all objectives
        for i in range(self.num_objectives):
            plt.subplot(self.num_objectives, 1, i+1)
            plt.plot(self.best_fitness[:self.generation+1, i], 'b-', label=f'Best {self.fitness_objectives[i]}')
            plt.plot(self.current_fitness[:self.generation+1, i], 'r--', label=f'Current {self.fitness_objectives[i]}')
            plt.legend()
            plt.grid(True)
            plt.ylabel('Fitness')
            if i == self.num_objectives - 1:
                plt.xlabel('Generation')
        
        plt.tight_layout()
        plt.savefig('fitness_history.png')
        plt.show()
    
    @staticmethod
    def randn():
        """Generate a random number from a normal distribution."""
        # Simple implementation using the central limit theorem
        return (np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() - 6.0)