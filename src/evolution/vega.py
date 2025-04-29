import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
import math


class VEGA:
    """
    Virus-Host coEvolutionary Genetic Algorithm (VEGA) implementation
    for multi-objective optimization of robot locomotion patterns.
    This is a direct port of the C++ VEGA implementation from the original ODE codebase.
    """
    
    def __init__(self, population_size=30, chromosome_length=10, generations=500):
        """
        Initialize the VEGA algorithm with parameters matching the C++ implementation.
        
        Args:
            population_size: Number of individuals in the population (GAN in C++)
            chromosome_length: Maximum length of locomotion sequences (GAL in C++)
            generations: Maximum number of generations to evolve
        """
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
        
        # Print rankings (for debugging)
        print("\nRankings:")
        for i in range(self.gan):
            print(f"r[{i}]:{self.gac[i]}, {self.fitness[i, self.gac[i]]:.2f}")
        print("\n")
    
    def reverse(self, n):
        """
        Reverse the motion sequence for an individual.
        Direct port of VEGA_reverse() from C++ implementation.
        
        Args:
            n: Individual index to reverse
        """
        print(f"Reverse: {n}")
        
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
        print(f"Phase Change: {n}")
        
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
            print(f"Search for {obj_names[h]}")
            
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
                print("-- insertion mutation --")
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
                print("-- deletion mutation --")
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
                print("-- phase exchange mutation --")
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
                    print("-- order exchange mutation --")
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
            print(f"Best Locomotion of {obj_names[h]}")
            self.gai = g2
        
        print(f"Iterations: {self.iteration}, host: {self.gai}")
    
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
        
        print(f"Walking distance: {d:.3f}, posture change: {q:.3f}, moving dir: {a:.3f}")
        print(f"Current fit[0,F]: {self.fitness[self.gai, 0]:.3f}/{f0:.3f}, "
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
            
        print(f"Best fit[0,F]: {self.bfith[self.iteration, 0]:.3f}, "
              f"fit[1,L]: {self.bfith[self.iteration, 1]:.3f}, "
              f"fit[2,R]: {self.bfith[self.iteration, 2]:.3f}")
        
        # Check if alignment is negative (moving backward) - if so, reverse the sequence
        if q < 0:
            print(f"\n\n[{self.gai}] Reverse: InnerP: {q}, angle: {a}\n\n")
            self.reverse(self.gai)
            
        # Optionally, exchange left-right phases if angle is negative
        # This is commented out in the original C++ code
        # elif a < 0:
        #     print(f"\n\n[{self.gai}] ExchangeLR InnerP: {q}, angle: {a}\n\n")
        #     self.exchange_lr(self.gai)
            
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
        """Create a controller from the current individual."""
        controller = {
            'type': 'sequence_controller',
            'sequence_length': self.host_lengths[self.gai],
            'sequences': self.hosts[self.gai, :self.host_lengths[self.gai]].copy()
        }
        return controller
    
    def save_data(self):
        """
        Save evolution data to a file.
        This is equivalent to writedata() in the C++ code.
        """
        filename = f"data{self.iteration // 100:03d}.txt"
        with open(filename, "w") as f:
            for i in range(self.iteration + 1):
                for j in range(3):
                    f.write(f"{self.bfith[i, j]:.6f}\t{self.cfith[i, j]:.6f}\t"
                           f"{self.bhostl[i, j]}\t{self.chostl[i]}\t")
                f.write("\n")
        print(f"DATA write end: {filename}")
        
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