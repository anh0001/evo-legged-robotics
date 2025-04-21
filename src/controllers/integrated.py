import numpy as np
import tensorflow as tf
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json
import pickle

class IntegratedController:
    """
    Integrated controller that combines SSGA, VEGA and neural network learning
    for robot locomotion in varying terrain conditions. This implements the
    comprehensive approach from main06.cpp.
    
    Key features:
    1. SSGA for standard locomotion generation
    2. Neural network adaptation with incremental learning
    3. VEGA for rough terrain locomotion
    4. Comprehensive data collection and analysis
    """
    
    def __init__(self, input_dim=15, hidden_dim=30, output_dim=12, log_dir=None):
        """
        Initialize the integrated controller.
        
        Args:
            input_dim: Dimension of neural network input (joint angles + orientation)
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output (target joint angles)
            log_dir: Directory to save logs and data
        """
        # Setup logging
        self.log_dir = log_dir or os.path.join('logs', 'integrated', 
                                            datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        self._setup_logging()
        
        # Neural network parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Learning parameters (matches main06.cpp)
        self.learning_rate = 0.01  # "a" parameter
        self.bias_learning_rate = 0.005  # "b" parameter
        self.momentum = 0.2  # "c" parameter
        self.dropout_rate = 1.0  # "d" parameter
        
        # Neural network weights and deltas
        self.nn_weights = [None] * 2  # Input→hidden and hidden→output weights
        self.nn_biases = [None] * 2   # Hidden and output biases
        self.nn_deltas = [None] * 2   # For momentum-based updates
        self.initialize_nn_weights()
        
        # Build TensorFlow model for batch learning
        self.model = self._build_model()
        
        # Evolutionary algorithm parameters
        self.gan = 30  # Host population size
        self.gav = 100  # Virus population size
        self.gal = 10  # Chromosome length (motion sequences)
        
        # Locomotion parameters
        self.dof = 3       # Degrees of freedom per leg
        self.leg_count = 6  # Number of legs
        
        # Min/max angles and ranges (from main06.cpp)
        self.q_min = np.array([-45, 0, 0])  # Min angles in degrees
        self.q_range = np.array([90, 60, 60])  # Angle ranges in degrees
        self.q_init = np.array([0, 45, 45])  # Initial angles in degrees
        
        # Initialize populations for evolutionary algorithms
        self.initialize_populations()
        
        # For tracking leg heights and robot orientation
        self.leg_heights = np.zeros(4)  # Heights of 4 corner legs
        self.min_leg_height = 0.0
        self.current_z_dir = 1.0
        self.previous_z_dir = 1.0
        
        # Current and target angles
        self.current_angles = np.zeros((4, 3))  # 4 active legs, 3 DOF each
        self.target_angles = np.zeros((6, 3))  # All 6 legs, 3 DOF each
        
        # Control mode
        self.use_neural = False
        self.phase_counter = 0
        self.iteration = 0
        self.gai = 0  # Current individual ID
        self.gaj = 0  # Current sequence ID
        
        # For dataset management
        self.dataset = []
        self.max_samples = 50  # Maximum dataset size
        self.num_samples = 0
        
        # For tracking training data
        self.training_data = pd.DataFrame(columns=[
            'iteration', 'timestamp', 'loss', 'z_direction', 
            'inputs', 'targets', 'outputs', 'errors'
        ])
        
        # Setup TensorBoard for visualization
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        
        self.logger.info(f"Initialized IntegratedController with dimensions: "
                        f"input={input_dim}, hidden={hidden_dim}, output={output_dim}")
    
    def _setup_logging(self):
        """Set up logging configuration with proper formatting."""
        self.logger = logging.getLogger('integrated_controller')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, 'controller.log'))
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def initialize_nn_weights(self):
        """Initialize neural network weights with small random values."""
        # Initialize weights and biases
        # Input to hidden weights
        self.nn_weights[0] = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.nn_biases[0] = np.random.randn(self.hidden_dim) * 0.1
        self.nn_deltas[0] = np.zeros((self.input_dim, self.hidden_dim))
        
        # Hidden to output weights
        self.nn_weights[1] = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        self.nn_biases[1] = np.random.randn(self.output_dim) * 0.1
        self.nn_deltas[1] = np.zeros((self.hidden_dim, self.output_dim))
        
        self.logger.debug("Neural network weights initialized")
    
    def _build_model(self):
        """Build the TensorFlow model for more efficient batch learning."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='sigmoid', 
                                 input_shape=(self.input_dim,)),
            tf.keras.layers.Dense(self.output_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate,
                momentum=self.momentum
            ),
            loss='mse'
        )
        
        return model
    
    def initialize_populations(self):
        """Initialize populations for evolutionary algorithms (SSGA and VEGA)."""
        # Initialize SSGA host population (sequence-based controllers)
        self.hosts = np.zeros((self.gan, self.gal, 2, self.dof))
        self.host_lengths = np.zeros(self.gan, dtype=int)
        self.fitness = np.zeros((self.gan, 3))  # 3 objectives: forward, left turn, right turn
        
        # Set initial random sequence lengths and angles
        for i in range(self.gan):
            self.host_lengths[i] = 2 + int(np.random.random() * 3)  # Length 2-4
            for j in range(self.host_lengths[i]):
                for phase in range(2):
                    for dof in range(self.dof):
                        self.hosts[i, j, phase, dof] = self.q_min[dof] + self.q_range[dof] * np.random.random()
        
        # Initialize VEGA population for rough terrain
        self.rvirus = np.zeros((self.gav, self.dof))
        self.rhost = np.zeros((self.gan, self.gal, self.leg_count), dtype=int)
        self.rhostl = np.zeros(self.gan, dtype=int)
        self.fitv = np.zeros((self.gav, 3))
        
        # Rankings for multi-objective optimization
        self.ranking = np.zeros((self.gan, 3))
        self.category = np.full(self.gan, -1)
        
        # Best fitness tracking
        self.bfith = np.zeros((1000, 3))  # For tracking best fitness over time
        self.cfith = np.zeros((1000, 3))  # Current fitness
        self.bhostl = np.zeros((1000, 3), dtype=int)
        self.chostl = np.zeros(1000, dtype=int)
        
        self.logger.info("Evolutionary algorithm populations initialized")
    
    def get_actions(self, state):
        """
        Get joint target angles based on current state.
        
        Args:
            state: Current robot state
            
        Returns:
            Target joint angles for all 6 legs
        """
        # Update orientation and leg height tracking
        self.update_orientation(state)
        self.update_leg_heights(state)
        
        # Increment phase counter
        self.phase_counter += 1
        
        # Choose between neural network and evolutionary control
        if self.phase_counter >= 50:  # Same as samstep in main06.cpp
            self.phase_counter = 0
            
            # Every 50 steps, decide whether to use neural network
            # Roughly follows the logic in loco_main2() from main06.cpp
            if self.current_z_dir < 0.8:  # Robot is tilting significantly
                self.use_neural = True
            elif self.current_z_dir > 0.95:  # Robot is mostly upright
                self.use_neural = False
        
        # Get target angles based on control mode
        if self.use_neural:
            self.logger.debug("Using neural network for control")
            return self.predict(state)
        else:
            self.logger.debug("Using evolutionary algorithm for control")
            return self.get_evolutionary_angles(state)
    
    def update_orientation(self, state):
        """Update tracking of vertical orientation."""
        if 'rotation_matrix' in state:
            rot_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
            self.previous_z_dir = self.current_z_dir
            self.current_z_dir = rot_matrix[2, 2]  # z-component of z-axis (vertical)
            
            # Log the orientation change
            if hasattr(self, 'tf_writer'):
                with self.tf_writer.as_default():
                    tf.summary.scalar('orientation/z_direction', self.current_z_dir, 
                                     step=len(self.training_data))
    
    def update_leg_heights(self, state):
        """Update tracking of leg heights for terrain adaptation."""
        if 'leg_positions' in state:
            leg_positions = state['leg_positions']
            # Get heights of the 4 corner legs (0, 2, 3, 5)
            corner_legs = [0, 2, 3, 5]
            for i, leg_idx in enumerate(corner_legs):
                if leg_idx < len(leg_positions):
                    self.leg_heights[i] = leg_positions[leg_idx][2]  # z-coordinate
            
            # Calculate minimum height for normalization
            self.min_leg_height = np.min(self.leg_heights)
            
            # Normalize heights relative to minimum
            for i in range(len(self.leg_heights)):
                self.leg_heights[i] -= self.min_leg_height
            
            # Log leg heights
            if hasattr(self, 'tf_writer'):
                with self.tf_writer.as_default():
                    for i, height in enumerate(self.leg_heights):
                        tf.summary.scalar(f'leg_heights/leg_{i}', height, 
                                         step=len(self.training_data))
    
    def get_evolutionary_angles(self, state):
        """
        Get target angles from the evolutionary algorithm.
        
        Args:
            state: Current robot state
            
        Returns:
            Target angles for all 6 legs
        """
        # Follows the logic from loco_main() in main06.cpp
        
        self.gaj += 1
        
        # Move to next individual if we've gone through all sequences
        if self.gaj >= self.host_lengths[self.gai]:
            self.gaj = 0
            self.iteration += 1
            
            # Evolve population every few iterations
            if self.iteration % 10 == 0:
                self.ssga_evolve()
        
        # Get target angles from current sequence
        angles = np.zeros((6, 3))
        
        # Set angles based on current individual and sequence
        for i in range(6):  # 6 legs
            for j in range(3):  # 3 DOF
                if j == 0:  # First DOF (leg angle)
                    phase = 0 if i % 2 == 0 else 1  # Alternating phases
                    angles[i, j] = np.radians(self.hosts[self.gai, self.gaj, phase, j])
                else:  # Other DOFs (middle and end joints)
                    phase = 0 if i % 2 == 0 else 1
                    # Apply signs based on which side of the robot
                    if i < 3:  # Right side
                        angles[i, j] = -np.radians(self.hosts[self.gai, self.gaj, phase, j])
                    else:  # Left side
                        angles[i, j] = np.radians(self.hosts[self.gai, self.gaj, phase, j])
        
        return angles
    
    def should_learn(self):
        """
        Determine if learning should occur based on vertical orientation improvement.
        In main06.cpp, learning occurs when vertical direction improves (curz > prez).
        
        Returns:
            True if learning should occur, False otherwise
        """
        return self.current_z_dir > self.previous_z_dir
    
    def learn(self, state):
        """
        Learn from the current state when vertical orientation has improved.
        
        Args:
            state: Current robot state
            
        Returns:
            Loss value from learning
        """
        if not self.should_learn():
            return 0.0
            
        # Get preprocessed input from the state
        inputs = self._preprocess_input(state)
        
        # Get current target angles
        targets = np.zeros(self.output_dim)
        k = 0
        for i in range(4):  # 4 active legs (corners)
            for j in range(3):  # 3 DOF
                targets[k] = self.current_angles[i][j]
                k += 1
        
        # Store this example in dataset
        self.add_to_dataset(inputs, targets)
        
        # Train neural network on this example
        squared_error = self._learn_single_example(inputs, targets)
        
        # Log training information
        timestamp = datetime.now()
        iter_num = len(self.training_data)
        
        # Get model predictions for logging
        outputs = self.model.predict(np.array([inputs]), verbose=0)[0]
        errors = targets - outputs
        
        # Add to training data DataFrame
        self.training_data = pd.concat([
            self.training_data, 
            pd.DataFrame([{
                'iteration': iter_num,
                'timestamp': timestamp,
                'loss': squared_error,
                'z_direction': self.current_z_dir,
                'inputs': inputs.tolist(),
                'targets': targets.tolist(),
                'outputs': outputs.tolist(),
                'errors': errors.tolist()
            }])
        ], ignore_index=True)
        
        # Log to TensorBoard
        if not hasattr(self, 'tf_writer'):
            self.tf_writer = tf.summary.create_file_writer(self.log_dir)
            
        with self.tf_writer.as_default():
            tf.summary.scalar('training/loss', squared_error, step=iter_num)
            tf.summary.scalar('training/orientation', self.current_z_dir, step=iter_num)
        
        # Occasionally save training data
        if iter_num % 50 == 0:
            self.save_training_data()
            
        # Occasionally perform batch learning on entire dataset (nnlearnall in main06.cpp)
        if self.num_samples > 45 and np.random.random() < 0.1:
            batch_error = self.learn_all_samples()
            self.logger.info(f"Batch learning performed. Average error: {batch_error:.6f}")
            
            # Log to TensorBoard
            with self.tf_writer.as_default():
                tf.summary.scalar('training/batch_loss', batch_error, step=iter_num)
        
        return squared_error
    
    def add_to_dataset(self, inputs, targets):
        """
        Add an example to the dataset with error-based replacement strategy.
        
        Args:
            inputs: Input vector
            targets: Target vector (joint angles)
        """
        # Calculate squared error for this example
        predictions = self.model.predict(np.array([inputs]), verbose=0)[0]
        squared_error = np.sum((targets - predictions)**2)
        
        # Create an example with inputs, targets, error, and orientation
        example = {
            "inputs": inputs.copy(),
            "targets": targets.copy(),
            "error": squared_error,
            "orientation": np.array([inputs[12], inputs[13], inputs[14]])  # Last 3 inputs are orientation
        }
        
        # Add to dataset if not full
        if self.num_samples < self.max_samples:
            self.dataset.append(example)
            self.num_samples += 1
            self.logger.debug(f"Added new example to dataset. Size: {self.num_samples}")
        else:
            # Replace example with minimum leg height (follows main06.cpp)
            replace_idx = self._find_min_height_example()
            self.dataset[replace_idx] = example
            self.logger.debug(f"Replaced example at index {replace_idx} in dataset")
    
    def _find_min_height_example(self):
        """
        Find the example with the lowest leg height in the dataset.
        Follows tdataminz() in main06.cpp.
        """
        min_idx = 0
        min_height = float('inf')
        
        for i, example in enumerate(self.dataset):
            # The z-component of orientation indicates height
            height = example["orientation"][2]
            if height < min_height:
                min_height = height
                min_idx = i
        
        return min_idx
    
    def _learn_single_example(self, inputs, targets):
        """
        Learn from a single example using gradient descent with momentum.
        Implements nnlearn() from main06.cpp.
        
        Args:
            inputs: Input vector
            targets: Target vector
            
        Returns:
            Squared error after learning
        """
        # Forward pass through the network (input to hidden)
        hidden_outputs = np.zeros(self.hidden_dim)
        for i in range(self.hidden_dim):
            hidden_outputs[i] = self.nn_biases[0][i]
            for j in range(self.input_dim):
                hidden_outputs[i] += inputs[j] * self.nn_weights[0][j, i]
            hidden_outputs[i] = self._sigmoid(hidden_outputs[i])
        
        # Forward pass (hidden to output)
        outputs = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            outputs[i] = self.nn_biases[1][i]
            for j in range(self.hidden_dim):
                outputs[i] += hidden_outputs[j] * self.nn_weights[1][j, i]
        
        # Calculate errors and squared error
        errors = targets - outputs
        squared_error = np.sum(errors**2)
        
        # Output layer weight updates (with momentum)
        for i in range(self.output_dim):
            for j in range(self.hidden_dim):
                delta = errors[i] * hidden_outputs[j]
                self.nn_weights[1][j, i] += self.learning_rate * (delta + self.momentum * self.nn_deltas[1][j, i])
                self.nn_deltas[1][j, i] = delta
            # Update bias
            self.nn_biases[1][i] += self.bias_learning_rate * errors[i]
        
        # Calculate hidden layer deltas
        hidden_deltas = np.zeros(self.hidden_dim)
        for j in range(self.hidden_dim):
            hidden_deltas[j] = 0
            for i in range(self.output_dim):
                hidden_deltas[j] += errors[i] * self.nn_weights[1][j, i]
            hidden_deltas[j] *= hidden_outputs[j] * (1.0 - hidden_outputs[j])
        
        # Hidden layer weight updates (with momentum and dropout)
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
                # Apply dropout (randomly skip updates)
                if np.random.random() < self.dropout_rate:
                    delta = hidden_deltas[i] * inputs[j]
                    self.nn_weights[0][j, i] += self.learning_rate * (delta + self.momentum * self.nn_deltas[0][j, i])
                    self.nn_deltas[0][j, i] = delta
            # Update bias
            self.nn_biases[0][i] += self.bias_learning_rate * hidden_deltas[i]
        
        return squared_error
    
    def learn_all_samples(self):
        """
        Learn from the entire dataset. Implements nnlearnall() from main06.cpp.
        
        Returns:
            Average squared error
        """
        total_error = 0.0
        iterations = 5000  # Large number of iterations as in main06.cpp
        
        # Prepare data for TensorFlow training
        if len(self.dataset) == 0:
            return 0.0
            
        X = np.array([example["inputs"] for example in self.dataset])
        y = np.array([example["targets"] for example in self.dataset])
        
        # Train model with TensorFlow
        history = self.model.fit(
            X, y,
            epochs=iterations // 1000,  # Reduce epochs for efficiency
            batch_size=len(self.dataset),
            verbose=0,
            callbacks=[self.tensorboard_callback]
        )
        
        # Update weights and biases from TensorFlow model to match our internal representation
        weights = self.model.get_weights()
        self.nn_weights[0] = weights[0]
        self.nn_biases[0] = weights[1]
        self.nn_weights[1] = weights[2]
        self.nn_biases[1] = weights[3]
        
        # Get final loss
        final_loss = history.history['loss'][-1]
        
        self.logger.info(f"Completed batch learning on {len(self.dataset)} examples. Final loss: {final_loss:.6f}")
        
        return final_loss
    
    def predict(self, state):
        """
        Generate target joint angles from current state using neural network.
        Implements nninf() from main06.cpp.
        
        Args:
            state: Current robot state
            
        Returns:
            Target joint angles for all 6 legs
        """
        # Preprocess input
        inputs = self._preprocess_input(state)
        
        # Forward pass through the network
        hidden_outputs = np.zeros(self.hidden_dim)
        for i in range(self.hidden_dim):
            hidden_outputs[i] = self.nn_biases[0][i]
            for j in range(self.input_dim):
                hidden_outputs[i] += inputs[j] * self.nn_weights[0][j, i]
            hidden_outputs[i] = self._sigmoid(hidden_outputs[i])
        
        # Hidden to output
        outputs = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            outputs[i] = self.nn_biases[1][i]
            for j in range(self.hidden_dim):
                outputs[i] += hidden_outputs[j] * self.nn_weights[1][j, i]
        
        # Convert to 6 legs x 3 DOF format
        angles = np.zeros((6, 3))
        
        # Set active corner legs (0, 2, 3, 5) with neural network outputs
        active_legs = [0, 2, 3, 5]
        for idx, leg_idx in enumerate(active_legs):
            for j in range(3):
                angles[leg_idx, j] = outputs[idx*3 + j]
        
        # Set middle legs (1, 4) to neutral position - same as in main06.cpp
        for leg_idx in [1, 4]:
            for j in range(3):
                angles[leg_idx, j] = 0.0
        
        return angles
    
    def _preprocess_input(self, state):
        """
        Preprocess the state for neural network input.
        Follows logic from main06.cpp's neural network input preparation.
        
        Args:
            state: Robot state
            
        Returns:
            Processed input vector of length input_dim
        """
        inputs = np.zeros(self.input_dim)
        
        # Get joint angles for corner legs (0, 2, 3, 5)
        if 'joint_angles' in state:
            joint_angles = state['joint_angles']
            active_legs = [0, 2, 3, 5]
            k = 0
            for i, leg_idx in enumerate(active_legs):
                for j in range(3):
                    inputs[k] = self._sigmoid(joint_angles[leg_idx, j])
                    # Store current angles for learning
                    self.current_angles[i, j] = joint_angles[leg_idx, j]
                    k += 1
        
        # Get orientation information (last 3 inputs)
        if 'rotation_matrix' in state:
            rot_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
            inputs[12] = self._sigmoid(rot_matrix[0, 2])  # x component of z-axis
            inputs[13] = self._sigmoid(rot_matrix[1, 2])  # y component of z-axis
            inputs[14] = self._sigmoid(rot_matrix[2, 2])  # z component of z-axis
        
        return inputs
    
    def _sigmoid(self, x):
        """Apply sigmoid function to normalize inputs."""
        return 1.0 / (1.0 + np.exp(-x))
    
    def ssga_rank(self):
        """
        Rank individuals based on fitness for SSGA.
        Implements SSGA_rank() from main06.cpp.
        """
        # Reset categories
        self.category = np.full(self.gan, -1)
        
        # For each population member, assign to an objective category
        for i in range(self.gan):
            h = i % 3  # Cycle through objectives (forward, left turn, right turn)
            k = 0
            while self.category[k] != -1:  # Find first unassigned individual
                k += 1
                
            # Find the best unassigned individual for this objective
            for j in range(k+1, self.gan):
                if self.category[j] == -1 and self.fitness[j, h] > self.fitness[k, h]:
                    k = j
                    
            self.category[k] = h
        
        # Log rankings
        self.logger.info("\nSSGA Rankings:")
        for i in range(self.gan):
            h = self.category[i]
            if h >= 0:
                self.logger.info(f"r[{i}]:{h}, {self.fitness[i, h]:.2f}")
    
    def ssga_evolve(self):
        """
        Evolve the population using SSGA.
        Implements SSGA_main() from main06.cpp.
        """
        self.ssga_rank()
        
        # Determine which objective to focus on for this generation
        h = self.iteration % 3  # Cycle through objectives
        objective_names = ["Forward", "Left Turn", "Right Turn"]
        
        self.logger.info(f"Search for {objective_names[h]}")
        
        # Find worst and best individuals for this objective
        g1 = 0  # Worst individual (to be replaced)
        while self.category[g1] != h:
            g1 += 1
        g2 = g1  # Best individual (to copy from)
        
        for i in range(g1+1, self.gan):
            if self.category[i] == h:
                if self.fitness[i, h] < self.fitness[g1, h]:  # Find worst
                    g1 = i
                elif self.fitness[i, h] > self.fitness[g2, h]:  # Find best
                    g2 = i
        
        # Random individual for crossover
        g3 = int(self.gan * np.random.random())
        r = np.random.random() * 0.5  # Crossover rate
        
        # Copy sequence length from best to worst
        self.host_lengths[g1] = self.host_lengths[g2]
        
        # Apply crossover and mutation
        for m in range(self.host_lengths[g1]):
            for i in range(2):  # Two phases
                for j in range(self.dof):
                    if np.random.random() < r and m < self.host_lengths[g3]:
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
        
        # Apply specialized mutations
        
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
                temp = self.hosts[g1, m, 0, j]
                self.hosts[g1, m, 0, j] = self.hosts[g1, m, 1, j]
                self.hosts[g1, m, 1, j] = temp
        
        # 4. Order exchange mutation (10% chance)
        elif np.random.random() < 0.1:
            k = int(self.host_lengths[g1] * np.random.random())
            m = int(self.host_lengths[g1] * np.random.random())
            
            if k != m:
                self.logger.info("-- order exchange mutation --")
                # Swap positions in sequence
                for i in range(2):
                    for j in range(self.dof):
                        temp = self.hosts[g1, k, i, j]
                        self.hosts[g1, k, i, j] = self.hosts[g1, m, i, j]
                        self.hosts[g1, m, i, j] = temp
        
        # Set current individual to evolved offspring
        self.gai = g1
        self.logger.info(f"New individual selected: {self.gai}")
    
    def evaluate_fitness(self, robot, distance, angle_change, alignment):
        """
        Calculate fitness values for the current motion.
        Implements calfit() logic from main06.cpp.
        
        Args:
            robot: Robot instance
            distance: Distance traveled
            angle_change: Change in orientation
            alignment: Alignment between direction and movement
            
        Returns:
            Updated fitness values
        """
        # Calculate fitness values for each objective
        # 1. Forward motion fitness
        self.fitness[self.gai, 0] = np.exp(-angle_change**2) + distance * 10 + alignment
        
        # 2. Left turn fitness
        self.fitness[self.gai, 1] = np.exp(-(angle_change + np.pi/2)**2) + np.exp(-distance**2)
        
        # 3. Right turn fitness
        self.fitness[self.gai, 2] = np.exp(-(angle_change - np.pi/2)**2) + np.exp(-distance**2)
        
        # Update current fitness values
        for i in range(3):
            self.cfith[self.iteration, i] = self.fitness[self.gai, i]
        
        self.chostl[self.iteration] = self.host_lengths[self.gai]
        
        # Find best fitness for each objective
        for i in range(3):
            best_idx = np.argmax(self.fitness[:, i])
            self.bfith[self.iteration, i] = self.fitness[best_idx, i]
            self.bhostl[self.iteration, i] = self.host_lengths[best_idx]
        
        # Log fitness values
        self.logger.info(f"Current fitness - Forward: {self.fitness[self.gai, 0]:.4f}, "
                       f"Left: {self.fitness[self.gai, 1]:.4f}, "
                       f"Right: {self.fitness[self.gai, 2]:.4f}")
        self.logger.info(f"Best fitness - Forward: {self.bfith[self.iteration, 0]:.4f}, "
                       f"Left: {self.bfith[self.iteration, 1]:.4f}, "
                       f"Right: {self.bfith[self.iteration, 2]:.4f}")
        
        return self.fitness[self.gai]
    
    def save_training_data(self):
        """Save the training data to CSV and JSON formats."""
        # Save to CSV (easier for external analysis)
        csv_path = os.path.join(self.log_dir, 'training_data.csv')
        
        # Create a copy for CSV export (convert lists to strings)
        df_export = self.training_data.copy()
        list_columns = ['inputs', 'targets', 'outputs', 'errors']
        for col in list_columns:
            df_export[col] = df_export[col].apply(lambda x: json.dumps(x))
            
        df_export.to_csv(csv_path, index=False)
        
        # Save dataset samples
        dataset_path = os.path.join(self.log_dir, 'dataset_samples.json')
        with open(dataset_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            dataset_json = []
            for example in self.dataset:
                example_copy = example.copy()
                for key, value in example_copy.items():
                    if isinstance(value, np.ndarray):
                        example_copy[key] = value.tolist()
                dataset_json.append(example_copy)
            
            json.dump(dataset_json, f, indent=2)
        
        # Save evolutionary algorithm state
        fitness_path = os.path.join(self.log_dir, 'fitness_history.csv')
        fitness_df = pd.DataFrame({
            'iteration': range(self.iteration + 1),
            'forward_best': self.bfith[:self.iteration + 1, 0],
            'left_best': self.bfith[:self.iteration + 1, 1],
            'right_best': self.bfith[:self.iteration + 1, 2],
            'forward_current': self.cfith[:self.iteration + 1, 0],
            'left_current': self.cfith[:self.iteration + 1, 1],
            'right_current': self.cfith[:self.iteration + 1, 2]
        })
        fitness_df.to_csv(fitness_path, index=False)
        
        self.logger.info(f"Saved training data to {csv_path}, dataset to {dataset_path}, and fitness to {fitness_path}")
    
    def plot_training_history(self):
        """Plot the training history including loss and fitness."""
        if len(self.training_data) == 0:
            self.logger.warning("No training history to plot.")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 15))
        
        # Plot loss
        plt.subplot(3, 1, 1)
        plt.plot(self.training_data['iteration'], self.training_data['loss'], 'b-')
        plt.xlabel("Training Iteration")
        plt.ylabel("Squared Error")
        plt.title("Neural Network Training Loss")
        plt.grid(True)
        
        # Plot vertical orientation
        plt.subplot(3, 1, 2)
        plt.plot(self.training_data['iteration'], self.training_data['z_direction'], 'g-')
        plt.xlabel("Training Iteration")
        plt.ylabel("Vertical Orientation")
        plt.title("Robot Vertical Orientation (Z-Direction)")
        plt.grid(True)
        
        # Plot fitness history if available
        if self.iteration > 0:
            plt.subplot(3, 1, 3)
            iterations = range(self.iteration + 1)
            plt.plot(iterations, self.bfith[:self.iteration + 1, 0], 'b-', label='Forward Best')
            plt.plot(iterations, self.bfith[:self.iteration + 1, 1], 'g-', label='Left Turn Best')
            plt.plot(iterations, self.bfith[:self.iteration + 1, 2], 'r-', label='Right Turn Best')
            plt.plot(iterations, self.cfith[:self.iteration + 1, 0], 'b--', label='Forward Current')
            plt.plot(iterations, self.cfith[:self.iteration + 1, 1], 'g--', label='Left Turn Current')
            plt.plot(iterations, self.cfith[:self.iteration + 1, 2], 'r--', label='Right Turn Current')
            plt.xlabel("Iteration")
            plt.ylabel("Fitness")
            plt.title("Evolutionary Algorithm Fitness")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'training_history.png')
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Saved training history plot to {plot_path}")
    
    def save(self, filename):
        """Save the controller to a file."""
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(filename)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model weights
        self.model.save_weights(filename + '.weights.h5')
        
        # Save training data
        self.training_data.to_pickle(filename + '.training_data.pkl')
        
        # Save evolutionary algorithm state
        ea_state = {
            'hosts': self.hosts,
            'host_lengths': self.host_lengths,
            'fitness': self.fitness,
            'bfith': self.bfith,
            'cfith': self.cfith,
            'bhostl': self.bhostl,
            'chostl': self.chostl,
            'iteration': self.iteration,
            'gai': self.gai,
            'gaj': self.gaj
        }
        with open(filename + '.ea_state.pkl', 'wb') as f:
            pickle.dump(ea_state, f)
        
        # Save neural network parameters
        nn_params = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'learning_rate': self.learning_rate,
            'bias_learning_rate': self.bias_learning_rate,
            'momentum': self.momentum,
            'dropout_rate': self.dropout_rate,
            'nn_weights': self.nn_weights,
            'nn_biases': self.nn_biases,
            'nn_deltas': self.nn_deltas,
            'dataset': self.dataset,
            'num_samples': self.num_samples,
            'log_dir': self.log_dir
        }
        with open(filename + '.params.pkl', 'wb') as f:
            pickle.dump(nn_params, f)
        
        self.logger.info(f"Saved controller to {filename}")
    
    @classmethod
    def load(cls, filename):
        """Load a controller from a file."""
        # Load parameters
        with open(filename + '.params.pkl', 'rb') as f:
            nn_params = pickle.load(f)
        
        # Create new controller
        controller = cls(
            input_dim=nn_params['input_dim'],
            hidden_dim=nn_params['hidden_dim'],
            output_dim=nn_params['output_dim'],
            log_dir=nn_params.get('log_dir', None)
        )
        
        # Set neural network parameters
        controller.learning_rate = nn_params['learning_rate']
        controller.bias_learning_rate = nn_params['bias_learning_rate']
        controller.momentum = nn_params['momentum']
        controller.dropout_rate = nn_params['dropout_rate']
        controller.nn_weights = nn_params['nn_weights']
        controller.nn_biases = nn_params['nn_biases']
        controller.nn_deltas = nn_params['nn_deltas']
        controller.dataset = nn_params['dataset']
        controller.num_samples = nn_params['num_samples']
        
        # Load model weights
        controller.model.load_weights(filename + '.weights.h5')
        
        # Load training data
        try:
            controller.training_data = pd.read_pickle(filename + '.training_data.pkl')
            controller.logger.info(f"Loaded training data with {len(controller.training_data)} entries")
        except (FileNotFoundError, IOError):
            controller.logger.warning("Training data file not found")
        
        # Load evolutionary algorithm state
        try:
            with open(filename + '.ea_state.pkl', 'rb') as f:
                ea_state = pickle.load(f)
                controller.hosts = ea_state['hosts']
                controller.host_lengths = ea_state['host_lengths']
                controller.fitness = ea_state['fitness']
                controller.bfith = ea_state['bfith']
                controller.cfith = ea_state['cfith']
                controller.bhostl = ea_state['bhostl']
                controller.chostl = ea_state['chostl']
                controller.iteration = ea_state['iteration']
                controller.gai = ea_state['gai']
                controller.gaj = ea_state['gaj']
                controller.logger.info(f"Loaded evolutionary algorithm state at iteration {controller.iteration}")
        except (FileNotFoundError, IOError):
            controller.logger.warning("Evolutionary algorithm state file not found")
        
        controller.logger.info(f"Loaded controller from {filename}")
        return controller
    
    @staticmethod
    def randn():
        """
        Generate random number from normal distribution.
        Implements rndn() from main06.cpp using Central Limit Theorem.
        """
        return (np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() +
                np.random.random() + np.random.random() + np.random.random() - 6.0)