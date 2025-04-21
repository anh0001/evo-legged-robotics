import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os
import time
import matplotlib.pyplot as plt
import logging
import json
from datetime import datetime

class NeuroAdaptiveTerrainController:
    """
    Integrated controller that combines neural network learning with evolutionary
    optimization, implementing terrain adaptation using leg height sensing.
    
    This controller:
    1. Uses a neural network to adapt to terrain in real-time based on leg heights
    2. Learns when the robot improves its vertical orientation
    3. Maintains a dataset of successful locomotion patterns
    4. Uses incremental learning for continuous adaptation
    """
    
    def __init__(self, input_dim=15, hidden_dim=30, output_dim=12, log_dir=None):
        """
        Initialize the neuro-adaptive controller.
        
        Args:
            input_dim: Dimension of input (joint angles + robot orientation)
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output (target joint angles)
            log_dir: Directory to save logs (defaults to 'logs/neuro_adaptive_terrain/')
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Setup logging
        self.log_dir = log_dir or os.path.join('logs', 'neuro_adaptive_terrain', 
                                            datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        self._setup_logging()
        
        # Learning parameters 
        self.learning_rate = 0.01  # "a" parameter 
        self.bias_learning_rate = 0.005  # "b" parameter
        self.momentum = 0.2  # "c" parameter
        self.dropout_rate = 1.0  # "d" parameter - default no dropout
        
        # Neural network weights and momentum terms
        self.nn_weights = [None] * 2  # Weights for input→hidden and hidden→output
        self.nn_biases = [None] * 2   # Biases for hidden and output layers
        self.momentum_weights = [None] * 2  # For momentum-based updates
        self.initialize_nn_weights()
        
        # Current and target joint angles
        self.current_angles = np.zeros((4, 3))  # 4 active legs, 3 DOF each
        self.target_angles = np.zeros((4, 3))
        
        # For tracking vertical orientation
        self.current_z_dir = 1.0
        self.previous_z_dir = 1.0
        
        # For tracking leg heights
        self.leg_heights = np.zeros(4)  # Heights of 4 corner legs
        self.min_leg_height = 0.0
        
        # Build TensorFlow model
        self.model = self._build_model()
        
        # For controlling when to use neural network vs. sequence
        self.use_neural = False
        self.sequence_counter = 0
        self.max_sequence = 100
        
        # For dataset management
        self.dataset = []
        self.max_samples = 50
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
        
        self.logger.info(f"Initialized NeuroAdaptiveTerrainController with dimensions: " 
                        f"input={input_dim}, hidden={hidden_dim}, output={output_dim}")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger('neuro_adaptive_terrain')
        self.logger.setLevel(logging.INFO)
        
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
        self.momentum_weights[0] = np.zeros((self.input_dim, self.hidden_dim))
        
        # Hidden to output weights
        self.nn_weights[1] = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        self.nn_biases[1] = np.random.randn(self.output_dim) * 0.1
        self.momentum_weights[1] = np.zeros((self.hidden_dim, self.output_dim))
    
    def _build_model(self):
        """Build the TensorFlow model for more efficient training."""
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
        
        # Decide whether to use neural network or sequence
        self.sequence_counter += 1
        if self.sequence_counter >= self.max_sequence:
            self.sequence_counter = 0
            self.use_neural = np.random.random() < 0.5  # Random switching
        
        # Get angles based on control mode
        if self.use_neural:
            # Use neural network
            self.logger.debug("Using neural network for control")
            return self.predict(state)
        else:
            # Use predefined or evolved sequences with noise
            self.logger.debug("Using sequence-based control")
            return self.generate_sequence_angles(state)
    
    def update_orientation(self, state):
        """
        Update tracking of vertical orientation.
        """
        if 'rotation_matrix' in state:
            rot_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
            self.previous_z_dir = self.current_z_dir
            self.current_z_dir = rot_matrix[2, 2]  # z-component of z-axis (vertical)
            
            # Log the orientation change in TensorBoard
            if hasattr(self, 'tf_writer'):
                with self.tf_writer.as_default():
                    tf.summary.scalar('orientation/z_direction', self.current_z_dir, 
                                    step=len(self.training_data))
    
    def update_leg_heights(self, state):
        """
        Update tracking of leg heights based on robot state.
        """
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
    
    def should_learn(self):
        """
        Determine if learning should occur based on vertical orientation improvement.
        
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
        for i in range(4):  # 4 active legs
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
            
        # Occasionally perform batch learning on entire dataset
        if self.num_samples > 45 and np.random.random() < 0.1:
            batch_error = self.learn_from_dataset()
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
            # Replace example with minimum leg height
            replace_idx = self._find_min_height_example()
            self.dataset[replace_idx] = example
            self.logger.debug(f"Replaced example at index {replace_idx} in dataset")
    
    def _find_min_height_example(self):
        """
        Find the example with the lowest leg height in the dataset.
        """
        min_idx = 0
        min_height = float('inf')
        
        for i, example in enumerate(self.dataset):
            # In the original implementation, this uses the z value
            height = example["orientation"][2]  # z-component of orientation
            if height < min_height:
                min_height = height
                min_idx = i
        
        return min_idx
    
    def _learn_single_example(self, inputs, targets):
        """
        Learn from a single example using gradient descent with momentum.
        
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
                self.nn_weights[1][j, i] += self.learning_rate * (delta + self.momentum * self.momentum_weights[1][j, i])
                self.momentum_weights[1][j, i] = delta
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
                    self.nn_weights[0][j, i] += self.learning_rate * (delta + self.momentum * self.momentum_weights[0][j, i])
                    self.momentum_weights[0][j, i] = delta
            # Update bias
            self.nn_biases[0][i] += self.bias_learning_rate * hidden_deltas[i]
        
        return squared_error
    
    def learn_from_dataset(self):
        """
        Learn from the entire dataset.
        
        Returns:
            Average squared error
        """
        total_error = 0
        
        # Train on each example in the dataset
        for example in self.dataset:
            inputs = example["inputs"]
            targets = example["targets"]
            
            # Forward and backward pass
            error = self._learn_single_example(inputs, targets)
            total_error += error
        
        # Calculate average error
        avg_error = total_error / len(self.dataset)
        return avg_error
    
    def predict(self, state):
        """
        Generate target joint angles from current state using neural network.
        
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
        
        # Set middle legs (1, 4) to neutral position
        for leg_idx in [1, 4]:
            for j in range(3):
                angles[leg_idx, j] = 0.0
        
        return angles
    
    def generate_sequence_angles(self, state):
        """
        Generate angles based on predefined sequences with noise.
        
        Args:
            state: Current robot state
            
        Returns:
            Target joint angles for all 6 legs
        """
        angles = np.zeros((6, 3))
        
        # Default angles
        q_init = np.array([0, 45, 45])
        
        # Add noise to create variation
        for i in range(6):
            for j in range(3):
                noise = np.random.randn() * 0.3  # Small noise term
                if i < 3:  # Right side legs
                    angles[i, j] = -np.radians(q_init[j]) + noise
                else:      # Left side legs
                    angles[i, j] = np.radians(q_init[j]) + noise
        
        return angles
    
    def _preprocess_input(self, state):
        """
        Preprocess the state for neural network input.
        
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
            inputs[12] = self._sigmoid(rot_matrix[0, 2])  # x component of z-axis (vertical)
            inputs[13] = self._sigmoid(rot_matrix[1, 2])  # y component of z-axis (vertical)
            inputs[14] = self._sigmoid(rot_matrix[2, 2])  # z component of z-axis (vertical)
        
        return inputs
    
    def _sigmoid(self, x):
        """Apply sigmoid function to normalize inputs."""
        return 1.0 / (1.0 + np.exp(-x))
    
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
            # We need to convert numpy arrays to lists for JSON serialization
            dataset_json = []
            for example in self.dataset:
                example_copy = example.copy()
                for key, value in example_copy.items():
                    if isinstance(value, np.ndarray):
                        example_copy[key] = value.tolist()
                dataset_json.append(example_copy)
            
            json.dump(dataset_json, f, indent=2)
        
        self.logger.info(f"Saved training data to {csv_path} and dataset to {dataset_path}")
    
    def plot_training_history(self):
        """Plot the training history including loss and vertical orientation."""
        if len(self.training_data) == 0:
            self.logger.warning("No training history to plot.")
            return
        
        plt.figure(figsize=(12, 10))
        
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
        
        # Plot average error over time (moving average)
        plt.subplot(3, 1, 3)
        window_size = min(20, len(self.training_data))
        if window_size > 0:
            rolling_loss = self.training_data['loss'].rolling(window=window_size).mean()
            plt.plot(self.training_data['iteration'], rolling_loss, 'r-')
            plt.xlabel("Training Iteration")
            plt.ylabel("Moving Average Error")
            plt.title(f"Moving Average Error (Window Size: {window_size})")
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
        
        # Save model
        self.model.save_weights(filename + '.weights.h5')
        
        # Save training data
        self.training_data.to_pickle(filename + '.training_data.pkl')
        
        # Save other parameters
        params = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'learning_rate': self.learning_rate,
            'bias_learning_rate': self.bias_learning_rate,
            'momentum': self.momentum,
            'dropout_rate': self.dropout_rate,
            'nn_weights': self.nn_weights,
            'nn_biases': self.nn_biases,
            'momentum_weights': self.momentum_weights,
            'current_z_dir': self.current_z_dir,
            'previous_z_dir': self.previous_z_dir,
            'num_samples': self.num_samples,
            'log_dir': self.log_dir
        }
        
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(params, f)
            
        # Save dataset
        with open(filename + '.dataset.pkl', 'wb') as f:
            pickle.dump(self.dataset, f)
            
        self.logger.info(f"Saved controller to {filename}")
    
    @classmethod
    def load(cls, filename):
        """Load a controller from a file."""
        # Load parameters
        with open(filename + '.pkl', 'rb') as f:
            params = pickle.load(f)
        
        # Create new controller
        controller = cls(
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            log_dir=params.get('log_dir', None)
        )
        
        # Set parameters
        controller.learning_rate = params['learning_rate']
        controller.bias_learning_rate = params['bias_learning_rate']
        controller.momentum = params['momentum']
        controller.dropout_rate = params['dropout_rate']
        controller.nn_weights = params['nn_weights']
        controller.nn_biases = params['nn_biases']
        controller.momentum_weights = params['momentum_weights']
        controller.current_z_dir = params['current_z_dir']
        controller.previous_z_dir = params['previous_z_dir']
        controller.num_samples = params['num_samples']
        
        # Load model weights
        controller.model.load_weights(filename + '.weights.h5')
        
        # Load training data if available
        try:
            controller.training_data = pd.read_pickle(filename + '.training_data.pkl')
            controller.logger.info(f"Loaded training data with {len(controller.training_data)} entries")
        except (FileNotFoundError, IOError):
            controller.logger.warning("Training data file not found")
        
        # Load dataset if available
        try:
            with open(filename + '.dataset.pkl', 'rb') as f:
                controller.dataset = pickle.load(f)
            controller.logger.info(f"Loaded dataset with {len(controller.dataset)} examples")
        except (FileNotFoundError, IOError):
            controller.logger.warning("Dataset file not found")
        
        controller.logger.info(f"Loaded controller from {filename}")
        return controller