import numpy as np
import tensorflow as tf
import pickle
import os
import time

class NeuroEvolutionaryController:
    """
    Integrated controller that combines neural network learning with evolutionary
    optimization, implementing the approach from main04.cpp.
    
    This controller:
    1. Uses a neural network to adapt to terrain in real-time
    2. Learns when the robot improves its vertical orientation
    3. Integrates with evolutionary optimization of motion sequences
    """
    
    def __init__(self, input_dim=15, hidden_dim=20, output_dim=12):
        """
        Initialize the neuro-evolutionary controller.
        
        Args:
            input_dim: Dimension of input (joint angles + robot orientation)
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output (target joint angles)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Learning parameters
        self.learning_rate = 0.01  # Same as in main04.cpp
        
        # Initialize neural network weights (nnw[20][20] in main04.cpp)
        self.nn_weights = np.zeros((self.hidden_dim, self.output_dim))
        self.nn_biases = np.zeros(self.output_dim)
        self.initialize_nn_weights()
        
        # Current and target joint angles
        self.current_angles = np.zeros((4, 3))  # 4 active legs, 3 DOF each
        self.target_angles = np.zeros((4, 3))
        
        # For tracking vertical orientation
        self.current_z_dir = 1.0
        self.previous_z_dir = 1.0
        
        # Build TensorFlow model
        self.model = self._build_model()
        
        # For controlling when to use neural network vs. sequence
        self.use_neural = False
        self.sequence_counter = 0
        self.max_sequence = 100
    
    def initialize_nn_weights(self):
        """Initialize neural network weights with small random values."""
        self.nn_weights = np.random.randn(self.input_dim, self.output_dim) * 0.01
        self.nn_biases = np.zeros(self.output_dim)
    
    def _build_model(self):
        """Build the TensorFlow model for more efficient training."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu', 
                                 input_shape=(self.input_dim,)),
            tf.keras.layers.Dense(self.output_dim, activation='tanh')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
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
        # Update orientation tracking
        self.update_orientation(state)
        
        # Decide whether to use neural network or sequence
        # In main04.cpp, this is based on time and random chance
        self.sequence_counter += 1
        if self.sequence_counter >= self.max_sequence:
            self.sequence_counter = 0
            self.use_neural = np.random.random() < 0.5
        
        # Get angles
        if self.use_neural:
            # Use neural network
            return self.predict(state)
        else:
            # Use noise-based motion
            return self.generate_sequence_angles(state)
    
    def update_orientation(self, state):
        """Update tracking of vertical orientation."""
        if 'rotation_matrix' in state:
            rot_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
            self.previous_z_dir = self.current_z_dir
            self.current_z_dir = rot_matrix[2, 2]  # z-component of z-axis (vertical)
    
    def should_learn(self):
        """
        Determine if learning should occur based on vertical orientation improvement.
        In main04.cpp, learning occurs when vertical direction improves (curz > prez).
        
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
        
        # Get current joint angles and orientation from state
        inputs = self._preprocess_input(state)
        
        # Get current target angles (which worked well since orientation improved)
        # We'll use these as the target for training
        targets = np.zeros(self.output_dim)
        k = 0
        for i in range(4):  # 4 active legs
            for j in range(3):  # 3 DOF
                targets[k] = self.current_angles[i][j]
                k += 1
        
        # Train the model for one step
        history = self.model.fit(
            np.array([inputs]), 
            np.array([targets]), 
            epochs=1, 
            verbose=0
        )
        
        return history.history['loss'][0]
    
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
        
        # Get prediction from neural network
        predictions = self.model.predict(np.array([inputs]), verbose=0)[0]
        
        # Convert to 6 legs x 3 DOF format
        angles = np.zeros((6, 3))
        
        # Set active corner legs (0, 2, 3, 5) with neural network outputs
        active_legs = [0, 2, 3, 5]
        for idx, leg_idx in enumerate(active_legs):
            for j in range(3):
                angles[leg_idx, j] = predictions[idx*3 + j]
        
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
        # In main04.cpp, this uses predefined angles with noise
        angles = np.zeros((6, 3))
        
        # Default angles like in q_init from main04.cpp
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
        Follows the input creation logic from main04.cpp:
        - Joint angles from corner legs
        - Robot orientation (vertical direction)
        
        Args:
            state: Robot state
            
        Returns:
            Processed input vector of length 15
        """
        inputs = np.zeros(self.input_dim)
        
        # Get joint angles for corner legs (0, 2, 3, 5) as in main04.cpp
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
        
        # Get orientation (last 3 inputs)
        if 'rotation_matrix' in state:
            rot_matrix = np.array(state['rotation_matrix']).reshape(3, 3)
            for i in range(3):
                inputs[12 + i] = self._sigmoid(rot_matrix[i, 2])  # z-axis components
        
        return inputs
    
    def _sigmoid(self, x):
        """Apply sigmoid function to normalize inputs."""
        return 1.0 / (1.0 + np.exp(-x))
    
    def save(self, filename):
        """Save the controller to a file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save model
        self.model.save_weights(filename + '.weights.h5')
        
        # Save other parameters
        params = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'learning_rate': self.learning_rate,
            'current_z_dir': self.current_z_dir,
            'previous_z_dir': self.previous_z_dir
        }
        
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(params, f)
    
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
            output_dim=params['output_dim']
        )
        
        # Set parameters
        controller.learning_rate = params['learning_rate']
        controller.current_z_dir = params['current_z_dir']
        controller.previous_z_dir = params['previous_z_dir']
        
        # Load model weights
        controller.model.load_weights(filename + '.weights.h5')
        
        return controller