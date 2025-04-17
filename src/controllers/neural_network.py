import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt


class NeuralController:
    """
    Neural network controller for robot locomotion.
    This is a port of the neural network implementation from the original C++ code.
    """
    
    def __init__(self, input_dim=15, hidden_dim=30, output_dim=12):
        """
        Initialize the neural network controller.
        
        Args:
            input_dim: Dimension of input (joint angles + robot state)
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output (target joint angles)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Set up TensorFlow model
        self.model = self._build_model()
        
        # For tracking training data
        self.training_data = []
        self.num_samples = 0
        self.max_samples = 200
        
        # Learning parameters
        self.learning_rate = 0.01
        self.momentum = 0.2
        self.dropout_rate = 1.0  # No dropout
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum
        )
    
    def _build_model(self):
        """
        Build the neural network model.
        
        Returns:
            TensorFlow model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hidden_dim, 
                activation='sigmoid',
                input_shape=(self.input_dim,),
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
            ),
            tf.keras.layers.Dense(
                self.output_dim,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
            )
        ])
        
        # Use MSE loss and SGD optimizer
        model.compile(
            optimizer=self.optimizer,
            loss='mse'
        )
        
        return model
    
    def predict(self, state):
        """
        Generate target joint angles from current state.
        
        Args:
            state: Current robot state including joint angles
            
        Returns:
            Target joint angles
        """
        # Preprocess input
        x = self._preprocess_input(state)
        
        # Make prediction
        y = self.model.predict(np.array([x]), verbose=0)[0]
        
        return y
    
    def _preprocess_input(self, state):
        """
        Preprocess the input state for the neural network.
        
        Args:
            state: Raw robot state
            
        Returns:
            Processed input vector
        """
        # Extract relevant information from state
        x = np.zeros(self.input_dim)
        
        # First 12 inputs are the current joint angles for 4 legs (3 DOF each)
        # Get joint angles from the first 4 legs
        for i in range(4):
            for j in range(3):
                if 'joint_angles' in state:
                    x[i*3 + j] = self._sigmoid(state['joint_angles'][i, j])
        
        # Last 3 inputs are the robot's orientation (vertical direction)
        if 'rotation_matrix' in state:
            rot_matrix = state['rotation_matrix']
            # Extract the vertical direction (z-axis in world frame)
            for i in range(3):
                x[12 + i] = self._sigmoid(rot_matrix[i*3 + 2])  # z-component (3rd column)
        
        return x
    
    def _sigmoid(self, x):
        """Apply sigmoid function to x."""
        return 1.0 / (1.0 + np.exp(-x))
    
    def learn(self, current_state, target_angles):
        """
        Learn from a single example.
        
        Args:
            current_state: Current state of the robot
            target_angles: Target joint angles that worked well
            
        Returns:
            Training loss
        """
        # Preprocess input
        x = self._preprocess_input(current_state)
        
        # Store training example
        self.training_data.append((x, target_angles))
        self.num_samples += 1
        
        # If we have too many samples, remove the oldest one
        if self.num_samples > self.max_samples:
            self.training_data.pop(0)
            self.num_samples -= 1
        
        # Train on this example
        x_batch = np.array([x])
        y_batch = np.array([target_angles])
        
        # Train for one step
        history = self.model.fit(x_batch, y_batch, epochs=1, verbose=0)
        return history.history['loss'][0]
    
    def batch_learn(self, epochs=10):
        """
        Learn from all stored examples in batch mode.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        if self.num_samples == 0:
            return None
        
        # Prepare batch data
        x_batch = np.array([x for x, _ in self.training_data])
        y_batch = np.array([y for _, y in self.training_data])
        
        # Train on all examples
        history = self.model.fit(x_batch, y_batch, epochs=epochs, verbose=1)
        return history
    
    def save(self, filename):
        """
        Save the controller to a file.
        
        Args:
            filename: Path to save the controller
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save model weights
        self.model.save_weights(filename + ".h5")
        
        # Save other controller parameters
        params = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'dropout_rate': self.dropout_rate,
            'num_samples': self.num_samples,
            'training_data': self.training_data
        }
        
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(params, f)
    
    @classmethod
    def load(cls, filename):
        """
        Load a controller from a file.
        
        Args:
            filename: Path to the saved controller
            
        Returns:
            Loaded controller
        """
        # Load parameters
        with open(filename + ".pkl", 'rb') as f:
            params = pickle.load(f)
        
        # Create a new controller
        controller = cls(
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim']
        )
        
        # Set parameters
        controller.learning_rate = params['learning_rate']
        controller.momentum = params['momentum']
        controller.dropout_rate = params['dropout_rate']
        controller.num_samples = params['num_samples']
        controller.training_data = params['training_data']
        
        # Load weights
        controller.model.load_weights(filename + ".h5")
        
        return controller
    
    def plot_training_data(self):
        """Plot the distribution of training data."""
        if self.num_samples == 0:
            print("No training data to plot.")
            return
        
        # Extract x and y values
        x_values = np.array([x for x, _ in self.training_data])
        y_values = np.array([y for _, y in self.training_data])
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot input distribution
        plt.subplot(2, 1, 1)
        plt.title('Input Distribution')
        plt.boxplot(x_values)
        plt.grid(True)
        plt.xlabel('Input Dimension')
        plt.ylabel('Value')
        
        # Plot output distribution
        plt.subplot(2, 1, 2)
        plt.title('Output Distribution')
        plt.boxplot(y_values)
        plt.grid(True)
        plt.xlabel('Output Dimension')
        plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig('training_data_distribution.png')
        plt.show()


class AdaptiveController:
    """
    An adaptive controller that combines neural network learning
    with evolutionary-optimized motion sequences.
    """
    
    def __init__(self, input_dim=15, hidden_dim=30, output_dim=12,
                 sequence_controller=None):
        """
        Initialize the adaptive controller.
        
        Args:
            input_dim: Input dimension for neural network
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            sequence_controller: Pre-trained sequence controller (optional)
        """
        # Create neural network controller
        self.nn_controller = NeuralController(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Store sequence controller
        self.sequence_controller = sequence_controller
        
        # Control mode
        self.use_sequence = True
        self.adaptation_threshold = 0.1  # Threshold for when to adapt
        
        # Sequence tracking
        self.current_sequence = 0
        self.max_sequences = 20
        self.sequence_step = 0
        self.steps_per_sequence = 50
        
        # For tracking performance
        self.last_height = 0
        self.stability_history = []
    
    def get_actions(self, state):
        """
        Get joint target angles based on current state.
        
        Args:
            state: Current robot state
            
        Returns:
            Target joint angles
        """
        # Get robot height (z-position)
        height = state['position'][2] if 'position' in state else 0
        
        # Check if we need to adapt (robot is tilting or height changed)
        need_adaptation = self._check_adaptation_needed(state, height)
        
        # Use neural network when adaptation is needed
        if need_adaptation and not self.use_sequence:
            # Neural network control
            return self.nn_controller.predict(state)
        
        # Otherwise use sequence controller
        elif self.sequence_controller is not None:
            # Get angles from sequence
            return self._get_sequence_angles(state)
        
        # Fallback to neural network
        else:
            return self.nn_controller.predict(state)
    
    def _check_adaptation_needed(self, state, height):
        """
        Check if adaptation is needed based on robot state.
        
        Args:
            state: Current robot state
            height: Current height of the robot
            
        Returns:
            True if adaptation is needed, False otherwise
        """
        # Check if height has changed significantly
        height_change = abs(height - self.last_height)
        self.last_height = height
        
        # Check if robot is tilting
        tilt = 0
        if 'rotation_matrix' in state:
            # Get the z-component of the up vector
            z_up = state['rotation_matrix'][10]  # 3rd column, 3rd row
            tilt = 1.0 - z_up
        
        # Store stability measure
        stability = 1.0 - (height_change + tilt)
        self.stability_history.append(stability)
        if len(self.stability_history) > 100:
            self.stability_history.pop(0)
        
        # Need adaptation if stability is low
        return tilt > self.adaptation_threshold or height_change > 0.05
    
    def _get_sequence_angles(self, state):
        """
        Get target angles from the sequence controller.
        
        Args:
            state: Current robot state
            
        Returns:
            Target angles for the robot joints
        """
        # Update sequence step
        self.sequence_step += 1
        
        # Move to next sequence if needed
        if self.sequence_step >= self.steps_per_sequence:
            self.sequence_step = 0
            self.current_sequence = (self.current_sequence + 1) % self.sequence_controller['sequence_length']
        
        # Get current sequence
        seq = self.sequence_controller['sequences'][self.current_sequence]
        
        # Convert sequence to target angles
        angles = np.zeros((6, 3))  # 6 legs, 3 DOF each
        
        # Manually set target angles for legs based on current sequence position
        for leg in range(6):
            for dof in range(3):
                # Right side legs (first 3)
                if leg < 3:
                    if leg % 2 == 0:  # Even legs use phase 0
                        angles[leg, dof] = -np.radians(seq[0, dof])
                    else:  # Odd legs use phase 1
                        angles[leg, dof] = -np.radians(seq[1, dof])
                # Left side legs (last 3)
                else:
                    if leg % 2 == 0:  # Even legs use phase 0
                        angles[leg, dof] = np.radians(seq[0, dof])
                    else:  # Odd legs use phase 1
                        angles[leg, dof] = np.radians(seq[1, dof])
        
        return angles.flatten()
    
    def adapt_to_terrain(self, state, reward):
        """
        Adapt controller based on terrain feedback.
        
        Args:
            state: Current state
            reward: Reward signal
            
        Returns:
            Whether adaptation occurred
        """
        # Switch to neural network control if stability is low
        if len(self.stability_history) > 10:
            avg_stability = np.mean(self.stability_history[-10:])
            if avg_stability < 0.8:
                self.use_sequence = False
                return True
        
        # If using neural network, learn from successful motions
        if not self.use_sequence and reward > 0:
            # Get current target angles
            target_angles = self._get_sequence_angles(state)
            
            # Learn from this example
            self.nn_controller.learn(state, target_angles)
            return True
        
        # Switch back to sequence control if stability improves
        if not self.use_sequence and len(self.stability_history) > 20:
            avg_stability = np.mean(self.stability_history[-10:])
            if avg_stability > 0.9:
                self.use_sequence = True
                return True
        
        return False
    
    def save(self, filename):
        """
        Save the controller to a file.
        
        Args:
            filename: Path to save the controller
        """
        # Save neural network controller
        self.nn_controller.save(filename + "_nn")
        
        # Save other parameters
        params = {
            'use_sequence': self.use_sequence,
            'adaptation_threshold': self.adaptation_threshold,
            'sequence_controller': self.sequence_controller,
            'stability_history': self.stability_history
        }
        
        with open(filename + "_params.pkl", 'wb') as f:
            pickle.dump(params, f)
    
    @classmethod
    def load(cls, filename):
        """
        Load a controller from a file.
        
        Args:
            filename: Path to the saved controller
            
        Returns:
            Loaded controller
        """
        # Load parameters
        with open(filename + "_params.pkl", 'rb') as f:
            params = pickle.load(f)
        
        # Load neural network controller
        nn_controller = NeuralController.load(filename + "_nn")
        
        # Create a new controller
        controller = cls(
            input_dim=nn_controller.input_dim,
            hidden_dim=nn_controller.hidden_dim,
            output_dim=nn_controller.output_dim,
            sequence_controller=params['sequence_controller']
        )
        
        # Set parameters
        controller.nn_controller = nn_controller
        controller.use_sequence = params['use_sequence']
        controller.adaptation_threshold = params['adaptation_threshold']
        controller.stability_history = params['stability_history']
        
        return controller
    
    def plot_stability_history(self):
        """Plot the stability history."""
        if not self.stability_history:
            print("No stability history to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.stability_history)
        plt.grid(True)
        plt.xlabel('Step')
        plt.ylabel('Stability')
        plt.title('Robot Stability History')
        plt.axhline(y=0.8, color='r', linestyle='--', label='Adaptation Threshold')
        plt.legend()
        plt.savefig('stability_history.png')
        plt.show()