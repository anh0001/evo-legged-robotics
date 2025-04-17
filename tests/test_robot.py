import unittest
import numpy as np
import pybullet as p

# Import our modules
from src.robot.leg_robot import LeggedRobot
from src.simulation.environment import Environment
from src.controllers.locomotion import LocomotionGenerator
from src.utils.math_utils import degree_to_radian


class TestLeggedRobot(unittest.TestCase):
    """Test cases for the LeggedRobot class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Connect to PyBullet in direct mode (no visualization)
        self.client = p.connect(p.DIRECT)
        
        # Create robot
        self.robot = LeggedRobot(client=self.client)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Disconnect from PyBullet
        p.disconnect(self.client)
    
    def test_robot_creation(self):
        """Test that robot is created properly."""
        # Check that robot body exists
        self.assertIsNotNone(self.robot.body_id)
        
        # Check that legs exist
        self.assertEqual(len(self.robot.leg_ids), self.robot.total_legs)
        
        # Check that dummy legs exist
        self.assertEqual(len(self.robot.dummy_leg_ids), self.robot.dummy_legs)
        
        # Check that joints exist
        self.assertEqual(len(self.robot.joint_ids), self.robot.total_legs)
    
    def test_reset_posture(self):
        """Test resetting robot posture."""
        # Reset posture
        self.robot.reset_posture()
        
        # Check that target angles match expected values
        for i in range(self.robot.leg_count):
            for j in range(self.robot.dof):
                if i < 3:  # Right side legs
                    self.assertAlmostEqual(
                        self.robot.t_angle[i][j],
                        -degree_to_radian(self.robot.q_init[j]),
                        places=5
                    )
                else:  # Left side legs
                    self.assertAlmostEqual(
                        self.robot.t_angle[i][j],
                        degree_to_radian(self.robot.q_init[j]),
                        places=5
                    )
    
    def test_get_position(self):
        """Test getting robot position."""
        # Get position
        pos = self.robot.get_position()
        
        # Check that position matches expected values
        self.assertAlmostEqual(pos[0], self.robot.box_pos[0], places=5)
        self.assertAlmostEqual(pos[1], self.robot.box_pos[1], places=5)
        self.assertAlmostEqual(pos[2], self.robot.box_pos[2], places=5)
    
    def test_get_state(self):
        """Test getting robot state."""
        # Get state
        state = self.robot.get_state()
        
        # Check that state contains expected keys
        self.assertIn('position', state)
        self.assertIn('orientation', state)
        self.assertIn('rotation_matrix', state)
        self.assertIn('joint_angles', state)
        
        # Check that joint angles shape is correct
        self.assertEqual(state['joint_angles'].shape, (self.robot.leg_count, self.robot.dof))
    
    def test_set_target_angles(self):
        """Test setting target angles."""
        # Create random target angles
        angles = np.random.uniform(-0.5, 0.5, (self.robot.leg_count, self.robot.dof))
        
        # Set target angles
        self.robot.set_target_angles(angles)
        
        # Check that target angles match expected values
        for i in range(self.robot.leg_count):
            for j in range(self.robot.dof):
                self.assertAlmostEqual(self.robot.t_angle[i][j], angles[i][j], places=5)


class TestLocomotionGenerator(unittest.TestCase):
    """Test cases for the LocomotionGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Connect to PyBullet in direct mode (no visualization)
        self.client = p.connect(p.DIRECT)
        
        # Create robot
        self.robot = LeggedRobot(client=self.client)
        
        # Create locomotion generator
        self.locomotion = LocomotionGenerator(self.robot)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Disconnect from PyBullet
        p.disconnect(self.client)
    
    def test_define_tripod_gait(self):
        """Test defining tripod gait."""
        # Define tripod gait
        self.locomotion.define_tripod_gait()
        
        # Check that number of phases is correct
        self.assertEqual(self.locomotion.num_phases, 2)
        
        # Check that phase angles have expected shape
        self.assertEqual(self.locomotion.phase_angles.shape, 
                        (self.locomotion.max_phases, 2, self.locomotion.dof))
    
    def test_define_wave_gait(self):
        """Test defining wave gait."""
        # Define wave gait
        self.locomotion.define_wave_gait()
        
        # Check that number of phases is correct
        self.assertEqual(self.locomotion.num_phases, 6)
    
    def test_define_ripple_gait(self):
        """Test defining ripple gait."""
        # Define ripple gait
        self.locomotion.define_ripple_gait()
        
        # Check that number of phases is correct
        self.assertEqual(self.locomotion.num_phases, 3)
    
    def test_define_turn_gaits(self):
        """Test defining turn gaits."""
        # Define turn left gait
        self.locomotion.define_turn_left_gait()
        self.assertEqual(self.locomotion.num_phases, 2)
        
        # Define turn right gait
        self.locomotion.define_turn_right_gait()
        self.assertEqual(self.locomotion.num_phases, 2)
    
    def test_get_next_angles(self):
        """Test getting next angles."""
        # Get next angles
        angles = self.locomotion.get_next_angles()
        
        # Check that angles have expected shape
        self.assertEqual(angles.shape, (self.robot.leg_count, self.robot.dof))
        
        # Check that angles are valid (within range)
        for i in range(self.robot.leg_count):
            for j in range(self.robot.dof):
                self.assertTrue(-np.pi <= angles[i][j] <= np.pi)
    
    def test_phase_progression(self):
        """Test that phases progress correctly."""
        # Get current phase
        initial_phase = self.locomotion.current_phase
        
        # Step through one full phase
        for _ in range(self.locomotion.steps_per_phase):
            self.locomotion.get_next_angles()
        
        # Check that phase has advanced
        expected_phase = (initial_phase + 1) % self.locomotion.num_phases
        self.assertEqual(self.locomotion.current_phase, expected_phase)


class TestEnvironment(unittest.TestCase):
    """Test cases for the Environment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create environment in direct mode (no visualization)
        self.env = Environment(render=False)
        
        # Create robot
        self.robot = LeggedRobot(client=self.env.client)
        self.env.add_robot(self.robot)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Close environment
        self.env.close()
    
    def test_step(self):
        """Test stepping the environment."""
        # Get initial time
        initial_time = self.env.sim_time
        
        # Step the environment
        obs = self.env.step()
        
        # Check that time has advanced
        self.assertGreater(self.env.sim_time, initial_time)
        
        # Check that observation contains expected keys
        self.assertIn('time', obs)
        self.assertIn('robot_position', obs)
        self.assertIn('robot_orientation', obs)
        self.assertIn('robot_rotation_matrix', obs)
    
    def test_reset(self):
        """Test resetting the environment."""
        # Step the environment a few times
        for _ in range(10):
            self.env.step()
        
        # Reset the environment
        obs = self.env.reset()
        
        # Check that time has been reset
        self.assertEqual(self.env.sim_time, 0)
        
        # Check that observation contains expected keys
        self.assertIn('time', obs)


if __name__ == '__main__':
    unittest.main()