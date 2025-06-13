#!/usr/bin/env python3
"""
Enhanced Evolution Script for Hexapod Locomotion

This script runs the enhanced VEGA evolution algorithm with comprehensive
stability measures to prevent leg vibrations and achieve robust locomotion.

Features:
- Enhanced physics parameters for stability
- Improved motor control with PD gains  
- Comprehensive 6-objective fitness function
- Real-time stability monitoring
- Detailed logging and visualization

Usage:
    python run_evolution.py                    # Default evolution run
    python run_evolution.py --no-render        # Headless mode
    python run_evolution.py --quick-test       # Quick 100-iteration test
    python run_evolution.py --max-iterations 2000  # Extended evolution
    
Author: Enhanced for vibration prevention based on PyBullet best practices
"""

import os
import sys
import argparse
import numpy as np
import time
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import enhanced components
try:
    import pybullet as p
    from robot.leg_robot import LeggedRobot
    from simulation.environment import Environment
    from evolution.vega import VEGA
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you have installed the requirements:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def setup_enhanced_physics():
    """Setup enhanced physics parameters for stability."""
    return {
        'time_step': 0.002,  # 1/500s for better stability
        'gravity': -9.81,
        'terrain_type': 'flat'  # Start with flat terrain for stability
    }


def setup_enhanced_robot(env):
    """Setup robot with enhanced motor control parameters."""
    robot = LeggedRobot(client=env.client)
    env.add_robot(robot)
    
    # Set conservative motor gains to prevent vibrations
    robot.set_motor_gains(
        kp=3.0,        # Position gain - reduced for stability
        kd=2.0,        # Velocity gain - critical damping
        max_force=5.0 # Conservative force limit
    )
    
    return robot


def setup_enhanced_evolution(population_size=30, chromosome_length=8, max_iterations=500):
    """Setup VEGA with enhanced multi-objective fitness."""
    vega = VEGA(
        population_size=population_size,
        chromosome_length=chromosome_length,
        generations=max_iterations
    )
    
    # Enhanced fitness weights focusing on stability
    vega.fitness_weights = {
        'forward_motion': 1.0,
        'stability': 2.5,      # High priority for stability
        'energy_efficiency': 0.5,
        'smoothness': 2.0,     # High priority for smooth motion
        'direction_control': 1.5,
        'foot_contact': 2.0
    }
    
    return vega


def run_evolution_loop(env, robot, vega, max_iterations, verbose=True, simulation_speed=1.0):
    """
    Main evolution loop with enhanced stability monitoring and timing control.
    
    Args:
        env: Simulation environment
        robot: Robot instance
        vega: Evolution algorithm instance
        max_iterations: Maximum number of iterations
        verbose: Whether to print detailed progress
        simulation_speed: Speed multiplier when not in real-time
        
    Returns:
        Dictionary with evolution results
    """
    # Calculate sleep time based on physics timestep and desired speed
    if simulation_speed > 0 and not env.real_time:
        sleep_time = env.time_step / simulation_speed
    else:
        sleep_time = 0
    
    # Enhanced control parameters
    vel_counter = 0
    times = 0
    timesmax = 30      # Steps per posture - increased for stability
    samstep = 30       # Evaluation interval
    
    # Initialize state tracking
    prev_pos = np.array(robot.get_position())
    prev_state = robot.get_state()
    prev_rot_matrix = np.array(prev_state['rotation_matrix']).reshape(3, 3)
    
    # Reset robot to stable initial posture
    robot.reset_posture()
    
    # Evolution tracking variables
    step_count = 0
    stability_failures = 0
    last_stability_check = 0
    start_time = time.time()
    
    # Results tracking
    results = {
        'completed_iterations': 0,
        'total_steps': 0,
        'stability_failures': 0,
        'final_fitness': None,
        'best_stability': 0.0,
        'evolution_time': 0.0,
        'convergence_achieved': False
    }
    
    if verbose:
        print(f"Starting enhanced evolution loop...")
        print(f"Parameters: timesmax={timesmax}, samstep={samstep}")
        print(f"Motor gains: kp={robot.kp}, kd={robot.kd}, max_force={robot.max_force}")
        print(f"Target iterations: {max_iterations}")
        print("-" * 60)
    
    try:
        while vega.iteration < max_iterations:
            vel_counter += 1
            step_count += 1
            
            # Change from motor control every 2 steps to every 10-20 steps
            if vel_counter % 20 == 0: # Now 25 Hz instead of 250 Hz
                # Update robot orientation state
                robot.update_orientation()
                
                # Apply current target angles
                robot.apply_target_angles()
                
                # Periodic stability monitoring
                if step_count - last_stability_check >= 20:
                    stability_metrics = robot.check_stability()
                    
                    if not stability_metrics['is_stable']:
                        stability_failures += 1
                        
                        if verbose and stability_failures % 10 == 0:
                            print(f"⚠️  Stability warning #{stability_failures}: "
                                  f"vertical={stability_metrics['vertical_stability']:.3f}, "
                                  f"angular_speed={stability_metrics['angular_speed']:.3f}")
                    
                    last_stability_check = step_count
                
                # Check for evaluation timing
                if vel_counter % samstep == 0:
                    vel_counter = 0
                    times += 1
                    
                    # Time to evaluate individual
                    if times >= timesmax:
                        times = 0
                        
                        # Get current robot state
                        curr_pos = np.array(robot.get_position())
                        curr_state = robot.get_state()
                        curr_rot_matrix = np.array(curr_state['rotation_matrix']).reshape(3, 3)
                        
                        # Evaluate fitness with enhanced multi-objective function
                        fitness_values = vega.evaluate_fitness(
                            robot, prev_pos, curr_pos,
                            prev_rot_matrix, curr_rot_matrix,
                            env.ground_id
                        )
                        
                        # Update results tracking
                        results['completed_iterations'] = vega.iteration
                        results['final_fitness'] = fitness_values.copy()
                        results['best_stability'] = max(results['best_stability'], 
                                                       np.max(vega.bfith[:vega.iteration+1, 1]))
                        
                        # Progress reporting
                        if verbose and (vega.iteration % 25 == 0 or vega.iteration < 10):
                            stability_metrics = robot.check_stability()
                            elapsed = time.time() - start_time
                            
                            print(f"\n📊 Iteration {vega.iteration:4d} | "
                                  f"Time: {elapsed:6.1f}s | "
                                  f"Steps: {step_count:6d}")
                            print(f"   Position: ({curr_pos[0]:5.2f}, {curr_pos[1]:5.2f}, {curr_pos[2]:5.2f})")
                            print(f"   Stability: {stability_metrics['vertical_stability']:5.3f} | "
                                  f"Angular: {stability_metrics['angular_speed']:5.2f}")
                            print(f"   Fitness: F={fitness_values[0]:6.1f} S={fitness_values[1]:6.1f} "
                                  f"E={fitness_values[2]:6.1f} Sm={fitness_values[3]:6.1f}")
                            print(f"   Best Stab: {results['best_stability']:6.1f} | "
                                  f"Failures: {stability_failures}")
                        
                        # Update camera for visualization
                        if env.client == p.GUI:
                            env.update_camera()
                        
                        # Store state for next evaluation
                        prev_pos = curr_pos.copy()
                        prev_rot_matrix = curr_rot_matrix.copy()
                        
                        # Advance evolution
                        vega.iteration += 1
                        
                        # Evolve population after initial evaluation phase
                        if vega.iteration >= vega.gan:
                            vega.evolve()
                        
                        # Smooth reset with settling time to prevent vibrations
                        robot.reset_posture(smooth=True)
                        
                        # Allow settling time for smooth transition
                        for settle_step in range(5):
                            env.step()
                            time.sleep(0.001)  # Small delay for settling
                        
                        vega.gaj = 0
                        
                        # Periodic data saving and analysis
                        if vega.iteration % 100 == 0:
                            vega.save_fitness_data()
                            
                            if verbose:
                                print(f"\n💾 Data checkpoint saved at iteration {vega.iteration}")
                        
                        # Check for early convergence
                        if (vega.iteration > 200 and 
                            results['best_stability'] > 180 and  # High stability achieved
                            np.max(vega.bfith[:vega.iteration+1, 0]) > 100):  # Good forward motion
                            
                            results['convergence_achieved'] = True
                            if verbose:
                                print(f"\n🎯 Early convergence achieved at iteration {vega.iteration}!")
                                print(f"   Stability: {results['best_stability']:.1f}")
                                print(f"   Forward motion: {np.max(vega.bfith[:vega.iteration+1, 0]):.1f}")
                            break
                    
                    else:
                        # Move to next posture in sequence
                        vega.gaj += 1
                        if vega.gaj >= vega.host_lengths[vega.gai]:
                            vega.gaj = 0
                        
                        # Apply next target angles
                        try:
                            angles = vega.get_target_angles()
                            robot.set_target_angles(angles)
                        except Exception as e:
                            if verbose:
                                print(f"⚠️  Angle generation error: {e}")
                            # Use safe fallback angles
                            robot.reset_posture()
            
            # Step the simulation
            env.step()
            
            # Add timing control for realistic motion
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Optional additional visualization delay (reduced since we now have proper timing)
            if env.client == p.GUI and step_count % 1000 == 0:  # Less frequent
                time.sleep(0.001)  # Much smaller delay
            
            # Safety reset for critically unstable robots
            if step_count % 2000 == 0:
                stability_metrics = robot.check_stability()
                if (stability_metrics['height'] < 0.15 or 
                    stability_metrics['vertical_stability'] < 0.2):
                    if verbose:
                        print(f"🔄 Safety reset at step {step_count} due to critical instability")
                    robot.reset_posture()
                    stability_failures += 1
    
    except KeyboardInterrupt:
        if verbose:
            print(f"\n⏹️  Evolution interrupted by user at iteration {vega.iteration}")
    
    except Exception as e:
        if verbose:
            print(f"\n❌ Evolution error: {e}")
        raise
    
    # Finalize results
    results['total_steps'] = step_count
    results['stability_failures'] = stability_failures
    results['evolution_time'] = time.time() - start_time
    
    return results

def save_evolution_results(vega, results, args):
    """Save comprehensive evolution results and analysis."""
    try:
        print(f"\n💾 Saving evolution results...")
        
        # Save fitness data and plots
        vega.save_fitness_data()
        plot_path = vega.plot_enhanced_fitness_history()
        
        # Save best controllers
        best_stability_idx = np.argmax(vega.fitness[:, 1])  # Focus on stability
        best_forward_idx = np.argmax(vega.fitness[:, 0])
        
        vega.gai = best_stability_idx
        stability_controller_path = vega.save_best_controller()
        
        # Generate summary
        summary_path = vega.save_summary()
        
        # Print comprehensive results
        print(f"\n📈 Evolution Results Summary:")
        print(f"   Duration: {results['evolution_time']:.1f} seconds")
        print(f"   Completed iterations: {results['completed_iterations']}")
        print(f"   Total simulation steps: {results['total_steps']}")
        print(f"   Stability failures: {results['stability_failures']}")
        print(f"   Early convergence: {'Yes' if results['convergence_achieved'] else 'No'}")
        
        if results['final_fitness'] is not None:
            fitness_names = ['Forward', 'Stability', 'Energy', 'Smoothness', 'Direction', 'Contact']
            print(f"\n🏆 Final Fitness Values:")
            for i, (name, value) in enumerate(zip(fitness_names, results['final_fitness'])):
                print(f"   {name:12s}: {value:8.2f}")
        
        print(f"\n📁 Files Saved:")
        print(f"   Data and plots: {vega.log_dir}")
        print(f"   Best controller: {stability_controller_path}")
        print(f"   Summary: {summary_path}")
        
        # Stability assessment
        final_stability = results['best_stability']
        if final_stability > 200:
            print(f"\n✅ SUCCESS: Excellent stability achieved! ({final_stability:.1f})")
        elif final_stability > 150:
            print(f"\n✅ SUCCESS: Good stability achieved! ({final_stability:.1f})")
        elif final_stability > 100:
            print(f"\n⚠️  PARTIAL: Moderate stability achieved ({final_stability:.1f})")
        else:
            print(f"\n❌ POOR: Low stability achieved ({final_stability:.1f})")
            print("   Consider adjusting parameters or running longer")
        
        return vega.log_dir
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return None


def main():
    """Main entry point for enhanced evolution."""
    parser = argparse.ArgumentParser(
        description="Enhanced Evolution for Hexapod Locomotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evolution.py                     # Standard evolution run
  python run_evolution.py --quick-test        # Quick test (100 iterations)
  python run_evolution.py --no-render         # Headless mode for servers
  python run_evolution.py --max-iterations 2000 --population-size 50  # Extended run
  python run_evolution.py --real-time         # Use PyBullet real-time stepping
  python run_evolution.py --speed 2.0         # 2x speed (non real-time)
  python run_evolution.py --speed 0.5         # Half speed when not in real-time
        """
    )
    
    # Simulation arguments
    parser.add_argument("--no-render", action="store_true", 
                       help="Run without visualization (faster)")
    parser.add_argument("--terrain", choices=["flat", "rough", "obstacles"], 
                       default="flat", help="Terrain type for evolution")
    
    # Evolution arguments
    parser.add_argument("--max-iterations", type=int, default=500,
                       help="Maximum evolution iterations (default: 500)")
    parser.add_argument("--population-size", type=int, default=30,
                       help="Population size (default: 30)")
    parser.add_argument("--chromosome-length", type=int, default=8,
                       help="Chromosome length (default: 8)")
    
    # Control arguments
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test run (100 iterations)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output (default: True)")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")
    
    # Timing arguments
    parser.add_argument("--real-time", action="store_true",
                       help="Use PyBullet's real-time mode (ignores --speed)")
    parser.add_argument("--speed", type=float, default=0.0,
                       help="Speed multiplier when not in real-time; scales physics timestep")
    
    args = parser.parse_args()
    
    # Adjust parameters for quick test
    if args.quick_test:
        args.max_iterations = 100
        args.population_size = 20
        print("🚀 Quick test mode: 100 iterations, population 20")
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    if verbose:
        print("=" * 60)
        print("🧬 Enhanced Hexapod Evolution with Stability Focus")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Max iterations: {args.max_iterations}")
        print(f"  Population size: {args.population_size}")
        print(f"  Chromosome length: {args.chromosome_length}")
        print(f"  Terrain: {args.terrain}")
        print(f"  Rendering: {'Disabled' if args.no_render else 'Enabled'}")
        print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Setup enhanced simulation environment with timing control
        physics_params = setup_enhanced_physics()
        physics_params['terrain_type'] = args.terrain

        env = Environment(render=not args.no_render, real_time=args.real_time, **physics_params)
        
        # Setup enhanced robot
        robot = setup_enhanced_robot(env)
        
        # Setup enhanced evolution algorithm
        vega = setup_enhanced_evolution(
            population_size=args.population_size,
            chromosome_length=args.chromosome_length,
            max_iterations=args.max_iterations
        )
        
        # Determine simulation speed
        if args.real_time:
            simulation_speed = 1.0  # Real-time mode handled by PyBullet
        elif args.speed > 0:
            simulation_speed = args.speed
        elif args.no_render:
            simulation_speed = 0  # Max speed for headless mode
        else:
            simulation_speed = 1.0  # Real-time for visualization
        
        if verbose:
            print(f"\n🤖 Robot initialized with enhanced motor control")
            print(f"🧠 VEGA initialized with 6-objective fitness function")
            print(f"🌍 Environment ready with enhanced physics")
            if args.real_time:
                print(f"🕐 Using PyBullet real-time simulation")
            elif simulation_speed == 0:
                print(f"🚀 Running at maximum speed (no timing control)")
            else:
                print(f"🕐 Running at {simulation_speed:.1f}x real-time speed")
        
        # Run evolution with timing control
        results = run_evolution_loop(env, robot, vega, args.max_iterations, verbose, simulation_speed)
        
        # Save results
        log_dir = save_evolution_results(vega, results, args)
        
        # Cleanup
        env.close()
        
        if verbose:
            print(f"\n🎉 Evolution completed successfully!")
            if log_dir:
                print(f"📂 All results saved to: {log_dir}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Evolution interrupted by user")
        try:
            env.close()
        except:
            pass
    
    except Exception as e:
        print(f"\n❌ Evolution failed: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        try:
            env.close()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
