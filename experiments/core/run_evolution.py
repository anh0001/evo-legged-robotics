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

# FIXED: Correct path resolution to project root and src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# FIXED: Configure matplotlib backend before importing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import enhanced components
try:
    import pybullet as p
    from robot.leg_robot import LeggedRobot
    from simulation.environment import Environment
    from evolution.vega import VEGA
    print("✅ All core modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you have installed the requirements:")
    print("pip install -r requirements.txt")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


def setup_enhanced_physics():
    """Setup enhanced physics parameters based on Bullet3 best practices."""
    return {
        'time_step': 1.0/240.0,  # Standard Bullet3 timestep (240 Hz)
        'gravity': -9.81,
        'terrain_type': 'flat',
        # Additional Bullet3 parameters for stability
        'num_solver_iterations': 50,  # More iterations for better constraint solving
        'enable_cone_friction': True,
        'split_impulse_enabled': True,
        'split_impulse_penetration_threshold': -0.02,
        'contact_breaking_threshold': 0.02,
        'restitution_velocity_threshold': 0.2,
        'erp': 0.8,  # Error Reduction Parameter
        'contact_erp': 0.8,
        'friction_erp': 0.8
    }


def setup_enhanced_robot(env):
    """Setup robot with enhanced motor control parameters."""
    robot = LeggedRobot(client=env.client)
    env.add_robot(robot)
    
    # Set conservative motor gains to prevent vibrations
    robot.set_motor_gains(
        kp=5.0,        # Position gain - reduced for stability
        kd=3.0,        # Velocity gain - critical damping
        max_force=10.0 # Conservative force limit
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


def check_convergence_criteria(vega, results, verbose=True):
    """
    FIXED: Check for early convergence with realistic thresholds.
    
    Args:
        vega: Evolution algorithm instance
        results: Results tracking dictionary
        verbose: Whether to print debug information
        
    Returns:
        bool: True if convergence criteria are met
    """
    # CRITICAL FIX 1: Don't check convergence until after real evolution starts
    min_generations_before_convergence = max(100, vega.gan * 2)  # At least 2 full population cycles
    if vega.iteration < min_generations_before_convergence:
        if verbose and vega.iteration % 25 == 0:
            print(f"🔄 Iteration {vega.iteration}: Too early for convergence check (need {min_generations_before_convergence})")
        return False
    
    # CRITICAL FIX 2: Much more realistic and achievable thresholds
    stability_threshold = 80.0       # Reduced from 120.0 - more achievable
    forward_threshold = 0.8          # Reduced from 1.5 - more achievable  
    convergence_window = 50          # Larger window for stability assessment
    stability_variance_threshold = 100.0  # Much lower threshold for meaningful convergence
    
    # CRITICAL FIX 3: Only check convergence if we have enough evolution history
    if vega.iteration >= convergence_window:
        start_idx = max(0, vega.iteration - convergence_window)
        end_idx = min(vega.iteration + 1, len(vega.bfith))
        
        if end_idx > start_idx and end_idx - start_idx >= convergence_window:
            # Get current best values
            current_best_stability = results['best_stability']
            current_best_forward = np.max(vega.bfith[:min(vega.iteration+1, len(vega.bfith)), 0]) if vega.iteration < len(vega.bfith) else 0
            
            # Analyze recent stability trend
            recent_stability = vega.bfith[start_idx:end_idx, 1]
            stability_variance = np.var(recent_stability)
            
            # Check for improvement stagnation (no improvement over window)
            stability_improvement = recent_stability[-1] - recent_stability[0]
            forward_improvement = vega.bfith[end_idx-1, 0] - vega.bfith[start_idx, 0]
            
            if verbose and vega.iteration % 25 == 0:
                print(f"\n🔍 Convergence Check (Iteration {vega.iteration}):")
                print(f"   Best Stability: {current_best_stability:.1f} (threshold: {stability_threshold})")
                print(f"   Best Forward: {current_best_forward:.1f} (threshold: {forward_threshold})")
                print(f"   Stability Variance: {stability_variance:.3f} (threshold: {stability_variance_threshold})")
                print(f"   Stability Improvement: {stability_improvement:.3f}")
                print(f"   Forward Improvement: {forward_improvement:.3f}")
            
            # FIXED: More stringent convergence criteria
            stability_converged = current_best_stability > stability_threshold
            forward_converged = current_best_forward > forward_threshold
            variance_converged = stability_variance < stability_variance_threshold
            
            # ADDITIONAL: Check for stagnation (no improvement)
            stability_stagnant = abs(stability_improvement) < 5.0  # No improvement in stability
            forward_stagnant = abs(forward_improvement) < 0.2      # No improvement in forward motion
            
            # Require ALL primary criteria AND stagnation for convergence
            criteria_met = stability_converged and forward_converged and variance_converged
            stagnation_met = stability_stagnant and forward_stagnant
            
            if criteria_met and stagnation_met:
                if verbose:
                    print(f"\n🎯 CONVERGENCE ACHIEVED at iteration {vega.iteration}!")
                    print(f"   ✅ Stability: {current_best_stability:.1f} > {stability_threshold}")
                    print(f"   ✅ Forward Motion: {current_best_forward:.1f} > {forward_threshold}")
                    print(f"   ✅ Stability Variance: {stability_variance:.3f} < {stability_variance_threshold}")
                    print(f"   ✅ Stability Stagnant: {stability_improvement:.3f}")
                    print(f"   ✅ Forward Stagnant: {forward_improvement:.3f}")
                
                return True
    
    return False


def run_evolution_loop(env, robot, vega, max_iterations, verbose=True, simulation_speed=1.0):
    """
    Main evolution loop with enhanced stability monitoring and better error handling.
    
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
    
    # Results tracking with FIXED convergence monitoring
    results = {
        'completed_iterations': 0,
        'total_steps': 0,
        'stability_failures': 0,
        'final_fitness': None,
        'best_stability': 0.0,
        'evolution_time': 0.0,
        'convergence_achieved': False,
        'convergence_iteration': None  # Track when convergence occurred
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
            
            # Change from motor control every 2 steps to every 20 steps
            if vel_counter % 20 == 0:
                # Update robot orientation state
                robot.update_orientation()
                
                # Apply current target angles
                robot.apply_target_angles()
                
                # Periodic stability monitoring
                if step_count - last_stability_check >= 20:
                    try:
                        stability_metrics = robot.check_stability()
                        
                        if not stability_metrics['is_stable']:
                            stability_failures += 1
                            
                            if verbose and stability_failures % 10 == 0:
                                print(f"⚠️  Stability warning #{stability_failures}: "
                                      f"vertical={stability_metrics['vertical_stability']:.3f}, "
                                      f"angular_speed={stability_metrics['angular_speed']:.3f}")
                        
                        last_stability_check = step_count
                    except Exception as e:
                        if verbose:
                            print(f"⚠️  Stability check error: {e}")
                
                # Check for evaluation timing
                if vel_counter % samstep == 0:
                    vel_counter = 0
                    times += 1
                    
                    # Time to evaluate individual
                    if times >= timesmax:
                        times = 0
                        
                        try:
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
                            
                            # FIXED: Safe array access for best stability
                            if vega.iteration < len(vega.bfith):
                                current_best_stab = np.max(vega.bfith[:vega.iteration+1, 1])
                                results['best_stability'] = max(results['best_stability'], current_best_stab)
                            
                            # Progress reporting
                            if verbose and (vega.iteration % 25 == 0 or vega.iteration < 10):
                                try:
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
                                    
                                    # Show if we're in initial sampling or evolution phase
                                    if vega.iteration < vega.gan:
                                        print(f"   Phase: Initial sampling ({vega.iteration + 1}/{vega.gan})")
                                    else:
                                        print(f"   Phase: Evolution (generation {vega.iteration - vega.gan + 1})")
                                    
                                except Exception as e:
                                    print(f"   Progress reporting error: {e}")
                            
                            # Update camera for visualization
                            if env.client == p.GUI:
                                env.update_camera()
                            
                            # Store state for next evaluation
                            prev_pos = curr_pos.copy()
                            prev_rot_matrix = curr_rot_matrix.copy()
                            
                            # Advance evolution
                            vega.iteration += 1

                            # Choose next individual or evolve population
                            if vega.iteration < vega.gan:
                                vega.gai = vega.iteration % vega.gan
                                vega.clear_motion_history()
                            else:
                                vega.evolve()  # gai updated inside evolve

                            # Reset gait index for new individual
                            vega.gaj = 0
                            
                            # Smooth reset with settling time to prevent vibrations
                            robot.reset_posture(smooth=True)
                            vega.prev_robot_state = None
                            
                            # Allow settling time for smooth transition
                            for settle_step in range(5):
                                env.step()
                                time.sleep(0.001)  # Small delay for settling
                            
                            # Periodic data saving and analysis
                            if vega.iteration % 25 == 0:  # More frequent saving for testing
                                try:
                                    vega.save_fitness_data()
                                    if verbose:
                                        print(f"\n💾 Data checkpoint saved at iteration {vega.iteration}")
                                except Exception as e:
                                    if verbose:
                                        print(f"⚠️  Data saving error: {e}")
                            
                            # FIXED: Only check for convergence after sufficient evolution
                            if vega.iteration > max(100, vega.gan * 2):  # After initial sampling + some evolution
                                if check_convergence_criteria(vega, results, verbose):
                                    results['convergence_achieved'] = True
                                    results['convergence_iteration'] = vega.iteration
                                    if verbose:
                                        print(f"\n🎯 Early convergence detected! Stopping evolution.")
                                    break
                        
                        except Exception as e:
                            if verbose:
                                print(f"⚠️  Evaluation error: {e}")
                            # Continue with next iteration
                            vega.iteration += 1
                    
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
            try:
                env.step()
            except Exception as e:
                if verbose:
                    print(f"⚠️  Simulation step error: {e}")
                break
            
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
        import traceback
        traceback.print_exc()
    
    # Finalize results
    results['total_steps'] = step_count
    results['stability_failures'] = stability_failures
    results['evolution_time'] = time.time() - start_time
    
    return results

def save_evolution_results(vega, results, args):
    """Save comprehensive evolution results and analysis."""
    try:
        print(f"\n💾 Saving evolution results...")
        
        # Save fitness data first (most important)
        try:
            vega.save_fitness_data()
            print("✅ Fitness data saved successfully")
        except Exception as e:
            print(f"⚠️  Fitness data saving error: {e}")
        
        # Try to save plots with error handling
        plot_path = None
        try:
            plot_path = vega.plot_enhanced_fitness_history()
            print("✅ Fitness plots saved successfully")
        except Exception as e:
            print(f"⚠️  Plot saving error (non-critical): {e}")
            # Continue without plots
        
        # Save best controllers
        try:
            best_stability_idx = np.argmax(vega.fitness[:, 1])
            vega.gai = best_stability_idx
            vega.clear_motion_history()
            stability_controller_path = vega.save_best_controller()
            print("✅ Best controller saved successfully")
        except Exception as e:
            print(f"⚠️  Controller saving error: {e}")
            stability_controller_path = None
        
        # Generate summary
        summary_path = vega.save_summary()
        
        # Print comprehensive results
        print(f"\n📈 Evolution Results Summary:")
        print(f"   Duration: {results['evolution_time']:.1f} seconds")
        print(f"   Completed iterations: {results['completed_iterations']}")
        print(f"   Total simulation steps: {results['total_steps']}")
        print(f"   Stability failures: {results['stability_failures']}")
        print(f"   Early convergence: {'Yes' if results['convergence_achieved'] else 'No'}")
        if results['convergence_achieved']:
            print(f"   Convergence at iteration: {results['convergence_iteration']}")
        
        if results['final_fitness'] is not None:
            fitness_names = ['Forward', 'Stability', 'Energy', 'Smoothness', 'Direction', 'Contact']
            print(f"\n🏆 Final Fitness Values:")
            for i, (name, value) in enumerate(zip(fitness_names, results['final_fitness'])):
                print(f"   {name:12s}: {value:8.2f}")
        
        print(f"\n📁 Files Saved:")
        print(f"   Data and plots: {vega.log_dir}")
        print(f"   Best controller: {stability_controller_path}")
        print(f"   Summary: {summary_path}")
        
        # FIXED: Realistic stability assessment
        final_stability = results['best_stability']
        if final_stability > 220:
            print(f"\n✅ EXCELLENT: Outstanding stability achieved! ({final_stability:.1f})")
        elif final_stability > 200:
            print(f"\n✅ SUCCESS: Excellent stability achieved! ({final_stability:.1f})")
        elif final_stability > 150:
            print(f"\n✅ GOOD: Good stability achieved! ({final_stability:.1f})")
        elif final_stability > 100:
            print(f"\n⚠️  MODERATE: Moderate stability achieved ({final_stability:.1f})")
        else:
            print(f"\n❌ POOR: Low stability achieved ({final_stability:.1f})")
            print("   Consider adjusting parameters or running longer")
        
        return vega.log_dir
        
    except Exception as e:
        print(f"❌ Error in results saving process: {e}")
        import traceback
        traceback.print_exc()
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
    
    env = None
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
            print(f"Target iterations: {args.max_iterations}")
            print(f"Convergence checking starts after iteration: {max(100, vega.gan * 2)}")
            if args.real_time:
                print(f"🕐 Using PyBullet real-time simulation")
            elif simulation_speed == 0:
                print(f"🚀 Running at maximum speed (no timing control)")
            else:
                print(f"🕐 Running at {simulation_speed:.1f}x real-time speed")
            print("-" * 60)
        
        # Run evolution with timing control
        results = run_evolution_loop(env, robot, vega, args.max_iterations, verbose, simulation_speed)
        
        # Save results
        log_dir = save_evolution_results(vega, results, args)
        
        if verbose:
            print(f"\n🎉 Evolution completed successfully!")
            if log_dir:
                print(f"📂 All results saved to: {log_dir}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Evolution interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Evolution failed: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # FIXED: Proper cleanup
        if env is not None:
            try:
                env.close()
                print("✅ Environment closed successfully")
            except Exception as e:
                print(f"⚠️  Environment cleanup error: {e}")


if __name__ == "__main__":
    main()
