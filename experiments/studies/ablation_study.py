#!/usr/bin/env python3
"""
Systematic Ablation Study Framework for Structural Mutation Operators
Implements the experimental configurations from the strategic guidance document.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import itertools
from concurrent.futures import ProcessPoolExecutor
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root and src directory are on the path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Now import project modules after path setup
from experiments.analysis.statistical_validation import StatisticalValidator
from robot.leg_robot import LeggedRobot
from simulation.environment import Environment
from evolution.enhanced_vega import EnhancedVEGA


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        return super().default(obj)


class AblationStudyManager:
    """
    Manages systematic ablation studies for structural mutation operators.
    Implements the 16 configurations from Table 3 in the strategic guidance.
    """
    
    def __init__(self, base_config_path="config/base_experiment.json"):
        self.base_config = self._load_base_config(base_config_path)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"results/ablation_study_{self.experiment_id}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Define ablation configurations (Table 3 from strategic guidance)
        self.ablation_configs = self._define_ablation_configurations()
        
        # Statistical parameters
        self.num_independent_runs = 30  # Minimum for statistical significance
        self.significance_level = 0.05
        self.target_effect_size = 0.5  # Cohen's d
        
        self.logger.info(f"Initialized AblationStudyManager with {len(self.ablation_configs)} configurations")
    
    def _load_base_config(self, config_path):
        """Load base experimental configuration."""
        default_config = {
            "population_size": 30,
            "chromosome_length": 8,  # Increased from 4 for better operator exploration
            "max_iterations": 500,
            "operator_probabilities": {
                "insertion": 0.15,
                "deletion": 0.15,
                "phase_exchange": 0.10,
                "order_exchange": 0.10
            },
            "fitness_weights": {
                "forward_motion": 1.0,
                "stability": 2.0,
                "energy_efficiency": 0.5,
                "smoothness": 1.5,
                "direction_control": 1.5,
                "foot_contact": 2.0
            },
            "physics_params": {
                "time_step": 0.002,
                "gravity": -9.81,
                "terrain_type": "flat"
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        self.logger = logging.getLogger('ablation_study')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.results_dir / 'ablation_study.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _define_ablation_configurations(self):
        """Define systematic ablation configurations from Table 3."""
        configs = {
            "C0_baseline": {
                "name": "Baseline (All Operators)",
                "active_operators": ["insertion", "deletion", "phase_exchange", "order_exchange"],
                "operator_probabilities": self.base_config["operator_probabilities"],
                "description": "All four structural operators active"
            },
            "C1_no_insertion": {
                "name": "No Insertion",
                "active_operators": ["deletion", "phase_exchange", "order_exchange"],
                "operator_probabilities": {k: v for k, v in self.base_config["operator_probabilities"].items() 
                                         if k != "insertion"},
                "description": "Test impact of removing insertion operator"
            },
            "C2_no_deletion": {
                "name": "No Deletion",
                "active_operators": ["insertion", "phase_exchange", "order_exchange"],
                "operator_probabilities": {k: v for k, v in self.base_config["operator_probabilities"].items() 
                                         if k != "deletion"},
                "description": "Test impact of removing deletion operator"
            },
            "C3_no_phase": {
                "name": "No Phase Exchange",
                "active_operators": ["insertion", "deletion", "order_exchange"],
                "operator_probabilities": {k: v for k, v in self.base_config["operator_probabilities"].items() 
                                         if k != "phase_exchange"},
                "description": "Test impact of removing phase exchange operator"
            },
            "C4_no_order": {
                "name": "No Order Exchange",
                "active_operators": ["insertion", "deletion", "phase_exchange"],
                "operator_probabilities": {k: v for k, v in self.base_config["operator_probabilities"].items() 
                                         if k != "order_exchange"},
                "description": "Test impact of removing order exchange operator"
            },
            "C5_exploration_only": {
                "name": "Exploration Only (I+D)",
                "active_operators": ["insertion", "deletion"],
                "operator_probabilities": {"insertion": 0.15, "deletion": 0.15},
                "description": "Test pure exploration hypothesis"
            },
            "C6_refinement_only": {
                "name": "Refinement Only (P+O)",
                "active_operators": ["phase_exchange", "order_exchange"],
                "operator_probabilities": {"phase_exchange": 0.10, "order_exchange": 0.10},
                "description": "Test pure refinement hypothesis"
            },
            "C7_insertion_phase": {
                "name": "Insertion + Phase",
                "active_operators": ["insertion", "phase_exchange"],
                "operator_probabilities": {"insertion": 0.15, "phase_exchange": 0.10},
                "description": "Test synergy between exploration and refinement"
            },
            "C8_insertion_order": {
                "name": "Insertion + Order",
                "active_operators": ["insertion", "order_exchange"],
                "operator_probabilities": {"insertion": 0.15, "order_exchange": 0.10},
                "description": "Test synergy between exploration and refinement"
            },
            "C9_deletion_phase": {
                "name": "Deletion + Phase",
                "active_operators": ["deletion", "phase_exchange"],
                "operator_probabilities": {"deletion": 0.15, "phase_exchange": 0.10},
                "description": "Test synergy between exploration and refinement"
            },
            "C10_deletion_order": {
                "name": "Deletion + Order",
                "active_operators": ["deletion", "order_exchange"],
                "operator_probabilities": {"deletion": 0.15, "order_exchange": 0.10},
                "description": "Test synergy between exploration and refinement"
            },
            "C11_no_structural": {
                "name": "No Structural Mutations",
                "active_operators": [],
                "operator_probabilities": {},
                "description": "Critical control: only selection and crossover"
            }
        }
        
        return configs
    
    def run_single_experiment(self, config_id, run_id, config_data):
        """Run a single experimental trial."""
        try:
            # Create unique experiment directory
            exp_dir = self.results_dir / config_id / f"run_{run_id:03d}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup environment and robot
            env = Environment(
                render=False,  # Headless for batch processing
                **self.base_config["physics_params"]
            )
            
            robot = LeggedRobot(client=env.client)
            env.add_robot(robot)
            
            # Configure VEGA with operator settings
            vega = EnhancedVEGA(
                population_size=self.base_config["population_size"],
                chromosome_length=self.base_config["chromosome_length"],
                generations=self.base_config["max_iterations"]
            )
            
            # Apply operator configuration
            vega.configure_operators(
                active_operators=config_data["active_operators"],
                operator_probabilities=config_data["operator_probabilities"]
            )
            
            # Apply fitness weights
            vega.fitness_weights = self.base_config["fitness_weights"]
            
            # Enhanced data collection
            experiment_data = {
                "config_id": config_id,
                "run_id": run_id,
                "start_time": datetime.now().isoformat(),
                "fitness_history": [],
                "operator_statistics": {},
                "convergence_metrics": {},
                "final_performance": {}
            }
            
            # Run evolution with enhanced monitoring
            results = self._run_evolution_with_monitoring(vega, robot, env, experiment_data)
            
            # Save results with custom encoder
            results_file = exp_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            # Save fitness data
            fitness_file = exp_dir / "fitness_data.csv"
            vega.save_fitness_data()
            
            # Cleanup
            env.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in experiment {config_id}_run_{run_id}: {str(e)}")
            return {"error": str(e), "config_id": config_id, "run_id": run_id}
    
    def _run_evolution_with_monitoring(self, vega, robot, env, experiment_data):
        """Run evolution with comprehensive monitoring."""
        # Implementation of the enhanced evolution loop from run_evolution.py
        # with additional operator-specific monitoring
        
        vel_counter = 0
        times = 0
        timesmax = 30
        samstep = 30
        
        prev_pos = np.array(robot.get_position())
        prev_state = robot.get_state()
        prev_rot_matrix = np.array(prev_state['rotation_matrix']).reshape(3, 3)
        
        robot.reset_posture()
        
        step_count = 0
        max_steps = 100000
        
        # Enhanced monitoring variables
        operator_applications = {op: 0 for op in ["insertion", "deletion", "phase_exchange", "order_exchange"]}
        operator_successes = {op: 0 for op in ["insertion", "deletion", "phase_exchange", "order_exchange"]}
        
        while step_count < max_steps and vega.iteration < vega.iterations:
            vel_counter += 1
            
            if vel_counter % 2 == 0:
                robot.update_orientation()
                robot.apply_target_angles()
                
                if vel_counter % samstep == 0:
                    vel_counter = 0
                    times += 1
                    
                    if times >= timesmax:
                        times = 0
                        
                        # Evaluate fitness
                        curr_pos = np.array(robot.get_position())
                        curr_state = robot.get_state()
                        curr_rot_matrix = np.array(curr_state['rotation_matrix']).reshape(3, 3)
                        
                        fitness_values = vega.evaluate_fitness(
                            robot, prev_pos, curr_pos,
                            prev_rot_matrix, curr_rot_matrix,
                            env.ground_id
                        )
                        
                        # Track operator statistics
                        if hasattr(vega, 'last_applied_operators'):
                            for op in vega.last_applied_operators:
                                operator_applications[op] += 1
                        
                        # Store fitness history
                        experiment_data["fitness_history"].append({
                            "iteration": vega.iteration,
                            "fitness": fitness_values.tolist(),
                            "hypervolume": vega.calculate_hypervolume() if hasattr(vega, 'calculate_hypervolume') else 0,
                            "diversity": vega.calculate_diversity() if hasattr(vega, 'calculate_diversity') else 0
                        })
                        
                        prev_pos = curr_pos.copy()
                        prev_rot_matrix = curr_rot_matrix.copy()
                        
                        vega.iteration += 1
                        
                        if vega.iteration >= vega.gan:
                            vega.evolve()
                        
                        robot.reset_posture()
                        vega.gaj = 0
                    
                    else:
                        vega.gaj += 1
                        if vega.gaj >= vega.host_lengths[vega.gai]:
                            vega.gaj = 0
                        
                        angles = vega.get_target_angles()
                        robot.set_target_angles(angles)
            
            env.step()
            step_count += 1
        
        # Compile final results
        experiment_data.update({
            "end_time": datetime.now().isoformat(),
            "total_steps": step_count,
            "final_iteration": vega.iteration,
            "operator_statistics": {
                "applications": operator_applications,
                "successes": operator_successes
            },
            "final_performance": {
                "best_fitness": vega.bfith[vega.iteration-1].tolist() if vega.iteration > 0 else [0]*6,
                "final_hypervolume": experiment_data["fitness_history"][-1]["hypervolume"] if experiment_data["fitness_history"] else 0
            }
        })
        
        return experiment_data
    
    def run_full_study(self, parallel=True, max_workers=4):
        """Run the complete ablation study."""
        self.logger.info(f"Starting full ablation study with {len(self.ablation_configs)} configurations")
        self.logger.info(f"Each configuration will run {self.num_independent_runs} independent trials")
        
        # Save study configuration
        study_config = {
            "experiment_id": self.experiment_id,
            "num_configurations": len(self.ablation_configs),
            "num_runs_per_config": self.num_independent_runs,
            "total_experiments": len(self.ablation_configs) * self.num_independent_runs,
            "base_config": self.base_config,
            "ablation_configs": self.ablation_configs,
            "start_time": datetime.now().isoformat()
        }
        
        config_file = self.results_dir / "study_configuration.json"
        with open(config_file, 'w') as f:
            json.dump(study_config, f, indent=2, cls=NumpyEncoder)
        
        all_results = []
        
        if parallel:
            self._run_parallel_experiments(all_results, max_workers)
        else:
            self._run_sequential_experiments(all_results)
        
        # Compile and analyze results
        self._analyze_results(all_results)
        
        self.logger.info("Ablation study completed successfully")
        return all_results
    
    def _run_parallel_experiments(self, all_results, max_workers):
        """Run experiments in parallel for efficiency."""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for config_id, config_data in self.ablation_configs.items():
                for run_id in range(self.num_independent_runs):
                    future = executor.submit(
                        self.run_single_experiment, 
                        config_id, run_id, config_data
                    )
                    futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout
                    all_results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Completed {i + 1}/{len(futures)} experiments")
                        
                except Exception as e:
                    self.logger.error(f"Experiment {i} failed: {str(e)}")
    
    def _run_sequential_experiments(self, all_results):
        """Run experiments sequentially for debugging."""
        total_experiments = len(self.ablation_configs) * self.num_independent_runs
        experiment_count = 0
        
        for config_id, config_data in self.ablation_configs.items():
            self.logger.info(f"Starting configuration: {config_data['name']}")
            
            for run_id in range(self.num_independent_runs):
                result = self.run_single_experiment(config_id, run_id, config_data)
                all_results.append(result)
                
                experiment_count += 1
                self.logger.info(f"Completed experiment {experiment_count}/{total_experiments}")
    
    def _analyze_results(self, all_results):
        """Perform comprehensive statistical analysis of results."""
        self.logger.info("Performing statistical analysis of results")
        
        # Convert results to DataFrame for analysis
        df_results = self._results_to_dataframe(all_results)
        
        # Perform statistical tests
        statistical_results = self._perform_statistical_tests(df_results)
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(df_results)
        
        # Generate summary report
        self._generate_summary_report(df_results, statistical_results, effect_sizes)
        
        # Create visualizations
        self._create_visualizations(df_results)
    
    def _results_to_dataframe(self, all_results):
        """Convert results list to structured DataFrame."""
        records = []
        objectives = list(self.base_config.get("fitness_weights", {}).keys())

        for result in all_results:
            # Skip failed experiments
            if not isinstance(result, dict) or result.get("error"):
                continue

            perf = result.get("final_performance", {})
            fitness = perf.get("best_fitness", [np.nan] * len(objectives))

            row = {
                "config_id": result.get("config_id"),
                "run_id": result.get("run_id"),
                "final_iteration": result.get("final_iteration", np.nan),
                "total_steps": result.get("total_steps", np.nan),
                "final_hypervolume": perf.get("final_hypervolume", np.nan),
            }

            for i, obj in enumerate(objectives):
                col = f"final_{obj}_fitness"
                row[col] = fitness[i] if i < len(fitness) else np.nan

            records.append(row)

        df = pd.DataFrame(records)
        csv_path = self.results_dir / "compiled_results.csv"
        df.to_csv(csv_path, index=False)
        return df
    
    def _perform_statistical_tests(self, df_results):
        """Perform MANOVA and post-hoc tests as specified in Table 5."""
        validator = StatisticalValidator(
            significance_level=self.significance_level,
            min_effect_size=self.target_effect_size,
        )

        manova = validator.perform_manova_analysis(df_results)
        anova = validator.perform_univariate_anova(df_results)
        normality = validator.test_normality_assumptions(df_results)
        homogeneity = validator.test_homogeneity_of_variance(df_results)
        power = validator.power_analysis(df_results)

        results = {
            "manova": manova,
            "anova": anova,
            "normality": normality,
            "homogeneity": homogeneity,
            "power_analysis": power,
        }

        with open(self.results_dir / "statistical_results.json", "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        return results
    
    def _calculate_effect_sizes(self, df_results):
        """Calculate Cohen's d and other effect size measures."""
        validator = StatisticalValidator(
            significance_level=self.significance_level,
            min_effect_size=self.target_effect_size,
        )

        effect_sizes = validator.calculate_effect_sizes(df_results)

        with open(self.results_dir / "effect_sizes.json", "w") as f:
            json.dump(effect_sizes, f, indent=2, cls=NumpyEncoder)

        return effect_sizes
    
    def _generate_summary_report(self, df_results, statistical_results, effect_sizes):
        """Generate comprehensive summary report."""
        objectives = list(self.base_config.get("fitness_weights", {}).keys())

        summary = {}
        for config_id, group in df_results.groupby("config_id"):
            stats = {"num_runs": len(group)}
            for obj in objectives:
                col = f"final_{obj}_fitness"
                if col in group:
                    stats[f"mean_{obj}"] = float(group[col].mean())
                    stats[f"std_{obj}"] = float(group[col].std(ddof=0))
            stats["mean_hypervolume"] = float(group["final_hypervolume"].mean())
            stats["std_hypervolume"] = float(group["final_hypervolume"].std(ddof=0))
            summary[config_id] = stats

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "statistical_results": statistical_results,
            "effect_sizes": effect_sizes,
        }

        with open(self.results_dir / "summary_report.json", "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

        summary_df = pd.DataFrame.from_dict(summary, orient="index")
        summary_df.to_csv(self.results_dir / "summary_report.csv")

        return report
    
    def _create_visualizations(self, df_results):
        """Create publication-ready visualizations."""
        plt.switch_backend("Agg")
        sns.set(style="whitegrid")

        # Boxplot for forward motion performance
        col = "final_forward_motion_fitness"
        if col in df_results.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x="config_id", y=col, data=df_results)
            plt.xticks(rotation=45)
            plt.title("Forward Motion Performance by Configuration")
            plt.tight_layout()
            plt.savefig(self.results_dir / "forward_motion_boxplot.png", dpi=300)
            plt.close()

        # Barplot of mean hypervolume per configuration
        if "final_hypervolume" in df_results.columns:
            plt.figure(figsize=(12, 6))
            hv_summary = df_results.groupby("config_id")["final_hypervolume"].mean().reset_index()
            sns.barplot(x="config_id", y="final_hypervolume", data=hv_summary)
            plt.xticks(rotation=45)
            plt.ylabel("Mean Hypervolume")
            plt.title("Mean Hypervolume by Configuration")
            plt.tight_layout()
            plt.savefig(self.results_dir / "hypervolume_barplot.png", dpi=300)
            plt.close()


if __name__ == "__main__":
    # Example usage
    study_manager = AblationStudyManager()
    
    # # Run a quick test with reduced parameters
    # study_manager.num_independent_runs = 5  # Reduced for testing
    # study_manager.base_config["max_iterations"] = 100  # Reduced for testing
    
    results = study_manager.run_full_study(parallel=False)
    print(f"Completed ablation study with {len(results)} total experiments")