#!/usr/bin/env python3
"""
Enhanced VEGA Implementation for Structural Mutation Operator Analysis
Key modifications to support systematic ablation studies and operator tracking.
"""

import numpy as np
import logging
from datetime import datetime
from collections import defaultdict
from evolution.vega import VEGA

class EnhancedVEGA(VEGA):
    """
    Enhanced VEGA with comprehensive operator tracking and configurable ablation support.
    Extends the existing VEGA implementation with critical research-focused features.
    """
    
    def __init__(self, population_size=30, chromosome_length=8, generations=500):
        # Call parent initialization (assuming it exists)
        super().__init__(population_size, chromosome_length, generations)
        
        # Enhanced operator configuration
        self.operator_config = {
            "insertion": {"active": True, "probability": 0.15},
            "deletion": {"active": True, "probability": 0.15},
            "phase_exchange": {"active": True, "probability": 0.10},
            "order_exchange": {"active": True, "probability": 0.10}
        }
        
        # Expanded chromosome length range (critical for meaningful structural evolution)
        self.min_chromosome_length = 2
        self.max_chromosome_length = chromosome_length  # Now 8 instead of 4
        
        # Operator tracking and statistics
        self.operator_statistics = {
            "applications": defaultdict(int),
            "successes": defaultdict(int),
            "failures": defaultdict(int),
            "improvements": defaultdict(list),
            "generation_history": defaultdict(list)
        }
        
        # Enhanced convergence monitoring
        self.convergence_monitor = {
            "hypervolume_history": [],
            "diversity_history": [],
            "stagnation_counter": 0,
            "convergence_achieved": False,
            "convergence_generation": None
        }
        
        # Performance tracking for each objective
        self.objective_performance = {
            "best_per_generation": np.zeros((generations, 6)),
            "mean_per_generation": np.zeros((generations, 6)),
            "std_per_generation": np.zeros((generations, 6))
        }
        
        # Last applied operators for effect attribution
        self.last_applied_operators = []
        
        self.logger = logging.getLogger('enhanced_vega')
    
    def configure_operators(self, active_operators=None, operator_probabilities=None):
        """
        Configure which operators are active and their probabilities.
        Essential for ablation studies.
        
        Args:
            active_operators: List of active operator names
            operator_probabilities: Dict of operator probabilities
        """
        if active_operators is not None:
            # Deactivate all operators first
            for op in self.operator_config:
                self.operator_config[op]["active"] = False
            
            # Activate specified operators
            for op in active_operators:
                if op in self.operator_config:
                    self.operator_config[op]["active"] = True
                    
        if operator_probabilities is not None:
            for op, prob in operator_probabilities.items():
                if op in self.operator_config:
                    self.operator_config[op]["probability"] = prob
        
        self.logger.info(f"Operator configuration updated: {self.operator_config}")
    
    def apply_structural_mutations(self, individual_idx):
        """
        Enhanced structural mutation application with comprehensive tracking.
        
        Args:
            individual_idx: Index of individual to mutate
            
        Returns:
            List of applied operators
        """
        applied_operators = []
        pre_mutation_fitness = self.fitness[individual_idx].copy()
        
        # Apply each operator based on configuration
        for operator_name, config in self.operator_config.items():
            if not config["active"]:
                continue
                
            if np.random.random() < config["probability"]:
                success = self._apply_single_operator(individual_idx, operator_name)
                
                applied_operators.append(operator_name)
                self.operator_statistics["applications"][operator_name] += 1
                self.operator_statistics["generation_history"][operator_name].append(self.iteration)
                
                if success:
                    self.operator_statistics["successes"][operator_name] += 1
                else:
                    self.operator_statistics["failures"][operator_name] += 1
        
        # Track performance improvement attribution
        if applied_operators:
            post_mutation_fitness = self.evaluate_individual_fitness(individual_idx)
            improvement = np.mean(post_mutation_fitness - pre_mutation_fitness)
            
            for op in applied_operators:
                self.operator_statistics["improvements"][op].append(improvement)
        
        self.last_applied_operators = applied_operators
        return applied_operators
    
    def _apply_single_operator(self, individual_idx, operator_name):
        """
        Apply a single structural mutation operator with enhanced validation.
        
        Args:
            individual_idx: Index of individual to mutate
            operator_name: Name of operator to apply
            
        Returns:
            bool: Success/failure of operation
        """
        try:
            original_length = self.host_lengths[individual_idx]
            
            if operator_name == "insertion":
                return self._apply_insertion_mutation(individual_idx)
            elif operator_name == "deletion":
                return self._apply_deletion_mutation(individual_idx)
            elif operator_name == "phase_exchange":
                return self._apply_phase_exchange_mutation(individual_idx)
            elif operator_name == "order_exchange":
                return self._apply_order_exchange_mutation(individual_idx)
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Operator {operator_name} failed on individual {individual_idx}: {e}")
            return False
    
    def _apply_insertion_mutation(self, individual_idx):
        """Enhanced insertion mutation with expanded length range."""
        current_length = self.host_lengths[individual_idx]
        
        # Check if insertion is possible (expanded range)
        if current_length >= self.max_chromosome_length:
            return False
        
        # Select insertion position
        insertion_pos = np.random.randint(0, current_length)
        
        # Shift existing elements
        for pos in range(current_length, insertion_pos, -1):
            for phase in range(2):
                for dof in range(self.dof):
                    self.hosts[individual_idx, pos, phase, dof] = \
                        self.hosts[individual_idx, pos-1, phase, dof]
        
        # Insert new random element with conservative bounds
        for phase in range(2):
            for dof in range(self.dof):
                if dof == 0:  # First DOF (leg angle) - more conservative range
                    angle = np.random.uniform(-20, 20)  # Reduced from ±45°
                else:  # Other DOFs
                    center = self.q_init[dof]
                    variation = self.q_range[dof] * 0.3  # 30% of full range
                    angle = center + np.random.uniform(-variation/2, variation/2)
                
                # Ensure bounds
                angle = np.clip(angle, self.q_min[dof], self.q_min[dof] + self.q_range[dof])
                self.hosts[individual_idx, insertion_pos, phase, dof] = angle
        
        # Update length
        self.host_lengths[individual_idx] += 1
        
        self.logger.debug(f"Insertion mutation: {current_length} -> {self.host_lengths[individual_idx]}")
        return True
    
    def _apply_deletion_mutation(self, individual_idx):
        """Enhanced deletion mutation with improved position selection."""
        current_length = self.host_lengths[individual_idx]
        
        # Check if deletion is possible
        if current_length <= self.min_chromosome_length:
            return False
        
        # Select deletion position (avoid critical positions if possible)
        if current_length > 3:
            # Avoid deleting first and last elements when possible
            deletion_pos = np.random.randint(1, current_length - 1)
        else:
            deletion_pos = np.random.randint(0, current_length)
        
        # Shift elements to fill gap
        for pos in range(deletion_pos, current_length - 1):
            for phase in range(2):
                for dof in range(self.dof):
                    self.hosts[individual_idx, pos, phase, dof] = \
                        self.hosts[individual_idx, pos + 1, phase, dof]
        
        # Update length
        self.host_lengths[individual_idx] -= 1
        
        self.logger.debug(f"Deletion mutation: {current_length} -> {self.host_lengths[individual_idx]}")
        return True
    
    def _apply_phase_exchange_mutation(self, individual_idx):
        """Enhanced phase exchange with position weighting."""
        current_length = self.host_lengths[individual_idx]
        
        if current_length == 0:
            return False
        
        # Select position (weight towards middle positions)
        if current_length > 2:
            weights = np.ones(current_length)
            weights[1:-1] *= 2  # Double weight for middle positions
            weights = weights / np.sum(weights)
            position = np.random.choice(current_length, p=weights)
        else:
            position = np.random.randint(0, current_length)
        
        # Exchange phases for all DOFs at selected position
        for dof in range(self.dof):
            temp = self.hosts[individual_idx, position, 0, dof]
            self.hosts[individual_idx, position, 0, dof] = \
                self.hosts[individual_idx, position, 1, dof]
            self.hosts[individual_idx, position, 1, dof] = temp
        
        self.logger.debug(f"Phase exchange mutation at position {position}")
        return True
    
    def _apply_order_exchange_mutation(self, individual_idx):
        """Enhanced order exchange with distance-based selection."""
        current_length = self.host_lengths[individual_idx]
        
        if current_length < 2:
            return False
        
        # Select two positions with preference for distant positions
        if current_length > 3:
            # Prefer swapping positions that are not adjacent
            pos1 = np.random.randint(0, current_length)
            possible_pos2 = [i for i in range(current_length) if abs(i - pos1) > 1]
            if possible_pos2:
                pos2 = np.random.choice(possible_pos2)
            else:
                pos2 = (pos1 + 1) % current_length
        else:
            pos1, pos2 = np.random.choice(current_length, 2, replace=False)
        
        # Exchange positions for all phases and DOFs
        for phase in range(2):
            for dof in range(self.dof):
                temp = self.hosts[individual_idx, pos1, phase, dof]
                self.hosts[individual_idx, pos1, phase, dof] = \
                    self.hosts[individual_idx, pos2, phase, dof]
                self.hosts[individual_idx, pos2, phase, dof] = temp
        
        self.logger.debug(f"Order exchange mutation: positions {pos1} <-> {pos2}")
        return True
    
    def enhanced_fitness_evaluation(self, robot, prev_pos, curr_pos, prev_rot, curr_rot):
        """
        Enhanced fitness evaluation with operator effect attribution.
        
        Returns:
            Tuple of (fitness_values, performance_metrics)
        """
        # Get base fitness evaluation
        fitness_values = super().evaluate_fitness(robot, prev_pos, curr_pos, prev_rot, curr_rot)
        
        # Calculate additional performance metrics
        performance_metrics = self._calculate_performance_metrics(robot, fitness_values)
        
        # Update objective performance tracking
        if self.iteration < len(self.objective_performance["best_per_generation"]):
            for obj_idx in range(len(fitness_values)):
                # Update best fitness for this generation
                current_best = np.max(self.fitness[:self.gai+1, obj_idx])
                self.objective_performance["best_per_generation"][self.iteration, obj_idx] = current_best
                
                # Update mean and std
                current_mean = np.mean(self.fitness[:self.gai+1, obj_idx])
                current_std = np.std(self.fitness[:self.gai+1, obj_idx])
                self.objective_performance["mean_per_generation"][self.iteration, obj_idx] = current_mean
                self.objective_performance["std_per_generation"][self.iteration, obj_idx] = current_std
        
        # Update convergence monitoring
        self._update_convergence_monitoring(fitness_values, performance_metrics)
        
        return fitness_values, performance_metrics
    
    def _calculate_performance_metrics(self, robot, fitness_values):
        """Calculate additional performance metrics for analysis."""
        metrics = {}
        
        # Hypervolume approximation (simplified)
        if len(fitness_values) >= 3:
            metrics["hypervolume"] = np.prod(fitness_values[:3])
        else:
            metrics["hypervolume"] = np.prod(fitness_values)
        
        # Population diversity
        if self.iteration > 0:
            metrics["genotypic_diversity"] = self._calculate_genotypic_diversity()
            metrics["phenotypic_diversity"] = self._calculate_phenotypic_diversity()
        else:
            metrics["genotypic_diversity"] = 0
            metrics["phenotypic_diversity"] = 0
        
        # Stability metrics from robot
        if hasattr(robot, 'check_stability'):
            stability_metrics = robot.check_stability()
            metrics.update(stability_metrics)
        
        return metrics
    
    def _calculate_genotypic_diversity(self):
        """Calculate genotypic diversity based on chromosome lengths and content."""
        # Length diversity
        length_variance = np.var(self.host_lengths[:self.gai+1])
        
        # Content diversity (simplified - average pairwise differences)
        if self.gai > 0:
            total_diff = 0
            comparisons = 0
            
            for i in range(min(self.gai+1, 10)):  # Sample for efficiency
                for j in range(i+1, min(self.gai+1, 10)):
                    if i < self.gan and j < self.gan:
                        diff = self._chromosome_difference(i, j)
                        total_diff += diff
                        comparisons += 1
            
            content_diversity = total_diff / comparisons if comparisons > 0 else 0
        else:
            content_diversity = 0
        
        return length_variance + content_diversity
    
    def _chromosome_difference(self, idx1, idx2):
        """Calculate difference between two chromosomes."""
        min_length = min(self.host_lengths[idx1], self.host_lengths[idx2])
        
        if min_length == 0:
            return abs(self.host_lengths[idx1] - self.host_lengths[idx2])
        
        total_diff = 0
        for pos in range(min_length):
            for phase in range(2):
                for dof in range(self.dof):
                    diff = abs(self.hosts[idx1, pos, phase, dof] - 
                              self.hosts[idx2, pos, phase, dof])
                    total_diff += diff
        
        # Add length difference
        total_diff += abs(self.host_lengths[idx1] - self.host_lengths[idx2]) * 10
        
        return total_diff / (min_length * 2 * self.dof + 1)
    
    def _calculate_phenotypic_diversity(self):
        """Calculate phenotypic diversity based on fitness values."""
        if self.gai == 0:
            return 0
        
        # Calculate variance across all objectives for current population
        fitness_matrix = self.fitness[:self.gai+1, :]
        return np.mean(np.var(fitness_matrix, axis=0))
    
    def _update_convergence_monitoring(self, fitness_values, performance_metrics):
        """Update convergence monitoring with enhanced criteria."""
        current_hypervolume = performance_metrics.get("hypervolume", 0)
        self.convergence_monitor["hypervolume_history"].append(current_hypervolume)
        
        current_diversity = performance_metrics.get("phenotypic_diversity", 0)
        self.convergence_monitor["diversity_history"].append(current_diversity)
        
        # Check for convergence using multiple criteria
        window_size = 50  # τ parameter from strategic guidance
        epsilon_hv = 0.01  # εHV parameter
        epsilon_div = 0.1  # εD parameter
        
        if len(self.convergence_monitor["hypervolume_history"]) >= window_size:
            # Hypervolume stagnation check
            recent_hv = self.convergence_monitor["hypervolume_history"][-window_size:]
            hv_change = abs(recent_hv[-1] - recent_hv[0])
            
            # Diversity preservation check
            recent_div = self.convergence_monitor["diversity_history"][-window_size:]
            avg_diversity = np.mean(recent_div)
            
            # Progress rate check
            if len(self.objective_performance["best_per_generation"]) > window_size:
                recent_progress = self.objective_performance["best_per_generation"][self.iteration-window_size:self.iteration, 0]
                progress_rate = (recent_progress[-1] - recent_progress[0]) / window_size
                epsilon_lambda = 0.001  # ελ parameter
                
                # Check convergence criteria
                hv_stagnant = hv_change < epsilon_hv
                diversity_adequate = avg_diversity > epsilon_div
                progress_slow = progress_rate < epsilon_lambda
                
                if hv_stagnant and diversity_adequate and progress_slow:
                    if not self.convergence_monitor["convergence_achieved"]:
                        self.convergence_monitor["convergence_achieved"] = True
                        self.convergence_monitor["convergence_generation"] = self.iteration
                        self.logger.info(f"Convergence achieved at generation {self.iteration}")
    
    def get_operator_statistics_summary(self):
        """Get comprehensive operator statistics for analysis."""
        summary = {}
        
        for operator in self.operator_config.keys():
            apps = self.operator_statistics["applications"][operator]
            succ = self.operator_statistics["successes"][operator]
            improvements = self.operator_statistics["improvements"][operator]
            
            summary[operator] = {
                "applications": apps,
                "successes": succ,
                "success_rate": succ / apps if apps > 0 else 0,
                "average_improvement": np.mean(improvements) if improvements else 0,
                "improvement_std": np.std(improvements) if improvements else 0,
                "active_generations": len(set(self.operator_statistics["generation_history"][operator])),
                "total_improvement": sum(improvements)
            }
        
        return summary
    
    def save_enhanced_results(self, filepath):
        """Save comprehensive results including operator statistics."""
        results = {
            "configuration": {
                "operator_config": self.operator_config,
                "population_size": self.gan,
                "chromosome_length_range": [self.min_chromosome_length, self.max_chromosome_length],
                "generations_completed": self.iteration
            },
            "operator_statistics": self.get_operator_statistics_summary(),
            "convergence_info": self.convergence_monitor,
            "objective_performance": {
                "best_per_generation": self.objective_performance["best_per_generation"][:self.iteration].tolist(),
                "mean_per_generation": self.objective_performance["mean_per_generation"][:self.iteration].tolist(),
                "std_per_generation": self.objective_performance["std_per_generation"][:self.iteration].tolist()
            },
            "final_population_stats": {
                "mean_chromosome_length": np.mean(self.host_lengths),
                "chromosome_length_std": np.std(self.host_lengths),
                "length_distribution": np.bincount(self.host_lengths).tolist()
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Enhanced results saved to {filepath}")
        
        return results


# Additional utility functions for integration

def integrate_enhanced_vega_with_existing():
    """
    Integration guide for enhancing existing VEGA implementation.
    
    This function outlines the key modifications needed in the existing codebase.
    """
    modifications = {
        "src/evolution/vega.py": [
            "Add operator configuration methods",
            "Implement enhanced mutation tracking",
            "Add convergence monitoring",
            "Expand chromosome length constraints",
            "Add performance metrics calculation"
        ],
        "experiments/run_evolution.py": [
            "Add operator configuration loading",
            "Implement enhanced data logging",
            "Add statistical result compilation",
            "Integrate with ablation framework"
        ],
        "src/robot/leg_robot.py": [
            "Enhance stability checking",
            "Add performance metrics",
            "Improve motor control parameters"
        ]
    }
    
    return modifications


def create_operator_configuration_template():
    """Create a template for operator configuration files."""
    template = {
        "experiment_name": "ablation_study_baseline",
        "operator_settings": {
            "insertion": {
                "active": True,
                "probability": 0.15,
                "constraints": {
                    "max_chromosome_length": 8,
                    "conservative_bounds": True
                }
            },
            "deletion": {
                "active": True,
                "probability": 0.15,
                "constraints": {
                    "min_chromosome_length": 2,
                    "avoid_critical_positions": True
                }
            },
            "phase_exchange": {
                "active": True,
                "probability": 0.10,
                "constraints": {
                    "prefer_middle_positions": True
                }
            },
            "order_exchange": {
                "active": True,
                "probability": 0.10,
                "constraints": {
                    "prefer_distant_positions": True
                }
            }
        },
        "convergence_criteria": {
            "hypervolume_stagnation_threshold": 0.01,
            "diversity_preservation_threshold": 0.1,
            "progress_rate_threshold": 0.001,
            "window_size": 50
        },
        "tracking_settings": {
            "detailed_operator_stats": True,
            "performance_attribution": True,
            "convergence_monitoring": True
        }
    }
    
    return template


if __name__ == "__main__":
    # Example usage
    enhanced_vega = EnhancedVEGA(population_size=30, chromosome_length=8, generations=500)
    
    # Configure for ablation study (example: no insertion)
    enhanced_vega.configure_operators(
        active_operators=["deletion", "phase_exchange", "order_exchange"],
        operator_probabilities={"deletion": 0.15, "phase_exchange": 0.10, "order_exchange": 0.10}
    )
    
    print("Enhanced VEGA initialized successfully")
    print("Operator configuration:", enhanced_vega.operator_config)