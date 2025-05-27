#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis Framework
Implements Latin Hypercube Sampling and Response Surface Methodology
as specified in the strategic guidance document.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import json
import os
from datetime import datetime
import logging

class ParameterSensitivityAnalyzer:
    """
    Comprehensive parameter sensitivity analysis using Latin Hypercube Sampling
    and Response Surface Methodology for structural mutation operator optimization.
    """
    
    def __init__(self, base_config_path="config/sensitivity_config.json"):
        self.config = self._load_config(base_config_path)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results/sensitivity_analysis_{self.experiment_id}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Parameter ranges as specified in strategic guidance
        self.parameter_ranges = {
            "insertion_prob": [0.05, 0.25],
            "deletion_prob": [0.05, 0.25], 
            "phase_exchange_prob": [0.05, 0.25],
            "order_exchange_prob": [0.05, 0.25],
            "max_sequence_length": [4, 6],  # Testing GAL expansion
            "penalty_coefficient": [5.0, 20.0]
        }
        
        # Constraint: sum of operator probabilities <= 0.5
        self.probability_constraint = 0.5
        
        self._setup_logging()
        
    def _load_config(self, config_path):
        """Load configuration for sensitivity analysis."""
        default_config = {
            "num_samples": 100,  # Number of LHS samples
            "num_replications": 5,  # Replications per sample point
            "max_iterations": 200,  # Reduced for sensitivity analysis
            "response_variables": [
                "hypervolume", 
                "convergence_speed", 
                "diversity_metric",
                "stability_performance"
            ]
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging for sensitivity analysis."""
        self.logger = logging.getLogger('sensitivity_analyzer')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(f"{self.results_dir}/sensitivity_analysis.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def generate_lhs_samples(self, num_samples=None):
        """
        Generate Latin Hypercube Samples for parameter space exploration.
        Implements Equation 34 from the strategic guidance.
        """
        if num_samples is None:
            num_samples = self.config["num_samples"]
        
        # Number of parameters
        n_params = len(self.parameter_ranges)
        
        # Generate LHS samples in [0,1]^n space
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_params, seed=42)
        unit_samples = sampler.random(n=num_samples)
        
        # Transform to actual parameter ranges
        samples = []
        param_names = list(self.parameter_ranges.keys())
        
        for i, sample in enumerate(unit_samples):
            param_dict = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = self.parameter_ranges[param_name]
                param_dict[param_name] = min_val + sample[j] * (max_val - min_val)
            
            # Apply constraint: sum of operator probabilities <= 0.5
            prob_sum = (param_dict["insertion_prob"] + 
                       param_dict["deletion_prob"] + 
                       param_dict["phase_exchange_prob"] + 
                       param_dict["order_exchange_prob"])
            
            if prob_sum <= self.probability_constraint:
                param_dict["sample_id"] = i
                param_dict["probability_sum"] = prob_sum
                samples.append(param_dict)
        
        self.logger.info(f"Generated {len(samples)} valid LHS samples from {num_samples} initial samples")
        
        # Save samples
        samples_df = pd.DataFrame(samples)
        samples_df.to_csv(f"{self.results_dir}/lhs_samples.csv", index=False)
        
        return samples
    
    def run_sensitivity_experiment(self, parameter_set):
        """
        Run a single sensitivity experiment with given parameters.
        Returns performance metrics for response surface analysis.
        """
        try:
            # Import here to avoid circular imports
            from evolution.vega import VEGA
            from robot.leg_robot import LeggedRobot
            from simulation.environment import Environment
            
            # Setup experiment with parameter set
            env = Environment(render=False, terrain_type="flat")
            robot = LeggedRobot(client=env.client)
            env.add_robot(robot)
            
            # Configure VEGA with sensitivity parameters
            vega = VEGA(
                population_size=30,
                chromosome_length=int(parameter_set["max_sequence_length"]),
                generations=self.config["max_iterations"]
            )
            
            # Set operator probabilities
            vega.configure_operators(
                active_operators=["insertion", "deletion", "phase_exchange", "order_exchange"],
                operator_probabilities={
                    "insertion": parameter_set["insertion_prob"],
                    "deletion": parameter_set["deletion_prob"], 
                    "phase_exchange": parameter_set["phase_exchange_prob"],
                    "order_exchange": parameter_set["order_exchange_prob"]
                }
            )
            
            # Set penalty coefficient
            vega.penalty_coefficient = parameter_set["penalty_coefficient"]
            
            # Run abbreviated evolution
            results = self._run_abbreviated_evolution(vega, robot, env)
            
            # Calculate response variables
            response_metrics = self._calculate_response_metrics(results, vega)
            
            # Cleanup
            env.close()
            
            return response_metrics
            
        except Exception as e:
            self.logger.error(f"Error in sensitivity experiment: {str(e)}")
            return {"error": str(e)}
    
    def _run_abbreviated_evolution(self, vega, robot, env):
        """Run abbreviated evolution for sensitivity analysis."""
        # Simplified evolution loop for faster execution
        fitness_history = []
        diversity_history = []
        
        for iteration in range(self.config["max_iterations"]):
            # Quick fitness evaluation
            fitness_values = vega.evaluate_fitness_simplified(robot)
            fitness_history.append(fitness_values)
            
            # Track diversity
            diversity = vega.calculate_population_diversity()
            diversity_history.append(diversity)
            
            # Evolve population
            if iteration >= vega.gan:
                vega.evolve()
            
            vega.iteration = iteration
        
        return {
            "fitness_history": fitness_history,
            "diversity_history": diversity_history,
            "final_population": vega.hosts.copy()
        }
    
    def _calculate_response_metrics(self, results, vega):
        """Calculate response variables for sensitivity analysis."""
        fitness_history = np.array(results["fitness_history"])
        diversity_history = np.array(results["diversity_history"])
        
        # Hypervolume (approximate)
        if len(fitness_history) > 0:
            final_hypervolume = np.prod(np.max(fitness_history, axis=0))
        else:
            final_hypervolume = 0
        
        # Convergence speed (generations to reach 90% of final performance)
        convergence_speed = self._calculate_convergence_speed(fitness_history)
        
        # Diversity metric (final population diversity)
        final_diversity = diversity_history[-1] if len(diversity_history) > 0 else 0
        
        # Stability performance (average stability across all objectives)
        stability_performance = np.mean(fitness_history[:, 1]) if len(fitness_history) > 0 else 0
        
        return {
            "hypervolume": final_hypervolume,
            "convergence_speed": convergence_speed,
            "diversity_metric": final_diversity,
            "stability_performance": stability_performance
        }
    
    def _calculate_convergence_speed(self, fitness_history):
        """Calculate convergence speed metric."""
        if len(fitness_history) < 10:
            return len(fitness_history)
        
        # Use first objective (forward motion) for convergence calculation
        forward_fitness = fitness_history[:, 0]
        final_performance = forward_fitness[-1]
        target_performance = 0.9 * final_performance
        
        # Find first generation where 90% performance is reached
        for i, fitness in enumerate(forward_fitness):
            if fitness >= target_performance:
                return i
        
        return len(fitness_history)  # Didn't converge
    
    def run_full_sensitivity_analysis(self):
        """Run complete sensitivity analysis with LHS sampling."""
        self.logger.info("Starting comprehensive parameter sensitivity analysis")
        
        # Generate LHS samples
        lhs_samples = self.generate_lhs_samples()
        
        # Run experiments for each sample
        all_results = []
        
        for i, sample in enumerate(lhs_samples):
            self.logger.info(f"Running sensitivity experiment {i+1}/{len(lhs_samples)}")
            
            # Run multiple replications for statistical robustness
            sample_results = []
            for rep in range(self.config["num_replications"]):
                result = self.run_sensitivity_experiment(sample)
                if "error" not in result:
                    result.update(sample)  # Add parameter values
                    result["replication"] = rep
                    sample_results.append(result)
            
            all_results.extend(sample_results)
            
            # Save intermediate results
            if (i + 1) % 10 == 0:
                self._save_intermediate_results(all_results)
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{self.results_dir}/sensitivity_results.csv", index=False)
        
        # Perform response surface analysis
        rsm_results = self._perform_response_surface_analysis(results_df)
        
        # Generate sensitivity report
        self._generate_sensitivity_report(results_df, rsm_results)
        
        return results_df, rsm_results
    
    def _perform_response_surface_analysis(self, results_df):
        """
        Perform Response Surface Methodology analysis.
        Implements Equation 35 from strategic guidance.
        """
        self.logger.info("Performing Response Surface Methodology analysis")
        
        # Parameter columns
        param_cols = [col for col in results_df.columns if col in self.parameter_ranges.keys()]
        
        # Response variables
        response_vars = self.config["response_variables"]
        
        rsm_results = {}
        
        for response_var in response_vars:
            if response_var not in results_df.columns:
                continue
            
            self.logger.info(f"Analyzing response surface for {response_var}")
            
            # Prepare data
            X = results_df[param_cols].values
            y = results_df[response_var].values
            
            # Standardize inputs
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit response surface models
            models = {
                "linear": self._fit_linear_model(X_scaled, y),
                "quadratic": self._fit_quadratic_model(X_scaled, y),
                "random_forest": self._fit_random_forest_model(X_scaled, y)
            }
            
            # Find optimal parameters
            optimal_params = self._find_optimal_parameters(models["random_forest"], scaler, param_cols)
            
            # Calculate sensitivity indices
            sensitivity_indices = self._calculate_sensitivity_indices(X_scaled, y, param_cols)
            
            rsm_results[response_var] = {
                "models": models,
                "optimal_parameters": optimal_params,
                "sensitivity_indices": sensitivity_indices,
                "scaler": scaler
            }
        
        # Save RSM results
        self._save_rsm_results(rsm_results)
        
        return rsm_results
    
    def _fit_linear_model(self, X, y):
        """Fit linear response surface model."""
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        model.fit(X, y)
        
        return {
            "model": model,
            "r2_score": r2_score(y, model.predict(X)),
            "coefficients": model.coef_,
            "intercept": model.intercept_
        }
    
    def _fit_quadratic_model(self, X, y):
        """Fit quadratic response surface model with interaction terms."""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        
        # Create polynomial features (degree 2 for quadratic)
        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear", LinearRegression())
        ])
        
        pipeline.fit(X, y)
        
        return {
            "model": pipeline,
            "r2_score": r2_score(y, pipeline.predict(X)),
            "feature_names": pipeline["poly"].get_feature_names_out()
        }
    
    def _fit_random_forest_model(self, X, y):
        """Fit Random Forest model for non-linear response surface."""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return {
            "model": model,
            "r2_score": r2_score(y, model.predict(X)),
            "feature_importance": model.feature_importances_
        }
    
    def _find_optimal_parameters(self, rf_model, scaler, param_names):
        """Find optimal parameters using the trained model."""
        def objective(x_scaled):
            return -rf_model["model"].predict(x_scaled.reshape(1, -1))[0]
        
        # Optimization bounds (scaled space)
        bounds = [(-2, 2) for _ in range(len(param_names))]
        
        # Add constraint for probability sum
        def constraint(x_scaled):
            # Transform back to original space
            x_orig = scaler.inverse_transform(x_scaled.reshape(1, -1))[0]
            prob_indices = [i for i, name in enumerate(param_names) if "prob" in name]
            prob_sum = sum(x_orig[i] for i in prob_indices)
            return self.probability_constraint - prob_sum
        
        # Optimize
        result = minimize(
            objective, 
            x0=np.zeros(len(param_names)),
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraint},
            method="SLSQP"
        )
        
        # Transform back to original space
        optimal_scaled = result.x
        optimal_params = scaler.inverse_transform(optimal_scaled.reshape(1, -1))[0]
        
        return dict(zip(param_names, optimal_params))
    
    def _calculate_sensitivity_indices(self, X, y, param_names):
        """Calculate Sobol sensitivity indices."""
        from scipy.stats import pearsonr
        
        # Simplified sensitivity analysis using correlation
        sensitivity_indices = {}
        
        for i, param_name in enumerate(param_names):
            correlation, p_value = pearsonr(X[:, i], y)
            sensitivity_indices[param_name] = {
                "correlation": correlation,
                "p_value": p_value,
                "importance": abs(correlation)
            }
        
        return sensitivity_indices
    
    def _save_intermediate_results(self, results):
        """Save intermediate results during long runs."""
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(f"{self.results_dir}/intermediate_results.csv", index=False)
    
    def _save_rsm_results(self, rsm_results):
        """Save Response Surface Methodology results."""
        # Save optimal parameters and sensitivity indices
        summary = {}
        
        for response_var, analysis in rsm_results.items():
            summary[response_var] = {
                "optimal_parameters": analysis["optimal_parameters"],
                "sensitivity_indices": analysis["sensitivity_indices"]
            }
        
        with open(f"{self.results_dir}/rsm_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _generate_sensitivity_report(self, results_df, rsm_results):
        """Generate comprehensive sensitivity analysis report."""
        report_content = []
        report_content.append("# Parameter Sensitivity Analysis Report\n")
        report_content.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_content.append(f"Number of samples: {len(results_df)}\n\n")
        
        # Parameter ranges
        report_content.append("## Parameter Ranges Tested\n")
        for param, (min_val, max_val) in self.parameter_ranges.items():
            report_content.append(f"- {param}: [{min_val:.3f}, {max_val:.3f}]\n")
        
        report_content.append(f"\nConstraint: Sum of operator probabilities ≤ {self.probability_constraint}\n\n")
        
        # Response surface analysis results
        for response_var, analysis in rsm_results.items():
            report_content.append(f"## Response Variable: {response_var}\n")
            
            # Optimal parameters
            report_content.append("### Optimal Parameters\n")
            for param, value in analysis["optimal_parameters"].items():
                report_content.append(f"- {param}: {value:.4f}\n")
            
            # Sensitivity indices
            report_content.append("\n### Sensitivity Indices\n")
            sens_sorted = sorted(
                analysis["sensitivity_indices"].items(), 
                key=lambda x: x[1]["importance"], 
                reverse=True
            )
            
            for param, indices in sens_sorted:
                report_content.append(
                    f"- {param}: Correlation = {indices['correlation']:.4f}, "
                    f"p-value = {indices['p_value']:.4f}\n"
                )
            
            report_content.append("\n")
        
        # Save report
        with open(f"{self.results_dir}/sensitivity_report.md", 'w') as f:
            f.writelines(report_content)
        
        # Create visualizations
        self._create_sensitivity_visualizations(results_df, rsm_results)
    
    def _create_sensitivity_visualizations(self, results_df, rsm_results):
        """Create comprehensive sensitivity analysis visualizations."""
        # Parameter correlation heatmap
        param_cols = [col for col in results_df.columns if col in self.parameter_ranges.keys()]
        response_cols = self.config["response_variables"]
        
        # Correlation matrix
        plt.figure(figsize=(12, 8))
        correlation_matrix = results_df[param_cols + response_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Parameter-Response Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/correlation_matrix.png", dpi=300)
        plt.close()
        
        # Response surface plots
        for response_var in response_cols:
            if response_var not in results_df.columns:
                continue
            
            # Create pairwise parameter plots
            for param1, param2 in combinations(param_cols[:4], 2):  # Limit to operator probabilities
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(
                    results_df[param1], 
                    results_df[param2], 
                    c=results_df[response_var],
                    alpha=0.6,
                    cmap='viridis'
                )
                plt.colorbar(scatter, label=response_var)
                plt.xlabel(param1)
                plt.ylabel(param2)
                plt.title(f'{response_var} Response Surface: {param1} vs {param2}')
                plt.tight_layout()
                plt.savefig(f"{self.results_dir}/{response_var}_{param1}_{param2}.png", dpi=300)
                plt.close()
        
        # Sensitivity indices bar plots
        for response_var, analysis in rsm_results.items():
            plt.figure(figsize=(10, 6))
            
            params = list(analysis["sensitivity_indices"].keys())
            importances = [analysis["sensitivity_indices"][p]["importance"] for p in params]
            
            plt.bar(params, importances)
            plt.title(f'Parameter Sensitivity for {response_var}')
            plt.ylabel('Absolute Correlation')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/{response_var}_sensitivity.png", dpi=300)
            plt.close()


if __name__ == "__main__":
    # Example usage
    analyzer = ParameterSensitivityAnalyzer()
    
    # Generate samples for testing
    samples = analyzer.generate_lhs_samples(num_samples=20)
    print(f"Generated {len(samples)} LHS samples")
    
    # Run full analysis (commented out for testing)
    # results_df, rsm_results = analyzer.run_full_sensitivity_analysis()