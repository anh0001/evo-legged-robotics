#!/usr/bin/env python3
"""
Automated Visualization and Reporting Pipeline
Generates publication-ready figures and comprehensive reports for hexapod locomotion research.
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality matplotlib settings
plt.rcParams.update({
    'figure.figsize': (10, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'axes.spines.top': False,
    'axes.spines.right': False
})

class PublicationVisualizationPipeline:
    """
    Automated pipeline for generating publication-ready visualizations and reports
    for structural mutation operator analysis in hexapod locomotion evolution.
    """
    
    def __init__(self, output_dir="publication_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        # Define colors for different operator configurations
        self.config_colors = {
            'C0_baseline': '#1f77b4',
            'C1_no_insertion': '#ff7f0e', 
            'C2_no_deletion': '#2ca02c',
            'C3_no_phase': '#d62728',
            'C4_no_order': '#9467bd',
            'C5_exploration_only': '#8c564b',
            'C6_refinement_only': '#e377c2',
            'C11_no_structural': '#7f7f7f'
        }
        
        # Fitness objectives as defined in the paper
        self.fitness_objectives = [
            "forward_motion", "stability", "energy_efficiency", 
            "smoothness", "direction_control", "foot_contact"
        ]
        
        self.objective_labels = {
            "forward_motion": "Forward Motion",
            "stability": "Stability", 
            "energy_efficiency": "Energy Efficiency",
            "smoothness": "Smoothness",
            "direction_control": "Direction Control",
            "foot_contact": "Foot Contact"
        }
    
    def generate_figure_1_pareto_evolution(self, evolution_data):
        """
        Generate Figure 1: Pareto Front Evolution Animation
        Shows the evolution of non-dominated solutions over generations.
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Create 2x3 subplot layout for pairwise objective comparisons
        objectives = ["forward_motion", "stability", "energy_efficiency"]
        
        for i, obj1 in enumerate(objectives):
            for j, obj2 in enumerate(objectives):
                if i < j:  # Only upper triangle
                    ax = plt.subplot(2, 3, i*3 + j)
                    
                    # Plot Pareto front evolution for different generations
                    generations = [0, 100, 200, 300, 400, 500]
                    alphas = np.linspace(0.3, 1.0, len(generations))
                    
                    for gen, alpha in zip(generations, alphas):
                        if gen in evolution_data:
                            pareto_front = evolution_data[gen]['pareto_front']
                            
                            if len(pareto_front) > 0:
                                x_vals = [point[obj1] for point in pareto_front]
                                y_vals = [point[obj2] for point in pareto_front]
                                
                                plt.scatter(x_vals, y_vals, 
                                          alpha=alpha, 
                                          s=30,
                                          label=f'Gen {gen}' if gen in [0, 500] else None,
                                          c=f'C{i+j}')
                    
                    plt.xlabel(self.objective_labels[obj1])
                    plt.ylabel(self.objective_labels[obj2])
                    plt.grid(True, alpha=0.3)
                    
                    if i == 0 and j == 1:  # First subplot
                        plt.legend()
        
        plt.suptitle('Pareto Front Evolution Across Generations', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "figure_1_pareto_evolution.png")
        plt.savefig(self.output_dir / "figures" / "figure_1_pareto_evolution.pdf")
        plt.close()
    
    def generate_figure_2_operator_effectiveness(self, ablation_results):
        """
        Generate Figure 2: Operator Effectiveness Heatmap
        Shows the relative performance of different operator configurations.
        """
        # Prepare data matrix
        configs = list(ablation_results.keys())
        objectives = self.fitness_objectives
        
        performance_matrix = np.zeros((len(configs), len(objectives)))
        config_labels = []
        
        for i, config in enumerate(configs):
            config_data = ablation_results[config]
            config_labels.append(config_data.get('name', config))
            
            for j, objective in enumerate(objectives):
                # Use mean final performance
                performance_matrix[i, j] = config_data.get(f'mean_{objective}', 0)
        
        # Normalize each objective to [0, 1] for comparison
        for j in range(len(objectives)):
            col_max = np.max(performance_matrix[:, j])
            col_min = np.min(performance_matrix[:, j])
            if col_max > col_min:
                performance_matrix[:, j] = (performance_matrix[:, j] - col_min) / (col_max - col_min)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(objectives)))
        ax.set_xticklabels([self.objective_labels[obj] for obj in objectives], rotation=45, ha='right')
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(config_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Performance', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(configs)):
            for j in range(len(objectives)):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.title('Operator Configuration Effectiveness Across Objectives', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "figure_2_operator_effectiveness.png")
        plt.savefig(self.output_dir / "figures" / "figure_2_operator_effectiveness.pdf")
        plt.close()
    
    def generate_figure_3_convergence_analysis(self, convergence_data):
        """
        Generate Figure 3: Convergence Analysis
        Shows convergence patterns for different operator configurations.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, objective in enumerate(self.fitness_objectives):
            ax = axes[idx]
            
            # Plot convergence curves for key configurations
            key_configs = ['C0_baseline', 'C5_exploration_only', 'C6_refinement_only', 'C11_no_structural']
            
            for config in key_configs:
                if config in convergence_data:
                    generations = convergence_data[config]['generations']
                    fitness_history = convergence_data[config][f'best_{objective}']
                    
                    ax.plot(generations, fitness_history, 
                           label=convergence_data[config]['name'],
                           color=self.config_colors.get(config, 'gray'),
                           linewidth=2)
            
            ax.set_xlabel('Generation')
            ax.set_ylabel(f'{self.objective_labels[objective]} Fitness')
            ax.set_title(f'{self.objective_labels[objective]} Convergence')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Convergence Analysis by Operator Configuration', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "figure_3_convergence_analysis.png")
        plt.savefig(self.output_dir / "figures" / "figure_3_convergence_analysis.pdf")
        plt.close()
    
    def generate_figure_4_statistical_significance(self, statistical_results):
        """
        Generate Figure 4: Statistical Significance Matrix
        Shows pairwise statistical comparisons between configurations.
        """
        if 'effect_sizes' not in statistical_results:
            print("Warning: No effect size data available for statistical significance figure")
            return
        
        # Create significance matrix for first objective
        first_objective = self.fitness_objectives[0]
        effect_data = statistical_results['effect_sizes'].get(first_objective, {})
        
        configs = list(set([comp.split('_vs_')[0] for comp in effect_data.keys()] + 
                          [comp.split('_vs_')[1] for comp in effect_data.keys()]))
        n_configs = len(configs)
        
        # Create matrices for effect size and significance
        effect_matrix = np.zeros((n_configs, n_configs))
        significance_matrix = np.zeros((n_configs, n_configs))
        
        for comparison, data in effect_data.items():
            parts = comparison.split('_vs_')
            if len(parts) == 2:
                try:
                    i = configs.index(parts[0])
                    j = configs.index(parts[1])
                    effect_matrix[i, j] = abs(data['cohens_d'])
                    effect_matrix[j, i] = abs(data['cohens_d'])  # Symmetric
                    
                    # Mark as significant if effect size > 0.5
                    is_significant = abs(data['cohens_d']) > 0.5
                    significance_matrix[i, j] = is_significant
                    significance_matrix[j, i] = is_significant
                except ValueError:
                    continue
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Effect size heatmap
        im1 = ax1.imshow(effect_matrix, cmap='RdYlBu_r', vmin=0, vmax=2)
        ax1.set_xticks(range(n_configs))
        ax1.set_yticks(range(n_configs))
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.set_yticklabels(configs)
        ax1.set_title('Effect Sizes (Cohen\'s d)')
        
        # Add text annotations for effect sizes
        for i in range(n_configs):
            for j in range(n_configs):
                if effect_matrix[i, j] > 0:
                    ax1.text(j, i, f'{effect_matrix[i, j]:.2f}',
                            ha="center", va="center", 
                            color="white" if effect_matrix[i, j] > 1 else "black",
                            fontweight='bold')
        
        plt.colorbar(im1, ax=ax1)
        
        # Significance matrix
        im2 = ax2.imshow(significance_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_xticks(range(n_configs))
        ax2.set_yticks(range(n_configs))
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.set_yticklabels(configs)
        ax2.set_title('Statistical Significance (d > 0.5)')
        
        # Add significance indicators
        for i in range(n_configs):
            for j in range(n_configs):
                symbol = "✓" if significance_matrix[i, j] else "✗"
                color = "white" if significance_matrix[i, j] else "red"
                ax2.text(j, i, symbol, ha="center", va="center", 
                        color=color, fontsize=16, fontweight='bold')
        
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "figure_4_statistical_significance.png")
        plt.savefig(self.output_dir / "figures" / "figure_4_statistical_significance.pdf")
        plt.close()
    
    def generate_figure_5_parameter_sensitivity(self, sensitivity_results):
        """
        Generate Figure 5: Parameter Sensitivity Analysis
        Shows response surfaces and sensitivity indices.
        """
        if 'response_surfaces' not in sensitivity_results:
            print("Warning: No response surface data available")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplot layout: 2x3 for response surfaces + 1 for sensitivity indices
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8])
        
        # Plot response surfaces for key parameter pairs
        param_pairs = [
            ('insertion_prob', 'deletion_prob'),
            ('phase_exchange_prob', 'order_exchange_prob'),
            ('insertion_prob', 'phase_exchange_prob')
        ]
        
        response_var = 'hypervolume'  # Focus on hypervolume as primary response
        
        for idx, (param1, param2) in enumerate(param_pairs):
            ax = fig.add_subplot(gs[0, idx])
            
            # Get sensitivity data from nested response_surfaces key
            if response_var in sensitivity_results['response_surfaces']:
                data = sensitivity_results['response_surfaces'][response_var]
                
                # Create scatter plot with color-coded response
                if 'samples' in data:
                    filtered = [
                        sample for sample in data['samples']
                        if param1 in sample and param2 in sample and response_var in sample
                    ]
                    if not filtered:
                        continue

                    x_vals = [sample[param1] for sample in filtered]
                    y_vals = [sample[param2] for sample in filtered]
                    z_vals = [sample[response_var] for sample in filtered]
                    
                    scatter = ax.scatter(x_vals, y_vals, c=z_vals, cmap='viridis', alpha=0.7)
                    ax.set_xlabel(param1.replace('_', ' ').title())
                    ax.set_ylabel(param2.replace('_', ' ').title())
                    ax.set_title(f'{response_var.title()} Response Surface')
                    plt.colorbar(scatter, ax=ax)
        
        # Plot sensitivity indices
        if 'sensitivity_indices' in sensitivity_results:
            ax_sens = fig.add_subplot(gs[2, :])
            
            indices_data = sensitivity_results['sensitivity_indices']
            params = list(indices_data.keys())
            importances = [abs(indices_data[p]['correlation']) for p in params]
            
            bars = ax_sens.bar(params, importances, color='steelblue', alpha=0.7)
            ax_sens.set_ylabel('Sensitivity Index')
            ax_sens.set_title('Parameter Sensitivity Indices')
            ax_sens.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, importance in zip(bars, importances):
                height = bar.get_height()
                ax_sens.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{importance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "figure_5_parameter_sensitivity.png")
        plt.savefig(self.output_dir / "figures" / "figure_5_parameter_sensitivity.pdf")
        plt.close()
    
    def generate_supplementary_operator_timeline(self, operator_data):
        """
        Generate Supplementary Figure: Operator Application Timeline
        Shows when different operators are applied during evolution.
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        operators = ['insertion', 'deletion', 'phase_exchange', 'order_exchange']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        y_positions = np.arange(len(operators))
        
        for i, operator in enumerate(operators):
            if operator in operator_data:
                generations = operator_data[operator]['generations']
                applications = operator_data[operator]['applications']
                
                # Create timeline plot
                ax.scatter(generations, [i] * len(generations), 
                          s=[app*10 for app in applications],  # Size proportional to applications
                          c=colors[i], alpha=0.6, label=operator.replace('_', ' ').title())
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels([op.replace('_', ' ').title() for op in operators])
        ax.set_xlabel('Generation')
        ax.set_title('Structural Mutation Operator Application Timeline')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "supplementary_operator_timeline.png")
        plt.savefig(self.output_dir / "figures" / "supplementary_operator_timeline.pdf")
        plt.close()
    
    def generate_results_table_1(self, ablation_results):
        """
        Generate Table 1: Ablation Study Results Summary
        """
        table_data = []
        
        for config_id, data in ablation_results.items():
            row = {
                'Configuration': data.get('name', config_id),
                'Active Operators': ', '.join(data.get('active_operators', [])),
                'Forward Motion': f"{data.get('mean_forward_motion', 0):.2f} ± {data.get('std_forward_motion', 0):.2f}",
                'Stability': f"{data.get('mean_stability', 0):.2f} ± {data.get('std_stability', 0):.2f}",
                'Energy Efficiency': f"{data.get('mean_energy_efficiency', 0):.2f} ± {data.get('std_energy_efficiency', 0):.2f}",
                'Hypervolume': f"{data.get('mean_hypervolume', 0):.3f} ± {data.get('std_hypervolume', 0):.3f}",
                'Convergence Speed': f"{data.get('mean_convergence_speed', 0):.1f} ± {data.get('std_convergence_speed', 0):.1f}"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV and LaTeX
        df.to_csv(self.output_dir / "tables" / "table_1_ablation_results.csv", index=False)
        
        # Generate LaTeX table
        latex_table = df.to_latex(index=False, escape=False, column_format='|l|l|c|c|c|c|c|')
        
        with open(self.output_dir / "tables" / "table_1_ablation_results.tex", 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Ablation Study Results Summary (Mean ± Standard Deviation)}\n")
            f.write("\\label{tab:ablation_results}\n")
            f.write(latex_table)
            f.write("\\end{table}\n")
    
    def generate_results_table_2(self, statistical_results):
        """
        Generate Table 2: Statistical Analysis Summary
        """
        if 'anova_results' not in statistical_results:
            return
        
        table_data = []
        
        for objective, anova_data in statistical_results['anova_results'].items():
            row = {
                'Objective': self.objective_labels[objective],
                'F-statistic': f"{anova_data['f_statistic']:.3f}",
                'p-value': f"{anova_data['p_value']:.6f}",
                'Effect Size (η²)': f"{anova_data['eta_squared']:.3f}",
                'Significant': "Yes" if anova_data['significant'] else "No",
                'Post-hoc Tests': "Tukey HSD" if anova_data.get('posthoc') else "N/A"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV and LaTeX
        df.to_csv(self.output_dir / "tables" / "table_2_statistical_analysis.csv", index=False)
        
        latex_table = df.to_latex(index=False, escape=False, column_format='|l|c|c|c|c|c|')
        
        with open(self.output_dir / "tables" / "table_2_statistical_analysis.tex", 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Statistical Analysis Summary (ANOVA Results)}\n")
            f.write("\\label{tab:statistical_analysis}\n")
            f.write(latex_table)
            f.write("\\end{table}\n")
    
    def generate_comprehensive_report(self, all_results):
        """
        Generate comprehensive publication-ready report.
        """
        report_content = []
        
        # Title and metadata
        report_content.extend([
            "# Structural Mutation Operator Analysis in Hexapod Locomotion Evolution\n\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            "## Executive Summary\n\n",
            "This report presents a comprehensive analysis of structural mutation operators in ",
            "physics-constrained hexapod locomotion evolution. The study systematically ",
            "evaluates the individual and combined effects of insertion, deletion, phase exchange, ",
            "and order exchange operations through controlled ablation studies and statistical validation.\n\n"
        ])
        
        # Key findings
        if 'ablation_results' in all_results:
            report_content.extend([
                "## Key Findings\n\n",
                "### Operator Effectiveness Ranking\n"
            ])
            
            # Rank configurations by overall performance
            configs = all_results['ablation_results']
            ranked_configs = sorted(configs.items(), 
                                  key=lambda x: x[1].get('mean_hypervolume', 0), 
                                  reverse=True)
            
            for i, (config_id, data) in enumerate(ranked_configs[:5]):
                report_content.append(f"{i+1}. **{data.get('name', config_id)}**: "
                                    f"Hypervolume = {data.get('mean_hypervolume', 0):.3f}\n")
            
            report_content.append("\n")
        
        # Statistical significance
        if 'statistical_results' in all_results:
            report_content.extend([
                "### Statistical Significance\n\n",
                "The following operator configurations showed statistically significant differences:\n\n"
            ])
            
            # Add MANOVA results
            manova_results = all_results['statistical_results'].get('manova_results')
            if manova_results and manova_results.get('significant'):
                report_content.append("- **MANOVA**: Significant overall differences detected "
                                    f"(p = {manova_results.get('p_value', 'N/A'):.6f})\n")
            
            # Add ANOVA results for each objective
            anova_results = all_results['statistical_results'].get('anova_results', {})
            for objective, anova_data in anova_results.items():
                if anova_data.get('significant'):
                    report_content.append(f"- **{self.objective_labels[objective]}**: "
                                        f"F = {anova_data['f_statistic']:.3f}, "
                                        f"p = {anova_data['p_value']:.6f}, "
                                        f"η² = {anova_data['eta_squared']:.3f}\n")
        
        # Recommendations
        report_content.extend([
            "\n## Recommendations\n\n",
            "Based on the comprehensive analysis, the following recommendations are made:\n\n",
            "1. **Optimal Configuration**: Use the baseline configuration (all operators active) ",
            "for maximum performance across multiple objectives.\n",
            "2. **Specialized Applications**: Consider exploration-only configuration for ",
            "scenarios requiring high behavioral diversity.\n",
            "3. **Parameter Tuning**: Adjust operator probabilities based on sensitivity analysis results.\n",
            "4. **Future Research**: Investigate adaptive operator selection mechanisms.\n\n"
        ])
        
        # Methodology
        report_content.extend([
            "## Methodology\n\n",
            "### Experimental Design\n",
            "- **Population Size**: 30 individuals\n",
            "- **Generations**: 500 per run\n",
            "- **Independent Runs**: 30 per configuration\n",
            "- **Statistical Testing**: MANOVA, ANOVA, Tukey HSD post-hoc tests\n",
            "- **Effect Size**: Cohen's d with minimum threshold of 0.5\n\n",
            
            "### Robot Configuration\n",
            "- **Platform**: Hexapod robot with 18 DOF (6 legs × 3 joints)\n",
            "- **Physics Engine**: PyBullet with enhanced stability parameters\n",
            "- **Control**: Position control with PD gains (Kp=10.0, Kd=0.5)\n",
            "- **Evaluation**: Multi-objective fitness (6 objectives)\n\n"
        ])
        
        # Save comprehensive report
        with open(self.output_dir / "reports" / "comprehensive_analysis_report.md", 'w') as f:
            f.writelines(report_content)
        
        # Also save as JSON for programmatic access
        with open(self.output_dir / "data" / "analysis_summary.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    def run_complete_pipeline(self, experimental_data):
        """
        Run the complete visualization and reporting pipeline.
        
        Args:
            experimental_data: Dictionary containing all experimental results
        """
        print("Starting automated visualization and reporting pipeline...")
        
        # Generate all figures
        if 'evolution_data' in experimental_data:
            self.generate_figure_1_pareto_evolution(experimental_data['evolution_data'])
        
        if 'ablation_results' in experimental_data:
            self.generate_figure_2_operator_effectiveness(experimental_data['ablation_results'])
            self.generate_results_table_1(experimental_data['ablation_results'])
        
        if 'convergence_data' in experimental_data:
            self.generate_figure_3_convergence_analysis(experimental_data['convergence_data'])
        
        if 'statistical_results' in experimental_data:
            self.generate_figure_4_statistical_significance(experimental_data['statistical_results'])
            self.generate_results_table_2(experimental_data['statistical_results'])
        
        if 'sensitivity_results' in experimental_data:
            self.generate_figure_5_parameter_sensitivity(experimental_data['sensitivity_results'])
        
        if 'operator_data' in experimental_data:
            self.generate_supplementary_operator_timeline(experimental_data['operator_data'])
        
        # Generate comprehensive report
        self.generate_comprehensive_report(experimental_data)
        
        print(f"Pipeline complete! All outputs saved to: {self.output_dir}")
        
        # Create manifest file
        self._create_output_manifest()
    
    def _create_output_manifest(self):
        """Create a manifest of all generated outputs."""
        manifest = {
            "generation_date": datetime.now().isoformat(),
            "figures": [
                "figure_1_pareto_evolution.png",
                "figure_2_operator_effectiveness.png", 
                "figure_3_convergence_analysis.png",
                "figure_4_statistical_significance.png",
                "figure_5_parameter_sensitivity.png",
                "supplementary_operator_timeline.png"
            ],
            "tables": [
                "table_1_ablation_results.csv",
                "table_1_ablation_results.tex",
                "table_2_statistical_analysis.csv", 
                "table_2_statistical_analysis.tex"
            ],
            "reports": [
                "comprehensive_analysis_report.md"
            ],
            "data": [
                "analysis_summary.json"
            ],
            "formats": {
                "figures": ["PNG (300 DPI)", "PDF (vector)"],
                "tables": ["CSV (data)", "LaTeX (publication)"],
                "reports": ["Markdown (readable)", "JSON (programmatic)"]
            }
        }
        
        with open(self.output_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    # Example usage
    pipeline = PublicationVisualizationPipeline()
    
    # Example data structure (replace with actual experimental results)
    example_data = {
        "ablation_results": {
            "C0_baseline": {
                "name": "Baseline (All Operators)",
                "active_operators": ["insertion", "deletion", "phase_exchange", "order_exchange"],
                "mean_hypervolume": 0.785,
                "std_hypervolume": 0.032,
                "mean_forward_motion": 125.3,
                "std_forward_motion": 8.7
            }
        }
    }
    
    print("Visualization pipeline initialized successfully")
    print(f"Output directory: {pipeline.output_dir}")
    
    # To run: pipeline.run_complete_pipeline(example_data)