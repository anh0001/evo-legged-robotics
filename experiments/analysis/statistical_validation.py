#!/usr/bin/env python3
"""
Statistical Validation and Analysis Framework
Implements comprehensive statistical testing as outlined in Table 5 of strategic guidance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency, mannwhitneyu
from statsmodels.stats.multivariate import multivariate_stats
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')


class StatisticalValidator:
    """
    Comprehensive statistical validation framework for evolutionary robotics experiments.
    Implements the statistical testing protocols from Table 5 in strategic guidance.
    """
    
    def __init__(self, significance_level=0.05, target_power=0.80, min_effect_size=0.5):
        self.alpha = significance_level
        self.power = target_power  
        self.min_effect_size = min_effect_size  # Cohen's d
        
        self.results_dir = f"results/statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define fitness objectives as in the paper
        self.fitness_objectives = [
            "forward_motion", "stability", "energy_efficiency", 
            "smoothness", "direction_control", "foot_contact"
        ]
        
        # Statistical test results storage
        self.test_results = {}
        
    def load_experimental_data(self, data_path):
        """Load experimental data from ablation studies."""
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
    
    def validate_experimental_design(self, df):
        """
        Validate that experimental design meets statistical requirements.
        
        Returns:
            dict: Validation results and recommendations
        """
        validation_results = {
            "sample_size_adequate": False,
            "replication_sufficient": False,
            "design_balanced": False,
            "recommendations": []
        }
        
        # Check sample size per configuration
        config_counts = df['config_id'].value_counts()
        min_samples = config_counts.min()
        
        if min_samples >= 30:
            validation_results["sample_size_adequate"] = True
        else:
            validation_results["recommendations"].append(
                f"Insufficient sample size: minimum {min_samples}, recommend ≥30 per configuration"
            )
        
        # Check for balanced design
        if config_counts.std() / config_counts.mean() < 0.1:
            validation_results["design_balanced"] = True
        else:
            validation_results["recommendations"].append(
                "Unbalanced design detected. Consider equal sample sizes across configurations."
            )
        
        # Check replication
        unique_runs = df.groupby('config_id')['run_id'].nunique()
        if unique_runs.min() >= 5:
            validation_results["replication_sufficient"] = True
        else:
            validation_results["recommendations"].append(
                "Insufficient replications. Recommend ≥5 independent runs per configuration."
            )
        
        return validation_results
    
    def perform_manova_analysis(self, df):
        """
        Perform Multivariate Analysis of Variance (MANOVA) as specified in Table 5.
        Tests: Do operator configurations differ in overall multi-objective performance?
        """
        print("Performing MANOVA Analysis...")
        
        # Prepare data - final fitness values for each objective
        dependent_vars = []
        for obj in self.fitness_objectives:
            final_col = f'final_{obj}_fitness'
            if final_col in df.columns:
                dependent_vars.append(final_col)
        
        if len(dependent_vars) < 2:
            print("Warning: Insufficient dependent variables for MANOVA")
            return None
        
        # Create formula for MANOVA
        dependent_formula = ' + '.join(dependent_vars)
        formula = f"{dependent_formula} ~ config_id"
        
        try:
            # Perform MANOVA
            manova = MANOVA.from_formula(formula, data=df)
            manova_results = manova.mv_test()
            
            # Extract results
            manova_summary = {
                "test_statistic": "Wilks_Lambda",
                "results": manova_results,
                "significant": False,
                "effect_sizes": {}
            }
            
            # Check significance
            if hasattr(manova_results, 'results'):
                for test_name, test_results in manova_results.results.items():
                    if 'config_id' in test_name:
                        p_value = test_results[0]['Pr > F']
                        manova_summary["p_value"] = p_value
                        manova_summary["significant"] = p_value < self.alpha
                        break
            
            self.test_results["manova"] = manova_summary
            
            print(f"MANOVA Results: {'Significant' if manova_summary['significant'] else 'Not Significant'}")
            if "p_value" in manova_summary:
                print(f"p-value: {manova_summary['p_value']:.6f}")
            
            return manova_summary
            
        except Exception as e:
            print(f"MANOVA analysis failed: {str(e)}")
            return None
    
    def perform_univariate_anova(self, df):
        """
        Perform univariate ANOVA for each objective separately.
        Includes post-hoc tests (Tukey's HSD) if significant.
        """
        print("Performing Univariate ANOVA Analysis...")
        
        anova_results = {}
        
        for objective in self.fitness_objectives:
            final_col = f'final_{objective}_fitness'
            if final_col not in df.columns:
                continue
            
            print(f"\nAnalyzing {objective}...")
            
            # Group data by configuration
            groups = [group[final_col].values for name, group in df.groupby('config_id')]
            group_names = [name for name, group in df.groupby('config_id')]
            
            # Perform one-way ANOVA
            f_statistic, p_value = f_oneway(*groups)
            
            # Calculate effect size (eta-squared)
            ss_between = sum(len(group) * (np.mean(group) - np.mean(df[final_col]))**2 for group in groups)
            ss_total = np.sum((df[final_col] - np.mean(df[final_col]))**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            anova_result = {
                "f_statistic": f_statistic,
                "p_value": p_value,
                "significant": p_value < self.alpha,
                "eta_squared": eta_squared,
                "groups": group_names,
                "group_means": [np.mean(group) for group in groups],
                "group_stds": [np.std(group) for group in groups]
            }
            
            # Perform post-hoc tests if significant
            if p_value < self.alpha:
                posthoc_results = self._perform_posthoc_tests(df, final_col, 'config_id')
                anova_result["posthoc"] = posthoc_results
            
            anova_results[objective] = anova_result
            
            print(f"F({len(groups)-1}, {len(df)-len(groups)}) = {f_statistic:.3f}, p = {p_value:.6f}")
            print(f"Effect size (η²) = {eta_squared:.3f}")
            
        self.test_results["anova"] = anova_results
        return anova_results
    
    def _perform_posthoc_tests(self, df, dependent_var, group_var):
        """Perform Tukey's HSD post-hoc tests."""
        try:
            tukey_results = pairwise_tukeyhsd(
                endog=df[dependent_var], 
                groups=df[group_var], 
                alpha=self.alpha
            )
            
            # Extract pairwise comparisons
            comparisons = []
            for i, (group1, group2, diff, p_adj, lower, upper) in enumerate(zip(
                tukey_results.data[:, 0], tukey_results.data[:, 1],
                tukey_results.data[:, 2], tukey_results.data[:, 3],
                tukey_results.data[:, 4], tukey_results.data[:, 5]
            )):
                comparisons.append({
                    "group1": group1,
                    "group2": group2, 
                    "mean_diff": diff,
                    "p_adj": p_adj,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "significant": p_adj < self.alpha
                })
            
            return {
                "method": "Tukey HSD",
                "comparisons": comparisons,
                "summary": str(tukey_results)
            }
            
        except Exception as e:
            print(f"Post-hoc test failed: {str(e)}")
            return None
    
    def calculate_effect_sizes(self, df):
        """
        Calculate Cohen's d effect sizes for all pairwise comparisons.
        """
        print("Calculating Effect Sizes...")
        
        effect_sizes = {}
        
        for objective in self.fitness_objectives:
            final_col = f'final_{objective}_fitness'
            if final_col not in df.columns:
                continue
            
            effect_sizes[objective] = {}
            
            # Get unique configurations
            configs = df['config_id'].unique()
            
            # Calculate pairwise effect sizes
            for i, config1 in enumerate(configs):
                for config2 in configs[i+1:]:
                    group1 = df[df['config_id'] == config1][final_col]
                    group2 = df[df['config_id'] == config2][final_col]
                    
                    # Cohen's d
                    cohens_d = self._calculate_cohens_d(group1, group2)
                    
                    # Interpretation
                    if abs(cohens_d) < 0.2:
                        interpretation = "negligible"
                    elif abs(cohens_d) < 0.5:
                        interpretation = "small"
                    elif abs(cohens_d) < 0.8:
                        interpretation = "medium"
                    else:
                        interpretation = "large"
                    
                    comparison_key = f"{config1}_vs_{config2}"
                    effect_sizes[objective][comparison_key] = {
                        "cohens_d": cohens_d,
                        "interpretation": interpretation,
                        "group1_mean": np.mean(group1),
                        "group2_mean": np.mean(group2),
                        "group1_std": np.std(group1),
                        "group2_std": np.std(group2)
                    }
        
        self.test_results["effect_sizes"] = effect_sizes
        return effect_sizes
    
    def _calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d
    
    def test_normality_assumptions(self, df):
        """Test normality assumptions for parametric tests."""
        print("Testing Normality Assumptions...")
        
        normality_results = {}
        
        for objective in self.fitness_objectives:
            final_col = f'final_{objective}_fitness'
            if final_col not in df.columns:
                continue
            
            # Shapiro-Wilk test for each configuration
            config_normality = {}
            for config in df['config_id'].unique():
                config_data = df[df['config_id'] == config][final_col]
                
                if len(config_data) >= 3:  # Minimum sample size for Shapiro-Wilk
                    stat, p_value = stats.shapiro(config_data)
                    config_normality[config] = {
                        "statistic": stat,
                        "p_value": p_value,
                        "normal": p_value > self.alpha
                    }
            
            normality_results[objective] = config_normality
        
        self.test_results["normality"] = normality_results
        return normality_results
    
    def test_homogeneity_of_variance(self, df):
        """Test homogeneity of variance assumptions (Levene's test)."""
        print("Testing Homogeneity of Variance...")
        
        variance_results = {}
        
        for objective in self.fitness_objectives:
            final_col = f'final_{objective}_fitness'
            if final_col not in df.columns:
                continue
            
            # Group data by configuration
            groups = [group[final_col].values for name, group in df.groupby('config_id')]
            
            if len(groups) >= 2:
                # Levene's test
                stat, p_value = stats.levene(*groups)
                
                variance_results[objective] = {
                    "levene_statistic": stat,
                    "p_value": p_value,
                    "homogeneous": p_value > self.alpha
                }
        
        self.test_results["homogeneity"] = variance_results
        return variance_results
    
    def power_analysis(self, df):
        """Perform post-hoc power analysis."""
        print("Performing Power Analysis...")
        
        from statsmodels.stats.power import ttest_power
        
        power_results = {}
        
        for objective in self.fitness_objectives:
            final_col = f'final_{objective}_fitness'
            if final_col not in df.columns:
                continue
            
            # Calculate observed effect sizes and power
            config_powers = {}
            configs = df['config_id'].unique()
            
            for i, config1 in enumerate(configs):
                for config2 in configs[i+1:]:
                    group1 = df[df['config_id'] == config1][final_col]
                    group2 = df[df['config_id'] == config2][final_col]
                    
                    if len(group1) >= 3 and len(group2) >= 3:
                        effect_size = self._calculate_cohens_d(group1, group2)
                        n = min(len(group1), len(group2))
                        
                        # Calculate achieved power
                        power = ttest_power(effect_size, n, self.alpha)
                        
                        comparison_key = f"{config1}_vs_{config2}"
                        config_powers[comparison_key] = {
                            "effect_size": effect_size,
                            "sample_size": n,
                            "achieved_power": power,
                            "adequate_power": power >= self.power
                        }
            
            power_results[objective] = config_powers
        
        self.test_results["power_analysis"] = power_results
        return power_results
    
    def generate_statistical_report(self, df):
        """Generate comprehensive statistical analysis report."""
        print("Generating Statistical Report...")
        
        # Validate experimental design
        validation = self.validate_experimental_design(df)
        
        # Perform all statistical tests
        manova_results = self.perform_manova_analysis(df)
        anova_results = self.perform_univariate_anova(df)
        effect_sizes = self.calculate_effect_sizes(df)
        normality_results = self.test_normality_assumptions(df)
        variance_results = self.test_homogeneity_of_variance(df)
        power_results = self.power_analysis(df)
        
        # Compile comprehensive report
        report = {
            "analysis_date": datetime.now().isoformat(),
            "sample_size": len(df),
            "num_configurations": df['config_id'].nunique(),
            "experimental_validation": validation,
            "manova_results": manova_results,
            "anova_results": anova_results,
            "effect_sizes": effect_sizes,
            "assumption_tests": {
                "normality": normality_results,
                "homogeneity": variance_results
            },
            "power_analysis": power_results,
            "recommendations": self._generate_recommendations()
        }
        
        # Save detailed results
        with open(f"{self.results_dir}/statistical_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary markdown report
        self._generate_markdown_report(report)
        
        # Create statistical visualizations
        self._create_statistical_visualizations(df)
        
        return report
    
    def _generate_recommendations(self):
        """Generate statistical recommendations based on results."""
        recommendations = []
        
        # Check if MANOVA was significant
        if "manova" in self.test_results and self.test_results["manova"].get("significant", False):
            recommendations.append(
                "MANOVA indicates significant overall differences between configurations. "
                "Proceed with univariate analyses and post-hoc tests."
            )
        
        # Check for adequate effect sizes
        if "effect_sizes" in self.test_results:
            large_effects = []
            for objective, comparisons in self.test_results["effect_sizes"].items():
                for comparison, data in comparisons.items():
                    if abs(data["cohens_d"]) >= 0.8:
                        large_effects.append(f"{objective}: {comparison}")
            
            if large_effects:
                recommendations.append(
                    f"Large effect sizes detected in: {', '.join(large_effects[:3])}"
                    + ("..." if len(large_effects) > 3 else "")
                )
        
        # Check assumption violations
        violations = []
        if "normality" in self.test_results:
            for objective, configs in self.test_results["normality"].items():
                non_normal = sum(1 for c in configs.values() if not c.get("normal", True))
                if non_normal > len(configs) * 0.3:  # >30% violations
                    violations.append(f"Normality violations in {objective}")
        
        if violations:
            recommendations.append(
                "Consider non-parametric alternatives due to assumption violations: "
                + ", ".join(violations)
            )
        
        return recommendations
    
    def _generate_markdown_report(self, report):
        """Generate human-readable markdown report."""
        content = []
        content.append("# Statistical Analysis Report\n")
        content.append(f"**Analysis Date:** {report['analysis_date']}\n")
        content.append(f"**Sample Size:** {report['sample_size']}\n")
        content.append(f"**Number of Configurations:** {report['num_configurations']}\n\n")
        
        # Experimental validation
        content.append("## Experimental Design Validation\n")
        validation = report['experimental_validation']
        content.append(f"- Sample size adequate: {validation['sample_size_adequate']}\n")
        content.append(f"- Design balanced: {validation['design_balanced']}\n")
        content.append(f"- Replication sufficient: {validation['replication_sufficient']}\n")
        
        if validation['recommendations']:
            content.append("\n**Recommendations:**\n")
            for rec in validation['recommendations']:
                content.append(f"- {rec}\n")
        
        # MANOVA results
        if report['manova_results']:
            content.append("\n## MANOVA Results\n")
            manova = report['manova_results']
            if 'p_value' in manova:
                content.append(f"- **p-value:** {manova['p_value']:.6f}\n")
                content.append(f"- **Significant:** {manova['significant']}\n")
            content.append("- **Interpretation:** ")
            if manova.get('significant', False):
                content.append("Significant overall differences detected between configurations.\n")
            else:
                content.append("No significant overall differences between configurations.\n")
        
        # Effect sizes summary
        content.append("\n## Effect Sizes Summary\n")
        if "effect_sizes" in report:
            for objective in self.fitness_objectives:
                if objective in report["effect_sizes"]:
                    large_effects = [
                        comp for comp, data in report["effect_sizes"][objective].items()
                        if abs(data["cohens_d"]) >= 0.8
                    ]
                    if large_effects:
                        content.append(f"- **{objective}:** {len(large_effects)} large effect(s)\n")
        
        # Recommendations
        content.append("\n## Recommendations\n")
        for rec in report['recommendations']:
            content.append(f"- {rec}\n")
        
        # Save markdown report
        with open(f"{self.results_dir}/statistical_report.md", 'w') as f:
            f.writelines(content)
    
    def _create_statistical_visualizations(self, df):
        """Create comprehensive statistical visualizations."""
        # Effect size heatmap
        plt.figure(figsize=(14, 10))
        
        # Create effect size matrix
        configs = sorted(df['config_id'].unique())
        effect_matrix = np.zeros((len(configs), len(configs)))
        
        if "effect_sizes" in self.test_results:
            # Use first objective for demonstration
            first_objective = self.fitness_objectives[0]
            if first_objective in self.test_results["effect_sizes"]:
                for i, config1 in enumerate(configs):
                    for j, config2 in enumerate(configs):
                        if i != j:
                            comparison_key = f"{config1}_vs_{config2}"
                            reverse_key = f"{config2}_vs_{config1}"
                            
                            if comparison_key in self.test_results["effect_sizes"][first_objective]:
                                effect_matrix[i, j] = abs(self.test_results["effect_sizes"][first_objective][comparison_key]["cohens_d"])
                            elif reverse_key in self.test_results["effect_sizes"][first_objective]:
                                effect_matrix[i, j] = abs(self.test_results["effect_sizes"][first_objective][reverse_key]["cohens_d"])
        
        sns.heatmap(effect_matrix, 
                   xticklabels=configs, 
                   yticklabels=configs,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   center=0.5)
        plt.title(f'Effect Sizes (Cohen\'s d) - {first_objective}')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/effect_sizes_heatmap.png", dpi=300)
        plt.close()
        
        # Box plots for each objective
        for objective in self.fitness_objectives:
            final_col = f'final_{objective}_fitness'
            if final_col in df.columns:
                plt.figure(figsize=(12, 6))
                df.boxplot(column=final_col, by='config_id', ax=plt.gca())
                plt.title(f'Distribution of {objective} by Configuration')
                plt.suptitle('')  # Remove automatic title
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{self.results_dir}/{objective}_boxplot.png", dpi=300)
                plt.close()


if __name__ == "__main__":
    # Example usage
    validator = StatisticalValidator()
    
    # Load example data (replace with actual data path)
    # df = validator.load_experimental_data("results/ablation_study_data.csv")
    # report = validator.generate_statistical_report(df)
    
    print("Statistical Validator initialized successfully")