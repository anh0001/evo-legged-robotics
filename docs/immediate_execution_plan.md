### **Step 1: Quick Setup (10 minutes)**

```bash
# 1. Save the validation script
# Copy the "Quick Start Validation Script" above to: scripts/validate_framework.py

# 2. Save the configuration  
# Copy the "Experiment Configuration" above to: experiments/configs/ablation_configs.yaml

# 3. Make the script executable
chmod +x scripts/validate_framework.py

# 4. Run validation
python scripts/validate_framework.py
```

### **Step 2: Quick Test Run (30 minutes)**

```bash
# Test the enhanced VEGA with minimal settings
python experiments/core/run_evolution.py --quick-test --max-iterations=100

# Test ablation framework with 2 configurations
python experiments/studies/ablation_study.py --configs=minimal --runs=3
```

### **Step 3: Production Runs (2-3 days)**

```bash
# Option A: Full sequential run
python experiments/studies/ablation_study.py --full-study --sequential

# Option B: Parallel run (recommended)
python experiments/studies/ablation_study.py --full-study --parallel --max-workers=8

# Option C: HPC cluster run
python experiments/automation/protocol_manager.py --submit-cluster --queue=long
```

### **Step 4: Analysis and Visualization (4 hours)**

```bash
# Statistical analysis
python experiments/analysis/statistical_validation.py --full-analysis

# Generate publication figures
python experiments/visualization/publication_plots.py --all-figures --publication-ready

# Create comprehensive report
python experiments/visualization/report_generator.py --latex-output --include-appendix
```

## **Expected Outputs**

1. **Ablation Results**: 12 configurations × 30 runs = 360 experiments
2. **Statistical Analysis**: MANOVA, ANOVA, effect sizes, assumption tests
3. **Publication Figures**: 5 main figures + supplementary materials
4. **LaTeX Report**: Camera-ready manuscript section

## **Resource Requirements**

- **CPU Time**: ~260 CPU hours (11 days sequential, 1.5 days with 8 cores)
- **Memory**: ~4GB per parallel process  
- **Storage**: ~50GB for complete dataset
- **Time to Results**: 3-5 days with proper parallelization

## **Troubleshooting Common Issues**

1. **PyBullet GUI Issues**: Use `--no-render` flag for headless operation
2. **Memory Constraints**: Reduce `max_concurrent_experiments` in config
3. **Long Runtime**: Start with `--quick-test` to validate setup
4. **Missing Dependencies**: Run `pip install -r requirements.txt`

**Start with the validation script first** - it will catch any setup issues before you invest time in the full experimental runs.