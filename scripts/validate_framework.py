#!/usr/bin/env python3
"""
Quick Start Validation Script
Validates the experimental framework and runs a minimal test study.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def setup_environment():
    """Setup the experimental environment."""
    print("🔧 Setting up experimental environment...")
    
    # Create necessary directories
    dirs = [
        "experiments/core",
        "experiments/studies", 
        "experiments/automation",
        "experiments/analysis",
        "experiments/visualization",
        "experiments/configs",
        "results/validation",
        "logs/validation"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create __init__.py files for Python packages
        if 'experiments/' in dir_path:
            (Path(dir_path) / "__init__.py").touch()
    
    print("✅ Directory structure created")

def validate_dependencies():
    """Validate all required dependencies."""
    print("📦 Validating dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'scipy', 'sklearn', 'tensorflow', 'pybullet'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies validated")
    return True

def test_basic_functionality():
    """Test basic robot and environment functionality."""
    print("🤖 Testing basic functionality...")
    
    try:
        # Test robot creation
        from src.robot.leg_robot import LeggedRobot
        from src.simulation.environment import Environment
        
        env = Environment(render=False)
        robot = LeggedRobot(client=env.client)
        env.add_robot(robot)
        
        # Test basic simulation steps
        for i in range(10):
            env.step()
        
        env.close()
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def run_minimal_ablation():
    """Run a minimal ablation study for validation."""
    print("🧪 Running minimal ablation study...")
    
    try:
        # Import the ablation study
        sys.path.append('experiments/studies')
        from ablation_study import AblationStudyManager
        
        # Create manager with minimal settings
        study_manager = AblationStudyManager()
        study_manager.num_independent_runs = 2  # Very minimal
        study_manager.base_config["max_iterations"] = 50  # Quick test
        
        # Run only baseline and one comparison
        minimal_configs = {
            "C0_baseline": study_manager.ablation_configs["C0_baseline"],
            "C11_no_structural": study_manager.ablation_configs["C11_no_structural"]
        }
        study_manager.ablation_configs = minimal_configs
        
        print("Running baseline vs no-structural comparison...")
        results = study_manager.run_full_study(parallel=False)
        
        print(f"✅ Minimal ablation completed: {len(results)} experiments")
        return True
        
    except Exception as e:
        print(f"❌ Minimal ablation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_evolution_test():
    """Run a quick evolution test."""
    print("🧬 Running quick evolution test...")
    
    try:
        from experiments.core.run_evolution import main
        
        # Simulate command line arguments for quick test
        sys.argv = ['run_evolution.py', '--quick-test', '--no-render', '--quiet']
        
        main()
        print("✅ Quick evolution test completed")
        return True
        
    except Exception as e:
        print(f"❌ Quick evolution test failed: {e}")
        return False

def generate_validation_report():
    """Generate a validation report."""
    print("📊 Generating validation report...")
    
    report_content = f"""
# Experimental Framework Validation Report

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Validation Results

### Environment Setup: ✅ PASSED
- Directory structure created
- Python packages configured

### Dependencies: ✅ PASSED  
- All required packages available
- Import tests successful

### Basic Functionality: ✅ PASSED
- Robot creation and control
- Environment simulation
- PyBullet integration

### Evolution Framework: ✅ PASSED
- Enhanced VEGA algorithm
- Fitness evaluation
- Data logging

### Ablation Framework: ✅ PASSED
- Configuration management
- Statistical protocols
- Result compilation

## Next Steps

1. **Run Full Ablation Study**: 
   ```bash
   python experiments/studies/ablation_study.py --full-study
   ```

2. **Parameter Sensitivity Analysis**:
   ```bash  
   python experiments/studies/sensitivity_analysis.py --lhs-samples=50
   ```

3. **Generate Publication Figures**:
   ```bash
   python experiments/visualization/publication_plots.py --all-figures
   ```

## Estimated Timeline

- **Full Ablation Study**: 48-72 hours (with parallelization)
- **Sensitivity Analysis**: 24-36 hours  
- **Statistical Validation**: 2-4 hours
- **Publication Materials**: 4-8 hours

**Total Estimated Runtime**: 5-7 days
"""
    
    with open("results/validation/validation_report.md", "w") as f:
        f.write(report_content)
    
    print("✅ Validation report saved to results/validation/validation_report.md")

def main():
    """Main validation workflow."""
    print("🚀 Starting Experimental Framework Validation")
    print("=" * 60)
    
    # Run validation steps
    setup_environment()
    
    if not validate_dependencies():
        return False
    
    if not test_basic_functionality():
        return False
    
    if not run_quick_evolution_test():
        return False
    
    if not run_minimal_ablation():
        return False
    
    generate_validation_report()
    
    print("=" * 60)
    print("🎉 Validation completed successfully!")
    print("\n📋 Summary:")
    print("- Framework is ready for full experiments")
    print("- All core components validated")
    print("- Minimal ablation study completed")
    print("\n🚀 Ready for production runs:")
    print("1. Full ablation study: python experiments/studies/ablation_study.py --full-study")
    print("2. Sensitivity analysis: python experiments/studies/sensitivity_analysis.py")
    print("3. Publication plots: python experiments/visualization/publication_plots.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)