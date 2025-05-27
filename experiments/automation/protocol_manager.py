#!/usr/bin/env python3
"""
Automated Experimental Protocol System
Orchestrates comprehensive experiments for structural mutation operator analysis.
"""

import os
import sys
import json
import yaml
import subprocess
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time
import psutil
import signal
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    config_id: str
    active_operators: List[str]
    operator_probabilities: Dict[str, float]
    num_runs: int = 30
    max_iterations: int = 500
    population_size: int = 30
    chromosome_length: int = 8
    timeout_minutes: int = 120
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class ExperimentBatch:
    """Batch of related experiments."""
    batch_id: str
    name: str
    description: str
    experiments: List[ExperimentConfig]
    estimated_duration_hours: float
    resource_requirements: Dict[str, any]

class AutomatedExperimentOrchestrator:
    """
    Orchestrates large-scale automated experiments for operator analysis.
    Handles resource management, scheduling, monitoring, and recovery.
    """
    
    def __init__(self, config_path="config/experiment_protocol.yaml"):
        self.config = self._load_configuration(config_path)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(f"experiments/automated_{self.experiment_id}")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        self.resource_manager = ResourceManager(self.config.get("resources", {}))
        
        # Experiment tracking
        self.experiment_queue = []
        self.running_experiments = {}
        self.completed_experiments = []
        self.failed_experiments = []
        
        # Setup logging
        self._setup_logging()
        
        # Load experimental protocols
        self.ablation_protocol = self._create_ablation_protocol()
        self.sensitivity_protocol = self._create_sensitivity_protocol()
        
        self.logger.info(f"Automated Experiment Orchestrator initialized: {self.experiment_id}")
    
    def _load_configuration(self, config_path):
        """Load system configuration."""
        default_config = {
            "resources": {
                "max_concurrent_experiments": 4,
                "cpu_cores_per_experiment": 2,
                "memory_gb_per_experiment": 4,
                "gpu_required": False
            },
            "scheduling": {
                "priority_scheduling": True,
                "adaptive_resource_allocation": True,
                "experiment_timeout_hours": 6
            },
            "monitoring": {
                "check_interval_seconds": 30,
                "progress_reporting_interval": 300,
                "system_health_monitoring": True
            },
            "recovery": {
                "auto_restart_failed": True,
                "max_restart_attempts": 3,
                "checkpoint_interval_minutes": 60
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        self.logger = logging.getLogger('experiment_orchestrator')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.base_output_dir / "orchestrator.log"
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
    
    def _create_ablation_protocol(self):
        """Create comprehensive ablation study protocol."""
        ablation_experiments = []
        
        # Define all ablation configurations as per Table 3
        ablation_configs = {
            "C0_baseline": {
                "name": "Baseline (All Operators)",
                "active_operators": ["insertion", "deletion", "phase_exchange", "order_exchange"],
                "probabilities": {"insertion": 0.15, "deletion": 0.15, "phase_exchange": 0.10, "order_exchange": 0.10},
                "priority": 1
            },
            "C1_no_insertion": {
                "name": "No Insertion",
                "active_operators": ["deletion", "phase_exchange", "order_exchange"],
                "probabilities": {"deletion": 0.15, "phase_exchange": 0.10, "order_exchange": 0.10},
                "priority": 1
            },
            "C2_no_deletion": {
                "name": "No Deletion", 
                "active_operators": ["insertion", "phase_exchange", "order_exchange"],
                "probabilities": {"insertion": 0.15, "phase_exchange": 0.10, "order_exchange": 0.10},
                "priority": 1
            },
            "C3_no_phase": {
                "name": "No Phase Exchange",
                "active_operators": ["insertion", "deletion", "order_exchange"],
                "probabilities": {"insertion": 0.15, "deletion": 0.15, "order_exchange": 0.10},
                "priority": 1
            },
            "C4_no_order": {
                "name": "No Order Exchange",
                "active_operators": ["insertion", "deletion", "phase_exchange"],
                "probabilities": {"insertion": 0.15, "deletion": 0.15, "phase_exchange": 0.10},
                "priority": 1
            },
            "C5_exploration_only": {
                "name": "Exploration Only (I+D)",
                "active_operators": ["insertion", "deletion"],
                "probabilities": {"insertion": 0.15, "deletion": 0.15},
                "priority": 2
            },
            "C6_refinement_only": {
                "name": "Refinement Only (P+O)",
                "active_operators": ["phase_exchange", "order_exchange"],
                "probabilities": {"phase_exchange": 0.10, "order_exchange": 0.10},
                "priority": 2
            },
            "C11_no_structural": {
                "name": "No Structural Mutations",
                "active_operators": [],
                "probabilities": {},
                "priority": 1
            }
        }
        
        # Create experiment configs
        for config_id, config_data in ablation_configs.items():
            experiment = ExperimentConfig(
                name=config_data["name"],
                config_id=config_id,
                active_operators=config_data["active_operators"],
                operator_probabilities=config_data["probabilities"],
                num_runs=30,  # Statistical requirement
                max_iterations=500,
                priority=config_data["priority"]
            )
            ablation_experiments.append(experiment)
        
        # Create batch
        ablation_batch = ExperimentBatch(
            batch_id="ablation_study",
            name="Comprehensive Ablation Study",
            description="Systematic evaluation of structural mutation operators",
            experiments=ablation_experiments,
            estimated_duration_hours=12.0,
            resource_requirements={"cpu_hours": 240, "memory_gb_hours": 960}
        )
        
        return ablation_batch
    
    def _create_sensitivity_protocol(self):
        """Create parameter sensitivity analysis protocol."""
        # Latin Hypercube Sampling parameters
        parameter_ranges = {
            "insertion_prob": [0.05, 0.25],
            "deletion_prob": [0.05, 0.25],
            "phase_exchange_prob": [0.05, 0.25], 
            "order_exchange_prob": [0.05, 0.25],
            "max_sequence_length": [4, 6, 8],
            "penalty_coefficient": [5.0, 10.0, 20.0]
        }
        
        # Generate LHS samples (reduced for practical execution)
        num_samples = 50  # Reduced from full protocol for efficiency
        sensitivity_experiments = []
        
        # Generate samples with constraints
        for i in range(num_samples):
            # Simple random sampling for example (replace with proper LHS)
            sample_probs = {}
            for param, (min_val, max_val) in parameter_ranges.items():
                if "prob" in param:
                    sample_probs[param] = np.random.uniform(min_val, max_val)
                elif param == "max_sequence_length":
                    sample_probs[param] = int(np.random.choice([4, 6, 8]))
                else:
                    sample_probs[param] = np.random.uniform(min_val, max_val)
            
            # Check probability constraint
            total_prob = sum(v for k, v in sample_probs.items() if "prob" in k)
            if total_prob > 0.5:
                # Normalize probabilities
                scale_factor = 0.5 / total_prob
                for k in sample_probs:
                    if "prob" in k:
                        sample_probs[k] *= scale_factor
            
            experiment = ExperimentConfig(
                name=f"Sensitivity Sample {i+1}",
                config_id=f"S{i+1:03d}",
                active_operators=["insertion", "deletion", "phase_exchange", "order_exchange"],
                operator_probabilities={k: v for k, v in sample_probs.items() if "prob" in k},
                num_runs=5,  # Fewer runs for sensitivity analysis
                max_iterations=200,  # Shorter for parameter exploration
                chromosome_length=sample_probs["max_sequence_length"],
                priority=3  # Lower priority
            )
            sensitivity_experiments.append(experiment)
        
        sensitivity_batch = ExperimentBatch(
            batch_id="sensitivity_analysis",
            name="Parameter Sensitivity Analysis", 
            description="Latin Hypercube Sampling of parameter space",
            experiments=sensitivity_experiments,
            estimated_duration_hours=8.0,
            resource_requirements={"cpu_hours": 160, "memory_gb_hours": 640}
        )
        
        return sensitivity_batch
    
    def schedule_experiment_batch(self, batch: ExperimentBatch, priority=1):
        """Schedule a batch of experiments for execution."""
        self.logger.info(f"Scheduling batch: {batch.name} ({len(batch.experiments)} experiments)")
        
        # Add experiments to queue with priority
        for experiment in batch.experiments:
            experiment_task = {
                "batch_id": batch.batch_id,
                "experiment": experiment,
                "priority": priority,
                "scheduled_time": datetime.now(),
                "status": "queued"
            }
            self.experiment_queue.append(experiment_task)
        
        # Sort queue by priority
        self.experiment_queue.sort(key=lambda x: (x["priority"], x["scheduled_time"]))
        
        self.logger.info(f"Batch scheduled. Queue length: {len(self.experiment_queue)}")
    
    def run_orchestrated_experiments(self):
        """Main orchestration loop for running experiments."""
        self.logger.info("Starting orchestrated experiment execution")
        
        # Schedule both protocols
        self.schedule_experiment_batch(self.ablation_protocol, priority=1)
        self.schedule_experiment_batch(self.sensitivity_protocol, priority=2)
        
        try:
            while self.experiment_queue or self.running_experiments:
                # System health check
                if not self.system_monitor.check_system_health():
                    self.logger.warning("System health issues detected. Pausing new experiments.")
                    time.sleep(60)
                    continue
                
                # Resource management
                available_slots = self.resource_manager.get_available_slots()
                
                # Start new experiments if resources available
                while available_slots > 0 and self.experiment_queue:
                    next_task = self.experiment_queue.pop(0)
                    
                    if self.resource_manager.can_allocate_resources(next_task["experiment"]):
                        self._start_experiment(next_task)
                        available_slots -= 1
                    else:
                        # Put back in queue if can't allocate resources
                        self.experiment_queue.insert(0, next_task)
                        break
                
                # Monitor running experiments
                self._monitor_running_experiments()
                
                # Progress reporting
                self._report_progress()
                
                # Sleep before next cycle
                time.sleep(self.config["monitoring"]["check_interval_seconds"])
                
        except KeyboardInterrupt:
            self.logger.info("Orchestration interrupted by user")
            self._graceful_shutdown()
        except Exception as e:
            self.logger.error(f"Orchestration error: {e}")
            self._emergency_shutdown()
        
        # Final reporting
        self._generate_final_report()
    
    def _start_experiment(self, task):
        """Start a single experiment."""
        experiment = task["experiment"]
        experiment_dir = self.base_output_dir / task["batch_id"] / experiment.config_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment script
        script_path = self._create_experiment_script(experiment, experiment_dir)
        
        # Start process
        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=experiment_dir
            )
            
            task.update({
                "status": "running",
                "process": process,
                "start_time": datetime.now(),
                "output_dir": experiment_dir,
                "progress": 0
            })
            
            self.running_experiments[experiment.config_id] = task
            self.resource_manager.allocate_resources(experiment)
            
            self.logger.info(f"Started experiment: {experiment.name} (PID: {process.pid})")
            
        except Exception as e:
            self.logger.error(f"Failed to start experiment {experiment.name}: {e}")
            task["status"] = "failed"
            self.failed_experiments.append(task)
    
    def _create_experiment_script(self, experiment: ExperimentConfig, output_dir: Path):
        """Create a standalone script for running an experiment."""
        script_content = f'''#!/usr/bin/env python3
"""
Automated experiment script for {experiment.name}
Generated by Automated Experiment Orchestrator
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath("{os.getcwd()}"))

from experiments.run_evolution import run_evolution_experiment

def main():
    # Experiment configuration
    config = {{
        "name": "{experiment.name}",
        "config_id": "{experiment.config_id}",
        "active_operators": {experiment.active_operators},
        "operator_probabilities": {experiment.operator_probabilities},
        "num_runs": {experiment.num_runs},
        "max_iterations": {experiment.max_iterations},
        "population_size": {experiment.population_size},
        "chromosome_length": {experiment.chromosome_length},
        "output_dir": "{output_dir}",
        "headless": True
    }}
    
    print(f"Starting experiment: {{config['name']}}")
    print(f"Configuration: {{json.dumps(config, indent=2)}}")
    
    # Run the experiment
    try:
        results = run_evolution_experiment(config)
        
        # Save results
        with open("experiment_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Experiment completed successfully")
        return 0
        
    except Exception as e:
        print(f"Experiment failed: {{e}}")
        with open("experiment_error.log", "w") as f:
            import traceback
            f.write(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        script_path = output_dir / "run_experiment.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def _monitor_running_experiments(self):
        """Monitor and manage running experiments."""
        completed_experiments = []
        
        for exp_id, task in self.running_experiments.items():
            process = task["process"]
            experiment = task["experiment"]
            
            # Check if process is still running
            if process.poll() is not None:
                # Process completed
                return_code = process.returncode
                
                if return_code == 0:
                    task["status"] = "completed"
                    task["end_time"] = datetime.now()
                    self.completed_experiments.append(task)
                    self.logger.info(f"Experiment completed: {experiment.name}")
                else:
                    task["status"] = "failed"
                    task["end_time"] = datetime.now()
                    self.failed_experiments.append(task)
                    self.logger.error(f"Experiment failed: {experiment.name} (exit code: {return_code})")
                
                # Release resources
                self.resource_manager.release_resources(experiment)
                completed_experiments.append(exp_id)
            
            else:
                # Check timeout
                runtime = datetime.now() - task["start_time"]
                timeout = timedelta(minutes=experiment.timeout_minutes)
                
                if runtime > timeout:
                    self.logger.warning(f"Experiment timeout: {experiment.name}")
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                    
                    task["status"] = "timeout"
                    task["end_time"] = datetime.now()
                    self.failed_experiments.append(task)
                    self.resource_manager.release_resources(experiment)
                    completed_experiments.append(exp_id)
                
                else:
                    # Update progress if possible
                    progress = self._estimate_progress(task)
                    task["progress"] = progress
        
        # Remove completed experiments from running list
        for exp_id in completed_experiments:
            del self.running_experiments[exp_id]
    
    def _estimate_progress(self, task):
        """Estimate experiment progress."""
        runtime = datetime.now() - task["start_time"]
        experiment = task["experiment"]
        
        # Simple time-based estimate (can be improved with actual progress monitoring)
        estimated_duration = timedelta(minutes=experiment.timeout_minutes * 0.8)
        progress = min(100, (runtime.total_seconds() / estimated_duration.total_seconds()) * 100)
        
        return progress
    
    def _report_progress(self):
        """Generate progress report."""
        total_experiments = len(self.completed_experiments) + len(self.failed_experiments) + len(self.running_experiments) + len(self.experiment_queue)
        completed = len(self.completed_experiments)
        failed = len(self.failed_experiments)
        running = len(self.running_experiments)
        queued = len(self.experiment_queue)
        
        # Log progress every 5 minutes
        if not hasattr(self, '_last_progress_report'):
            self._last_progress_report = datetime.now()
        
        if (datetime.now() - self._last_progress_report).total_seconds() >= 300:
            self.logger.info(f"Progress: {completed}/{total_experiments} completed, "
                           f"{running} running, {queued} queued, {failed} failed")
            self._last_progress_report = datetime.now()
    
    def _graceful_shutdown(self):
        """Gracefully shutdown running experiments."""
        self.logger.info("Initiating graceful shutdown...")
        
        for exp_id, task in self.running_experiments.items():
            process = task["process"]
            self.logger.info(f"Stopping experiment: {task['experiment'].name}")
            
            # Send SIGTERM
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                process.kill()
                process.wait()
            
            self.resource_manager.release_resources(task["experiment"])
    
    def _emergency_shutdown(self):
        """Emergency shutdown of all experiments."""
        self.logger.error("Emergency shutdown initiated")
        
        for exp_id, task in self.running_experiments.items():
            process = task["process"]
            try:
                process.kill()
                self.resource_manager.release_resources(task["experiment"])
            except:
                pass
    
    def _generate_final_report(self):
        """Generate final orchestration report."""
        report = {
            "orchestration_id": self.experiment_id,
            "start_time": self._last_progress_report if hasattr(self, '_last_progress_report') else datetime.now(),
            "end_time": datetime.now(),
            "summary": {
                "total_experiments": len(self.completed_experiments) + len(self.failed_experiments),
                "completed": len(self.completed_experiments),
                "failed": len(self.failed_experiments),
                "success_rate": len(self.completed_experiments) / max(1, len(self.completed_experiments) + len(self.failed_experiments))
            },
            "completed_experiments": [
                {
                    "name": task["experiment"].name,
                    "config_id": task["experiment"].config_id,
                    "duration_minutes": (task["end_time"] - task["start_time"]).total_seconds() / 60,
                    "output_dir": str(task["output_dir"])
                }
                for task in self.completed_experiments
            ],
            "failed_experiments": [
                {
                    "name": task["experiment"].name,
                    "config_id": task["experiment"].config_id,
                    "failure_reason": task["status"]
                }
                for task in self.failed_experiments
            ]
        }
        
        report_path = self.base_output_dir / "orchestration_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Final report saved: {report_path}")
        return report


class SystemMonitor:
    """Monitor system health and resource usage."""
    
    def check_system_health(self):
        """Check if system is healthy for running experiments."""
        # CPU usage check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            return False
        
        # Memory usage check
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            return False
        
        # Disk space check
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            return False
        
        return True


class ResourceManager:
    """Manage computational resources for experiments."""
    
    def __init__(self, resource_config):
        self.config = resource_config
        self.allocated_resources = {
            "cpu_cores": 0,
            "memory_gb": 0,
            "experiments": 0
        }
        
        # System limits
        self.limits = {
            "cpu_cores": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "experiments": resource_config.get("max_concurrent_experiments", 4)
        }
    
    def can_allocate_resources(self, experiment: ExperimentConfig):
        """Check if resources can be allocated for an experiment."""
        required_cpu = self.config.get("cpu_cores_per_experiment", 2)
        required_memory = self.config.get("memory_gb_per_experiment", 4)
        
        return (
            self.allocated_resources["experiments"] < self.limits["experiments"] and
            self.allocated_resources["cpu_cores"] + required_cpu <= self.limits["cpu_cores"] * 0.8 and
            self.allocated_resources["memory_gb"] + required_memory <= self.limits["memory_gb"] * 0.8
        )
    
    def allocate_resources(self, experiment: ExperimentConfig):
        """Allocate resources for an experiment."""
        self.allocated_resources["cpu_cores"] += self.config.get("cpu_cores_per_experiment", 2)
        self.allocated_resources["memory_gb"] += self.config.get("memory_gb_per_experiment", 4)
        self.allocated_resources["experiments"] += 1
    
    def release_resources(self, experiment: ExperimentConfig):
        """Release resources from a completed experiment."""
        self.allocated_resources["cpu_cores"] -= self.config.get("cpu_cores_per_experiment", 2)
        self.allocated_resources["memory_gb"] -= self.config.get("memory_gb_per_experiment", 4)
        self.allocated_resources["experiments"] -= 1
        
        # Ensure non-negative values
        for key in self.allocated_resources:
            self.allocated_resources[key] = max(0, self.allocated_resources[key])
    
    def get_available_slots(self):
        """Get number of available experiment slots."""
        return max(0, self.limits["experiments"] - self.allocated_resources["experiments"])


if __name__ == "__main__":
    # Example usage
    orchestrator = AutomatedExperimentOrchestrator()
    
    # Run automated experiments
    # orchestrator.run_orchestrated_experiments()
    
    print("Automated Experiment Orchestrator initialized successfully")
    print(f"Ablation protocol: {len(orchestrator.ablation_protocol.experiments)} experiments")
    print(f"Sensitivity protocol: {len(orchestrator.sensitivity_protocol.experiments)} experiments")