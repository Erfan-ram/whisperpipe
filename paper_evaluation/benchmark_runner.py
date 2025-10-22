#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Benchmark Runner for Academic Paper Evaluation
Executes group_benchmark.py with comprehensive results logging and structured output
"""

import os
import sys
import json
import csv
import time
import subprocess
import platform
import psutil
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from group_benchmark import run_benchmark, get_test_files

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class BenchmarkRunner:
    """Enhanced benchmark runner with comprehensive results logging"""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize benchmark runner with configuration"""
        self.config = self._load_config(config_path)
        self.results_dir = Path(self.config['output']['base_dir'])
        self.run_id = datetime.now().strftime(self.config['output']['naming']['timestamp_format'])
        self.run_dir = self.results_dir / f"{self.config['output']['naming']['run_prefix']}_{self.run_id}"
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in self.config['output']['subdirs'].values():
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if file not found"""
        return {
            'benchmark': {
                'audio': {'data_dir': 'test_audio', 'file_limit': 4, 'max_chunk_duration_seconds': 30},
                'model': {'name': 'base', 'language': 'en'},
                'runs': {'count': 3, 'parallel': False}
            },
            'output': {
                'base_dir': 'results',
                'naming': {'timestamp_format': '%Y%m%d_%H%M%S', 'run_prefix': 'run'}
            }
        }
    
    def _capture_system_info(self) -> Dict:
        """Capture system information for reproducibility"""
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version()
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'software': {
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        # Capture GPU info if available
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total_mb': torch.cuda.get_device_properties(i).total_memory / (1024**2),
                    'compute_capability': torch.cuda.get_device_properties(i).major
                })
            system_info['gpu'] = gpu_info
        
        return system_info
    
    def _capture_git_info(self) -> Dict:
        """Capture git information for reproducibility"""
        git_info = {
            'available': False,
            'commit_hash': None,
            'branch': None,
            'dirty': False
        }
        
        try:
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=project_root)
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
                git_info['available'] = True
            
            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, cwd=project_root)
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Check if working directory is dirty
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=project_root)
            if result.returncode == 0:
                git_info['dirty'] = len(result.stdout.strip()) > 0
                
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return git_info
    
    def _run_single_benchmark(self, run_number: int) -> Dict:
        """Run a single benchmark execution"""
        print(f"\n{'='*60}")
        print(f"Running Benchmark {run_number + 1}/{self.config['benchmark']['runs']['count']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Get test files
            test_files = get_test_files(
                self.config['benchmark']['audio']['data_dir'],
                limit=self.config['benchmark']['audio']['file_limit']
            )
            
            if not test_files:
                raise ValueError("No test files found")
            
            # Run benchmark with modified group_benchmark
            results = self._run_modified_benchmark(test_files)
            
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            results['run_number'] = run_number
            results['timestamp'] = datetime.now().isoformat()
            
            print(f"Benchmark {run_number + 1} completed in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            print(f"Error in benchmark {run_number + 1}: {e}")
            return {
                'error': str(e),
                'run_number': run_number,
                'timestamp': datetime.now().isoformat(),
                'execution_time': time.time() - start_time
            }
    
    def _run_modified_benchmark(self, test_files: List[Dict]) -> Dict:
        """Run the benchmark with enhanced result capture"""
        
        # Run the benchmark from group_benchmark.py
        results = run_benchmark(
            test_files,
            max_chunk_duration_seconds=self.config['benchmark']['audio']['max_chunk_duration_seconds']
        )
        
        return results
    
    def run_benchmarks(self) -> Dict:
        """Run multiple benchmark executions for statistical significance"""
        print(f"Starting benchmark evaluation with {self.config['benchmark']['runs']['count']} runs")
        print(f"Results will be saved to: {self.run_dir}")
        
        # Capture system and git info
        system_info = self._capture_system_info()
        git_info = self._capture_git_info()
        
        # Save configuration and system info
        self._save_config()
        self._save_system_info(system_info, git_info)
        
        # Run benchmarks
        all_results = []
        
        for i in range(self.config['benchmark']['runs']['count']):
            result = self._run_single_benchmark(i)
            all_results.append(result)
            
            # Save individual run results
            self._save_run_results(result, i)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)
        
        # Save aggregated results
        self._save_aggregated_results(aggregated_results)
        
        # Generate summary
        self._generate_summary(aggregated_results)
        
        print(f"\nBenchmark evaluation completed!")
        print(f"Results saved to: {self.run_dir}")
        
        return aggregated_results
    
    def _save_config(self):
        """Save configuration to results directory"""
        config_path = self.run_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, cls=NumpyEncoder)
    
    def _save_system_info(self, system_info: Dict, git_info: Dict):
        """Save system and git information"""
        system_path = self.run_dir / 'system_info.json'
        with open(system_path, 'w') as f:
            json.dump({
                'system': system_info,
                'git': git_info
            }, f, indent=2, cls=NumpyEncoder)
    
    def _save_run_results(self, results: Dict, run_number: int):
        """Save individual run results"""
        run_path = self.run_dir / f'run_{run_number:02d}_results.json'
        with open(run_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results from multiple runs"""
        # Filter out failed runs
        successful_runs = [r for r in all_results if 'error' not in r]
        
        if not successful_runs:
            return {'error': 'No successful runs', 'total_runs': len(all_results)}
        
        aggregated = {
            'total_runs': len(all_results),
            'successful_runs': len(successful_runs),
            'failed_runs': len(all_results) - len(successful_runs),
            'runs': all_results
        }
        
        # Add statistical aggregation here
        # This would include mean, std, confidence intervals, etc.
        
        return aggregated
    
    def _save_aggregated_results(self, results: Dict):
        """Save aggregated results"""
        aggregated_path = self.run_dir / 'aggregated_results.json'
        with open(aggregated_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    def _generate_summary(self, results: Dict):
        """Generate summary report"""
        summary_path = self.run_dir / 'summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("Benchmark Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total runs: {results['total_runs']}\n")
            f.write(f"Successful runs: {results['successful_runs']}\n")
            f.write(f"Failed runs: {results['failed_runs']}\n")
            f.write(f"Success rate: {results['successful_runs']/results['total_runs']*100:.1f}%\n\n")
            
            if results['successful_runs'] > 0:
                f.write("Results will be processed by statistical analysis module.\n")
                f.write("Plots and tables will be generated by plot_generator.py\n")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run academic paper benchmark evaluation')
    parser.add_argument('--config', default='configs/default.yaml', 
                       help='Configuration file path')
    parser.add_argument('--runs', type=int, 
                       help='Number of benchmark runs (overrides config)')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BenchmarkRunner(args.config)
    
    # Override runs if specified
    if args.runs:
        runner.config['benchmark']['runs']['count'] = args.runs
    
    # Run benchmarks
    results = runner.run_benchmarks()
    
    print(f"\nBenchmark evaluation completed!")
    print(f"Next steps:")
    print(f"1. Run statistical analysis: python paper_evaluation/statistical_analysis.py")
    print(f"2. Generate plots: python paper_evaluation/plot_generator.py")
    print(f"3. Create LaTeX tables: python paper_evaluation/latex_generator.py")


if __name__ == "__main__":
    main()

