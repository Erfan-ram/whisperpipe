#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Execution Script for Academic Paper Evaluation
Single-command entry point for the complete evaluation pipeline
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import yaml
import json
import shutil


class PaperEvaluationRunner:
    """Main runner for academic paper evaluation pipeline"""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize the evaluation runner"""
        self.config_path = config_path
        self.config = self._load_config()
        self.start_time = datetime.now()
        
        # Define results directory
        self.results_dir = Path(self.config['output']['base_dir'])
        self.run_dir = self.results_dir / "latest_run"
        
        print(f"🚀 Starting Academic Paper Evaluation")
        print(f"📁 Using results directory: {self.run_dir}")
        print(f"⏰ Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Warning: Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            print(f"⚠️  Warning: Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'benchmark': {
                'audio': {'data_dir': 'test_audio', 'file_limit': 20, 'max_chunk_duration_seconds': 43},
                'model': {'name': 'base', 'language': 'en'},
                'runs': {'count': 2, 'parallel': False}
            },
            'output': {
                'base_dir': 'results',
                'naming': {'timestamp_format': '%Y%m%d_%H%M%S', 'run_prefix': 'run'}
            }
        }
    
    def _run_command(self, command: str, description: str) -> bool:
        """Run a command and return success status"""
        print(f"\n🔄 {description}")
        print(f"   Command: {command}")
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                print(f"   ✅ Success")
                if result.stdout:
                    print(f"   Output: {result.stdout.strip()}")
                return True
            else:
                print(f"   ❌ Failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        print("\n🔍 Checking dependencies...")
        
        required_packages = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 
            'torch', 'whisper', 'soundfile', 'librosa'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} (missing)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("   Install with: pip install -r requirements.txt")
            return False
        
        print("   ✅ All dependencies available")
        return True
    
    def _check_data_availability(self) -> bool:
        """Check if required data is available"""
        print("\n📊 Checking data availability...")
        
        # Check audio data directory
        audio_dir = Path(self.config['benchmark']['audio']['data_dir'])
        if not audio_dir.exists():
            print(f"   ❌ Audio directory not found: {audio_dir}")
            print(f"   Please ensure audio files are available in {audio_dir}")
            return False
        
        # Check for audio files
        audio_files = list(audio_dir.glob("*.flac")) + list(audio_dir.glob("*.wav"))
        if not audio_files:
            print(f"   ❌ No audio files found in {audio_dir}")
            print(f"   Please add audio files (.flac or .wav) to {audio_dir}")
            return False
        
        print(f"   ✅ Found {len(audio_files)} audio files")
        return True
    
    def run_benchmark_evaluation(self) -> bool:
        """Run the benchmark evaluation"""
        print("\n" + "="*60)
        print("📊 STEP 1: BENCHMARK EVALUATION")
        print("="*60)
        
        # Run benchmark runner
        command = f"/home/erfan/venvs/torchzone/bin/python paper_evaluation/benchmark_runner.py --config {self.config_path} --output-dir {self.run_dir}"
        if self.config['benchmark']['runs']['count'] > 1:
            command += f" --runs {self.config['benchmark']['runs']['count']}"
        
        success = self._run_command(command, "Running benchmark evaluation")
        
        if not success:
            print("❌ Benchmark evaluation failed")
            return False
        
        print("✅ Benchmark evaluation completed")
        return True
    
    def run_statistical_analysis(self) -> bool:
        """Run statistical analysis"""
        print("\n" + "="*60)
        print("📈 STEP 2: STATISTICAL ANALYSIS")
        print("="*60)
        
        command = f"/home/erfan/venvs/torchzone/bin/python paper_evaluation/statistical_analysis.py --run-dir {self.run_dir}"
        success = self._run_command(command, "Running statistical analysis")
        
        if not success:
            print("❌ Statistical analysis failed")
            return False
        
        print("✅ Statistical analysis completed")
        return True
    
    def generate_plots(self) -> bool:
        """Generate publication-ready plots"""
        print("\n" + "="*60)
        print("📊 STEP 3: PLOT GENERATION")
        print("="*60)
        
        command = f"/home/erfan/venvs/torchzone/bin/python paper_evaluation/plot_generator.py --run-dir {self.run_dir} --config {self.config_path}"
        success = self._run_command(command, "Generating publication-ready plots")
        
        if not success:
            print("❌ Plot generation failed")
            return False
        
        print("✅ Plot generation completed")
        return True
    
    def generate_latex_tables(self) -> bool:
        """Generate LaTeX tables"""
        print("\n" + "="*60)
        print("📋 STEP 4: LATEX TABLE GENERATION")
        print("="*60)
        
        command = f"/home/erfan/venvs/torchzone/bin/python paper_evaluation/latex_generator.py --run-dir {self.run_dir}"
        success = self._run_command(command, "Generating LaTeX tables")
        
        if not success:
            print("❌ LaTeX table generation failed")
            return False
        
        print("✅ LaTeX table generation completed")
        return True
    
    def generate_reports(self) -> bool:
        """Generate comprehensive reports"""
        print("\n" + "="*60)
        print("📄 STEP 5: REPORT GENERATION")
        print("="*60)
        
        command = f"/home/erfan/venvs/torchzone/bin/python paper_evaluation/report_generator.py --run-dir {self.run_dir} --format all"
        success = self._run_command(command, "Generating comprehensive reports")
        
        if not success:
            print("❌ Report generation failed")
            return False
        
        print("✅ Report generation completed")
        return True
    
    def create_jupyter_notebook(self) -> bool:
        """Create interactive Jupyter notebook"""
        print("\n" + "="*60)
        print("📓 STEP 6: JUPYTER NOTEBOOK CREATION")
        print("="*60)
        
        # Copy the interactive notebook to results directory
        notebook_source = Path("paper_evaluation/interactive_analysis.ipynb")
        notebook_dest = self.run_dir / "interactive_analysis.ipynb"
        
        if notebook_source.exists():
            import shutil
            shutil.copy2(notebook_source, notebook_dest)
            print(f"✅ Interactive notebook created: {notebook_dest}")
            return True
        else:
            print("❌ Interactive notebook not found")
            return False
    
    def generate_summary(self) -> None:
        """Generate final summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print("\n" + "="*60)
        print("🎉 EVALUATION COMPLETED")
        print("="*60)
        
        print(f"⏰ Total duration: {duration}")
        print(f"📁 Results directory: {self.run_dir}")
        
        # List generated files
        print("\n📋 Generated files:")
        
        subdirs = ['plots', 'tables', 'reports', 'data']
        for subdir in subdirs:
            subdir_path = self.run_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*"))
                if files:
                    print(f"   📂 {subdir}/ ({len(files)} files)")
                    for file in files[:3]:  # Show first 3 files
                        print(f"      - {file.name}")
                    if len(files) > 3:
                        print(f"      ... and {len(files) - 3} more")
        
        # Show key files
        key_files = [
            "aggregated_results.json",
            "statistical_analysis.json", 
            "evaluation_report.html",
            "evaluation_report.pdf",
            "interactive_analysis.ipynb"
        ]
        
        print(f"\n🔑 Key files:")
        for key_file in key_files:
            file_path = self.run_dir / key_file
            if file_path.exists():
                print(f"   ✅ {key_file}")
            else:
                print(f"   ❌ {key_file} (not found)")
        
        print(f"\n📖 Next steps:")
        print(f"   1. Review results in: {self.run_dir}")
        print(f"   2. Open HTML report: {self.run_dir}/reports/evaluation_report.html")
        print(f"   3. Use LaTeX tables in: {self.run_dir}/tables/")
        print(f"   4. Open interactive notebook: {self.run_dir}/interactive_analysis.ipynb")
        print(f"   5. Copy plots from: {self.run_dir}/plots/")
    
    def run_full_evaluation(self, mode: str) -> bool:
        """Run the complete evaluation pipeline based on the selected mode."""
        print(f"🚀 Starting Academic Paper Evaluation Pipeline in '{mode}' mode")
        print("=" * 60)

        # Clean directory for modes that generate new test data
        if mode in ['test', 'default']:
            print(f"🧹 Cleaning results directory: {self.run_dir}")
            if self.run_dir.exists():
                shutil.rmtree(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)

        # Check dependencies and data
        if not self._check_dependencies():
            print("❌ Dependency check failed. Please install missing packages.")
            return False
        
        if mode in ['test', 'default'] and not self._check_data_availability():
            print("❌ Data availability check failed. Please ensure audio files are available.")
            return False
        
        # Define pipeline steps for each mode
        all_steps = {
            "benchmark": ("Benchmark Evaluation", self.run_benchmark_evaluation),
            "stats": ("Statistical Analysis", self.run_statistical_analysis),
            "plots": ("Plot Generation", self.generate_plots),
            "tables": ("LaTeX Table Generation", self.generate_latex_tables),
            "reports": ("Report Generation", self.generate_reports),
            "notebook": ("Jupyter Notebook Creation", self.create_jupyter_notebook)
        }

        steps_to_run = []
        if mode == 'test':
            steps_to_run = [all_steps["benchmark"]]
        elif mode == 'evaluate':
            steps_to_run = [all_steps["stats"], all_steps["plots"], all_steps["tables"], all_steps["reports"], all_steps["notebook"]]
        else:  # default mode
            steps_to_run = list(all_steps.values())

        failed_steps = []
        for step_name, step_function in steps_to_run:
            try:
                if not step_function():
                    failed_steps.append(step_name)
                    # Stop pipeline if a step fails
                    break
            except Exception as e:
                print(f"❌ Exception in {step_name}: {e}")
                failed_steps.append(step_name)
                break
        
        # Generate summary
        self.generate_summary()
        
        if failed_steps:
            print(f"\n⚠️  Mode '{mode}' failed at step: {', '.join(failed_steps)}")
            print("   Please check the logs above for details.")
            return False
        else:
            print(f"\n🎉 Mode '{mode}' completed successfully!")
            return True


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run complete academic paper evaluation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_paper_evaluation.py                    # Run default mode (test + evaluate)
  python run_paper_evaluation.py --mode test        # Run only benchmark tests
  python run_paper_evaluation.py --mode evaluate    # Run only evaluation on existing results
  python run_paper_evaluation.py --config my_config.yaml  # Use custom config
  python run_paper_evaluation.py --quick           # Quick run (1 iteration)
  python run_paper_evaluation.py --runs 5          # Run 5 iterations for statistics
        """
    )
    
    parser.add_argument('--config', default='configs/default.yaml',
                       help='Configuration file path (default: configs/default.yaml)')
    parser.add_argument('--mode', default='default', choices=['default', 'test', 'evaluate'],
                       help='Execution mode (default: default)')
    parser.add_argument('--runs', type=int,
                       help='Number of benchmark runs (overrides config)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with minimal iterations')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = PaperEvaluationRunner(args.config)
    
    # Override config based on arguments
    if args.quick:
        runner.config['benchmark']['runs']['count'] = 1
        print("🏃 Quick mode: Running 1 iteration only")
    elif args.runs:
        runner.config['benchmark']['runs']['count'] = args.runs
        print(f"🔄 Custom runs: {args.runs} iterations")
    
    # Run evaluation
    success = runner.run_full_evaluation(mode=args.mode)
    
    if success:
        print("\n🎉 Academic paper evaluation completed successfully!")
        print("   Your results are ready for paper submission.")
        sys.exit(0)
    else:
        print("\n❌ Evaluation completed with errors.")
        print("   Please check the logs above and fix any issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()

