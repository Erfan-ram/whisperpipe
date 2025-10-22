#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical Analysis Module for Academic Paper Evaluation
Implements rigorous statistical testing for benchmark results
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, wilcoxon, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for benchmark results"""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize statistical analyzer"""
        self.results_dir = Path(results_dir)
        self.latest_run_dir = self._find_latest_run()
        
        if not self.latest_run_dir:
            raise ValueError(f"No benchmark results found in {results_dir}")
        
        print(f"Analyzing results from: {self.latest_run_dir}")
    
    def _find_latest_run(self) -> Optional[Path]:
        """Find the most recent benchmark run directory"""
        if not self.results_dir.exists():
            return None
        
        run_dirs = [d for d in self.results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        if not run_dirs:
            return None
        
        # Sort by modification time and get the latest
        latest = max(run_dirs, key=lambda x: x.stat().st_mtime)
        return latest
    
    def _load_results(self) -> Dict:
        """Load benchmark results from JSON files"""
        results = {}
        
        # Load aggregated results
        aggregated_path = self.latest_run_dir / 'aggregated_results.json'
        if aggregated_path.exists():
            with open(aggregated_path, 'r') as f:
                results['aggregated'] = json.load(f)
        
        # Load individual run results
        run_results = []
        for run_file in self.latest_run_dir.glob('run_*_results.json'):
            with open(run_file, 'r') as f:
                run_data = json.load(f)
                run_results.append(run_data)
        
        results['runs'] = run_results
        return results
    
    def _extract_metrics(self, results: Dict) -> Dict:
        """Extract metrics from benchmark results"""
        metrics = {
            'whisperpipe': {
                'wer': [],
                'stability_index': [],
                'avg_latency_ms': [],
                'peak_gpu_memory_mb': [],
                'peak_ram_mb': [],
                'mean_gpu_util_pct': [],
                'mean_cpu_util_pct': [],
                'processing_time': []
            },
            'baseline': {
                'wer': [],
                'stability_index': [],
                'avg_latency_ms': [],
                'peak_gpu_memory_mb': [],
                'peak_ram_mb': [],
                'mean_gpu_util_pct': [],
                'mean_cpu_util_pct': [],
                'processing_time': []
            }
        }
        
        # Extract metrics from each run
        for run in results['runs']:
            if 'error' in run:
                continue
            
            # Extract whisperpipe metrics
            if 'whisperpipe' in run and 'metrics' in run['whisperpipe']:
                wp_metrics = run['whisperpipe']['metrics']
                for key in metrics['whisperpipe']:
                    if key in wp_metrics:
                        metrics['whisperpipe'][key].append(wp_metrics[key])
            
            # Extract baseline metrics
            if 'baseline' in run and 'metrics' in run['baseline']:
                bl_metrics = run['baseline']['metrics']
                for key in metrics['baseline']:
                    if key in bl_metrics:
                        metrics['baseline'][key].append(bl_metrics[key])
        
        return metrics
    
    def _calculate_descriptive_stats(self, data: List[float]) -> Dict:
        """Calculate descriptive statistics"""
        if not data:
            return {}
        
        data_array = np.array(data)
        return {
            'mean': float(np.mean(data_array)),
            'std': float(np.std(data_array, ddof=1)),
            'median': float(np.median(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'q25': float(np.percentile(data_array, 25)),
            'q75': float(np.percentile(data_array, 75)),
            'count': len(data_array)
        }
    
    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data"""
        if len(data) < 2:
            return (float(data[0]) if data else 0, float(data[0]) if data else 0)
        
        data_array = np.array(data)
        n = len(data_array)
        mean = np.mean(data_array)
        std_err = stats.sem(data_array)
        
        # Calculate t-statistic for confidence interval
        alpha = 1 - confidence
        t_val = stats.t.ppf(1 - alpha/2, n - 1)
        
        margin_error = t_val * std_err
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        return (float(ci_lower), float(ci_upper))
    
    def _calculate_effect_size(self, group1: List[float], group2: List[float]) -> Dict:
        """Calculate Cohen's d effect size"""
        if len(group1) < 2 or len(group2) < 2:
            return {'cohens_d': 0, 'magnitude': 'no effect'}
        
        # Calculate pooled standard deviation
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        # Calculate Cohen's d
        mean_diff = np.mean(group1) - np.mean(group2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            magnitude = 'negligible'
        elif abs_d < 0.5:
            magnitude = 'small'
        elif abs_d < 0.8:
            magnitude = 'medium'
        else:
            magnitude = 'large'
        
        return {
            'cohens_d': float(cohens_d),
            'magnitude': magnitude,
            'interpretation': f"{magnitude} effect size"
        }
    
    def _perform_t_tests(self, metrics: Dict) -> Dict:
        """Perform t-tests for all metrics"""
        t_tests = {}
        
        for metric_name in metrics['whisperpipe']:
            wp_data = metrics['whisperpipe'][metric_name]
            bl_data = metrics['baseline'][metric_name]
            
            if len(wp_data) < 2 or len(bl_data) < 2:
                continue
            
            # Paired t-test (if same number of samples)
            if len(wp_data) == len(bl_data):
                try:
                    t_stat, p_value = ttest_rel(wp_data, bl_data)
                    test_type = 'paired'
                except:
                    t_stat, p_value = ttest_ind(wp_data, bl_data)
                    test_type = 'independent'
            else:
                t_stat, p_value = ttest_ind(wp_data, bl_data)
                test_type = 'independent'
            
            # Calculate effect size
            effect_size = self._calculate_effect_size(wp_data, bl_data)
            
            # Determine significance
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'ns'
            
            t_tests[metric_name] = {
                'test_type': test_type,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significance': significance,
                'effect_size': effect_size,
                'sample_sizes': {
                    'whisperpipe': len(wp_data),
                    'baseline': len(bl_data)
                }
            }
        
        return t_tests
    
    def _perform_wilcoxon_tests(self, metrics: Dict) -> Dict:
        """Perform Wilcoxon signed-rank tests for non-parametric analysis"""
        wilcoxon_tests = {}
        
        for metric_name in metrics['whisperpipe']:
            wp_data = metrics['whisperpipe'][metric_name]
            bl_data = metrics['baseline'][metric_name]
            
            if len(wp_data) < 2 or len(bl_data) < 2:
                continue
            
            # Use paired test if same length, otherwise independent
            if len(wp_data) == len(bl_data):
                try:
                    statistic, p_value = wilcoxon(wp_data, bl_data)
                    test_type = 'paired'
                except:
                    statistic, p_value = mannwhitneyu(wp_data, bl_data, alternative='two-sided')
                    test_type = 'independent'
            else:
                statistic, p_value = mannwhitneyu(wp_data, bl_data, alternative='two-sided')
                test_type = 'independent'
            
            # Determine significance
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'ns'
            
            wilcoxon_tests[metric_name] = {
                'test_type': test_type,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significance': significance,
                'sample_sizes': {
                    'whisperpipe': len(wp_data),
                    'baseline': len(bl_data)
                }
            }
        
        return wilcoxon_tests
    
    def _calculate_improvements(self, metrics: Dict) -> Dict:
        """Calculate improvement percentages"""
        improvements = {}
        
        for metric_name in metrics['whisperpipe']:
            wp_data = metrics['whisperpipe'][metric_name]
            bl_data = metrics['baseline'][metric_name]
            
            if not wp_data or not bl_data:
                continue
            
            wp_mean = np.mean(wp_data)
            bl_mean = np.mean(bl_data)
            
            # Calculate relative improvement
            if bl_mean != 0:
                relative_improvement = ((bl_mean - wp_mean) / bl_mean) * 100
            else:
                relative_improvement = 0
            
            # Calculate absolute difference
            absolute_difference = wp_mean - bl_mean
            
            improvements[metric_name] = {
                'whisperpipe_mean': float(wp_mean),
                'baseline_mean': float(bl_mean),
                'absolute_difference': float(absolute_difference),
                'relative_improvement_pct': float(relative_improvement),
                'improvement_direction': 'better' if relative_improvement > 0 else 'worse'
            }
        
        return improvements
    
    def analyze(self) -> Dict:
        """Perform comprehensive statistical analysis"""
        print("Loading benchmark results...")
        results = self._load_results()
        
        print("Extracting metrics...")
        metrics = self._extract_metrics(results)
        
        print("Calculating descriptive statistics...")
        descriptive_stats = {}
        for system in ['whisperpipe', 'baseline']:
            descriptive_stats[system] = {}
            for metric_name, data in metrics[system].items():
                descriptive_stats[system][metric_name] = self._calculate_descriptive_stats(data)
        
        print("Calculating confidence intervals...")
        confidence_intervals = {}
        for system in ['whisperpipe', 'baseline']:
            confidence_intervals[system] = {}
            for metric_name, data in metrics[system].items():
                if data:
                    ci_lower, ci_upper = self._calculate_confidence_interval(data)
                    confidence_intervals[system][metric_name] = {
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'confidence_level': 0.95
                    }
        
        print("Performing t-tests...")
        t_tests = self._perform_t_tests(metrics)
        
        print("Performing Wilcoxon tests...")
        wilcoxon_tests = self._perform_wilcoxon_tests(metrics)
        
        print("Calculating improvements...")
        improvements = self._calculate_improvements(metrics)
        
        # Compile comprehensive analysis
        analysis = {
            'metadata': {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'results_directory': str(self.latest_run_dir),
                'total_runs': len(results['runs']),
                'successful_runs': len([r for r in results['runs'] if 'error' not in r])
            },
            'descriptive_statistics': descriptive_stats,
            'confidence_intervals': confidence_intervals,
            't_tests': t_tests,
            'wilcoxon_tests': wilcoxon_tests,
            'improvements': improvements,
            'raw_metrics': metrics
        }
        
        # Save analysis results
        self._save_analysis(analysis)
        
        # Generate summary report
        self._generate_summary_report(analysis)
        
        print(f"\nStatistical analysis completed!")
        print(f"Results saved to: {self.latest_run_dir}")
        
        return analysis
    
    def _save_analysis(self, analysis: Dict):
        """Save statistical analysis results"""
        analysis_path = self.latest_run_dir / 'statistical_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def _generate_summary_report(self, analysis: Dict):
        """Generate summary report of statistical analysis"""
        report_path = self.latest_run_dir / 'statistical_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("Statistical Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Key findings
            f.write("KEY FINDINGS:\n")
            f.write("-" * 20 + "\n")
            
            for metric, improvement in analysis['improvements'].items():
                if improvement['relative_improvement_pct'] > 0:
                    f.write(f"✓ {metric.upper()}: {improvement['relative_improvement_pct']:.1f}% improvement\n")
                else:
                    f.write(f"✗ {metric.upper()}: {abs(improvement['relative_improvement_pct']):.1f}% worse\n")
            
            f.write(f"\nStatistical Tests:\n")
            f.write("-" * 20 + "\n")
            
            for metric, test in analysis['t_tests'].items():
                f.write(f"{metric}: p={test['p_value']:.4f} {test['significance']}\n")
            
            f.write(f"\nEffect Sizes (Cohen's d):\n")
            f.write("-" * 20 + "\n")
            
            for metric, test in analysis['t_tests'].items():
                effect = test['effect_size']
                f.write(f"{metric}: d={effect['cohens_d']:.3f} ({effect['magnitude']})\n")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform statistical analysis on benchmark results')
    parser.add_argument('--results-dir', default='results', 
                       help='Results directory path')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(args.results_dir)
    
    # Perform analysis
    analysis = analyzer.analyze()
    
    print(f"\nStatistical analysis completed!")
    print(f"Next steps:")
    print(f"1. Generate plots: python paper_evaluation/plot_generator.py")
    print(f"2. Create LaTeX tables: python paper_evaluation/latex_generator.py")


if __name__ == "__main__":
    main()

