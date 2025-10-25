#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Report Generator for Academic Paper Evaluation
Creates comprehensive HTML/PDF reports with embedded plots and analysis
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import report generation dependencies
try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("Warning: jinja2 not available. HTML templates will use basic formatting.")

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("Warning: weasyprint not available. PDF generation will be disabled.")


class ReportGenerator:
    """Generate comprehensive HTML/PDF reports"""
    
    def __init__(self, results_dir: str = "results", run_dir: Optional[str] = None):
        """Initialize report generator"""
        if run_dir:
            self.latest_run_dir = Path(run_dir)
        else:
            self.results_dir = Path(results_dir)
            self.latest_run_dir = self._find_latest_run()

        if not self.latest_run_dir or not self.latest_run_dir.exists():
            raise ValueError(f"No benchmark results found in {run_dir or results_dir}")
        
        # Create output directory
        self.reports_dir = self.latest_run_dir / 'reports'
        self.reports_dir.mkdir(exist_ok=True)
        
        print(f"Generating reports for results from: {self.latest_run_dir}")
    

    
    def _load_analysis_data(self) -> Dict:
        """Load statistical analysis data"""
        analysis_path = self.latest_run_dir / 'statistical_analysis.json'
        if not analysis_path.exists():
            raise FileNotFoundError("Statistical analysis not found. Run statistical_analysis.py first.")
        
        with open(analysis_path, 'r') as f:
            return json.load(f)
    
    def _load_system_info(self) -> Dict:
        """Load system information"""
        system_path = self.latest_run_dir / 'system_info.json'
        if system_path.exists():
            with open(system_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        config_path = self.latest_run_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _generate_executive_summary(self, analysis: Dict) -> str:
        """Generate executive summary with key findings"""
        summary = []
        summary.append("# Executive Summary")
        summary.append("")
        summary.append("This report presents a comprehensive evaluation of the enhanced whisperpipe streaming system compared to baseline Whisper streaming.")
        summary.append("")
        
        # Key findings
        summary.append("## Key Findings")
        summary.append("")
        
        improvements = analysis.get('improvements', {})
        significant_improvements = []
        
        for metric, improvement in improvements.items():
            try:
                improvement_val = float(improvement['relative_improvement_pct'])
            except (ValueError, TypeError):
                improvement_val = np.nan # Use NaN for non-numeric values

            if not np.isnan(improvement_val) and improvement_val > 0:
                significant_improvements.append({
                    'metric': metric.replace('_', ' ').title(),
                    'improvement': improvement_val
                })
        
        if significant_improvements:
            summary.append("The enhanced whisperpipe system demonstrates significant improvements across multiple metrics:")
            summary.append("")
            for imp in significant_improvements:
                summary.append(f"- **{imp['metric']}**: {imp['improvement']:+.1f}% improvement")
        
        # Statistical significance
        t_tests = analysis.get('t_tests', {})
        significant_tests = [k for k, v in t_tests.items() if v['p_value'] < 0.05]
        
        if significant_tests:
            summary.append(f"Statistical significance (p < 0.05) was achieved for {len(significant_tests)} metrics:")
            for metric in significant_tests:
                p_val = t_tests[metric]['p_value']
                summary.append(f"- {metric.replace('_', ' ').title()}: p = {p_val:.4f}")
            summary.append("")
        
        # Effect sizes
        large_effects = []
        for metric, test in t_tests.items():
            if 'effect_size' in test:
                effect_size = test['effect_size']['cohens_d']
                if abs(effect_size) > 0.8:
                    large_effects.append({
                        'metric': metric.replace('_', ' ').title(),
                        'effect_size': effect_size,
                        'magnitude': test['effect_size']['magnitude']
                    })
        
        if large_effects:
            summary.append("Large effect sizes (Cohen's d > 0.8) were observed for:")
            for effect in large_effects:
                summary.append(f"- {effect['metric']}: d = {effect['effect_size']:.3f} ({effect['magnitude']})")
            summary.append("")
        
        return "\n".join(summary)
    
    def _generate_methodology_section(self) -> str:
        """Generate methodology section"""
        methodology = []
        methodology.append("# Methodology")
        methodology.append("")
        methodology.append("## Experimental Setup")
        methodology.append("")
        methodology.append("The evaluation was conducted using a comprehensive benchmark framework that compares:")
        methodology.append("")
        methodology.append("1. **Enhanced whisperpipe system** with dual-buffer architecture, similarity-based stabilization, and noise rejection")
        methodology.append("2. **Baseline Whisper streaming** using a simple sliding window approach")
        methodology.append("")
        methodology.append("## Evaluation Metrics")
        methodology.append("")
        methodology.append("The following metrics were used to assess system performance:")
        methodology.append("")
        methodology.append("- **Word Error Rate (WER)**: Transcription accuracy")
        methodology.append("- **Stability Index (SI)**: Output consistency (novel metric)")
        methodology.append("- **Latency**: End-to-end processing time")
        methodology.append("- **Resource Usage**: GPU memory, RAM, CPU utilization")
        methodology.append("")
        methodology.append("## Statistical Analysis")
        methodology.append("")
        methodology.append("Statistical significance was assessed using:")
        methodology.append("")
        methodology.append("- **Paired t-tests** for normally distributed data")
        methodology.append("- **Wilcoxon signed-rank tests** for non-parametric data")
        methodology.append("- **95% confidence intervals** for all metrics")
        methodology.append("- **Cohen's d** for effect size calculation")
        methodology.append("")
        
        return "\n".join(methodology)
    
    def _generate_results_section(self, analysis: Dict) -> str:
        """Generate results section with key findings"""
        results = []
        results.append("# Results")
        results.append("")
        
        # Performance comparison table
        results.append("## Performance Comparison")
        results.append("")
        results.append("| Metric | whisperpipe | Baseline | Improvement | Significance |")
        results.append("|--------|-------------|----------|-------------|--------------|")
        
        metrics = ['wer', 'stability_index', 'avg_latency_ms']
        metric_labels = ['WER (%)', 'Stability Index (%)', 'Latency (ms)']
        
        for metric, label in zip(metrics, metric_labels):
            if metric in analysis['descriptive_statistics']['whisperpipe']:
                wp_stats = analysis['descriptive_statistics']['whisperpipe'][metric]
                bl_stats = analysis['descriptive_statistics']['baseline'][metric]
                
                wp_mean = f"{wp_stats['mean']:.2f} ± {wp_stats['std']:.2f}"
                bl_mean = f"{bl_stats['mean']:.2f} ± {bl_stats['std']:.2f}"
                
                if metric in analysis['improvements']:
                    improvement_val = analysis['improvements'][metric]['relative_improvement_pct']
                    try:
                        improvement_str = f"{float(improvement_val):+.1f}%"
                    except (ValueError, TypeError):
                        improvement_str = str(improvement_val)
                else:
                    improvement_str = "---"
                
                significance = ""
                if metric in analysis['t_tests']:
                    p_val = analysis['t_tests'][metric]['p_value']
                    if p_val < 0.001:
                        significance = "***"
                    elif p_val < 0.01:
                        significance = "**"
                    elif p_val < 0.05:
                        significance = "*"
                    else:
                        significance = "ns"
                
                results.append(f"| {label} | {wp_mean} | {bl_mean} | {improvement_str} | {significance} |")
        
        results.append("")
        results.append("*Note: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant*")
        results.append("")
        
        # Key findings
        results.append("## Key Findings")
        results.append("")
        
        improvements = analysis.get('improvements', {})
        for metric, improvement in improvements.items():
            improvement_val = improvement['relative_improvement_pct']
            try:
                numeric_improvement_val = float(improvement_val)
            except (ValueError, TypeError):
                numeric_improvement_val = np.nan # Use NaN for non-numeric values

            if not np.isnan(numeric_improvement_val) and numeric_improvement_val > 0:
                results.append(f"- **{metric.replace('_', ' ').title()}**: {numeric_improvement_val:+.1f}% improvement")
            elif not np.isnan(numeric_improvement_val):
                results.append(f"- **{metric.replace('_', ' ').title()}**: {numeric_improvement_val:+.1f}% (worse)")
        
        results.append("")
        
        return "\n".join(results)
    
    def _generate_plots_section(self) -> str:
        """Generate plots section with embedded images"""
        plots = []
        plots.append("# Visualizations")
        plots.append("")
        plots.append("The following plots provide detailed visual analysis of the benchmark results:")
        plots.append("")
        
        # List of generated plots
        plot_descriptions = [
            ("Main Performance Comparison", "plot_1_main_performance_comparison", "Bar chart comparing WER, SI, and Latency"),
            ("Resource Usage Comparison", "plot_2_resource_usage_comparison", "Multi-panel comparison of GPU, RAM, and CPU usage"),
            ("Memory Usage Over Time", "plot_3_memory_usage_time_series", "Time series showing memory usage patterns"),
            ("Processing Latency Per Chunk", "plot_4_latency_time_series", "Latency evolution across audio chunks"),
            ("Stability Index Distribution", "plot_5_stability_index_distribution", "Distribution comparison of stability metrics"),
            ("WER vs Audio Duration", "plot_6_wer_vs_duration_scatter", "Scatter plot showing WER performance over time"),
            ("Computational Efficiency", "plot_7_computational_efficiency_radar", "Radar chart comparing efficiency metrics"),
            ("Error Analysis", "plot_8_error_analysis_heatmap", "Heatmap of error types by system"),
            ("Memory Growth Rate", "plot_9_memory_growth_rate", "Analysis of memory growth patterns"),
            ("Latency Distribution", "plot_10_latency_distribution_histogram", "Histogram comparison of latency distributions")
        ]
        
        plots_dir = self.latest_run_dir / 'plots'
        
        for i, (title, filename, description) in enumerate(plot_descriptions, 1):
            plots.append(f"## {i}. {title}")
            plots.append("")
            plots.append(description)
            plots.append("")
            
            # Check if plot files exist
            plot_paths = list(plots_dir.glob(f"{filename}.*"))
            if plot_paths:
                # Use the first available format
                plot_path = plot_paths[0]
                plots.append(f"![{title}]({plot_path.relative_to(self.latest_run_dir)})")
            else:
                plots.append(f"*Plot not available: {filename}*")
            
            plots.append("")
        
        return "\n".join(plots)
    
    def _generate_statistical_analysis_section(self, analysis: Dict) -> str:
        """Generate detailed statistical analysis section"""
        stats = []
        stats.append("# Statistical Analysis")
        stats.append("")
        
        # T-tests summary
        stats.append("## T-Test Results")
        stats.append("")
        stats.append("| Metric | t-statistic | p-value | Significance | Effect Size (d) |")
        stats.append("|--------|-------------|---------|--------------|-----------------|")
        
        t_tests = analysis.get('t_tests', {})
        for metric, test in t_tests.items():
            metric_label = metric.replace('_', ' ').title()
            t_stat = f"{test['t_statistic']:.3f}"
            p_val = f"{test['p_value']:.4f}"
            significance = test['significance']
            
            effect_size = ""
            if 'effect_size' in test:
                effect_size = f"{test['effect_size']['cohens_d']:.3f}"
            
            stats.append(f"| {metric_label} | {t_stat} | {p_val} | {significance} | {effect_size} |")
        
        stats.append("")
        stats.append("*Note: Effect size interpretation: negligible (< 0.2), small (0.2-0.5), medium (0.5-0.8), large (> 0.8)*")
        stats.append("")
        
        # Confidence intervals
        stats.append("## Confidence Intervals (95%)")
        stats.append("")
        stats.append("| Metric | whisperpipe CI | Baseline CI | Overlap |")
        stats.append("|--------|----------------|-------------|---------|")
        
        ci_data = analysis.get('confidence_intervals', {})
        for system in ['whisperpipe', 'baseline']:
            if system in ci_data:
                for metric, ci in ci_data[system].items():
                    metric_label = metric.replace('_', ' ').title()
                    ci_str = f"[{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]"
                    
                    # Check for overlap with other system
                    other_system = 'baseline' if system == 'whisperpipe' else 'whisperpipe'
                    if other_system in ci_data and metric in ci_data[other_system]:
                        other_ci = ci_data[other_system][metric]
                        overlap = not (ci['ci_upper'] < other_ci['ci_lower'] or other_ci['ci_upper'] < ci['ci_lower'])
                        overlap_str = "Yes" if overlap else "No"
                    else:
                        overlap_str = "N/A"
                    
                    stats.append(f"| {metric_label} | {ci_str} | --- | {overlap_str} |")
        
        stats.append("")
        
        return "\n".join(stats)
    
    def _generate_system_specifications(self, system_info: Dict) -> str:
        """Generate system specifications section"""
        specs = []
        specs.append("# System Specifications")
        specs.append("")
        
        if 'system' in system_info:
            sys_info = system_info['system']
            specs.append("## Hardware")
            specs.append("")
            specs.append(f"- **CPU**: {sys_info.get('processor', 'Unknown')}")
            specs.append(f"- **CPU Cores**: {sys_info.get('cpu_count', 'Unknown')}")
            mem_gb = sys_info.get('memory_total_gb')
            if isinstance(mem_gb, (int, float)):
                specs.append(f"- **Memory**: {mem_gb:.1f} GB")
            else:
                specs.append(f"- **Memory**: {mem_gb or 'Unknown'}")
            specs.append(f"- **Operating System**: {sys_info.get('system', 'Unknown')} {sys_info.get('release', '')}")
            specs.append("")
        
        if 'software' in system_info:
            sw_info = system_info['software']
            specs.append("## Software")
            specs.append("")
            py_ver = system_info.get('platform', {}).get('python_version', 'Unknown')
            specs.append(f"- **Python**: {py_ver}")
            specs.append(f"- **PyTorch**: {sw_info.get('torch_version', 'Unknown')}")
            specs.append(f"- **CUDA Available**: {sw_info.get('cuda_available', 'Unknown')}")
            if sw_info.get('cuda_available'):
                specs.append(f"- **CUDA Version**: {sw_info.get('cuda_version', 'Unknown')}")
            specs.append("")
        
        if 'gpu' in system_info.get('system', {}):
            specs.append("## GPU Information")
            specs.append("")
            for i, gpu in enumerate(system_info['system']['gpu']):
                specs.append(f"**GPU {i}:**")
                specs.append(f"- Name: {gpu.get('name', 'Unknown')}")
                specs.append(f"- Memory: {gpu.get('memory_total_mb', 0):.0f} MB")
                specs.append(f"- Compute Capability: {gpu.get('compute_capability', 'Unknown')}")
                specs.append("")
        
        return "\n".join(specs)
    
    def _generate_reproducibility_section(self, system_info: Dict) -> str:
        """Generate reproducibility section"""
        repro = []
        repro.append("# Reproducibility")
        repro.append("")
        repro.append("## Git Information")
        repro.append("")
        
        if 'git' in system_info:
            git_info = system_info['git']
            if git_info.get('available'):
                repro.append(f"- **Commit Hash**: {git_info.get('commit_hash', 'Unknown')}")
                repro.append(f"- **Branch**: {git_info.get('branch', 'Unknown')}")
                repro.append(f"- **Working Directory Clean**: {'Yes' if not git_info.get('dirty') else 'No'}")
            else:
                repro.append("- Git information not available")
        else:
            repro.append("- Git information not available")
        
        repro.append("")
        repro.append("## Configuration")
        repro.append("")
        repro.append("The benchmark was run with the following configuration:")
        repro.append("")
        
        config = self._load_config()
        if config:
            repro.append("```yaml")
            repro.append(json.dumps(config, indent=2))
            repro.append("```")
        else:
            repro.append("- Configuration not available")
        
        repro.append("")
        repro.append("## Reproducibility Checklist")
        repro.append("")
        repro.append("- [ ] All dependencies installed from requirements.txt")
        repro.append("- [ ] Audio data available in test_audio/ directory")
        repro.append("- [ ] GPU drivers and CUDA properly configured")
        repro.append("- [ ] Python environment matches system specifications")
        repro.append("- [ ] Git repository in clean state")
        repro.append("")
        
        return "\n".join(repro)
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report"""
        print("Loading analysis data...")
        analysis = self._load_analysis_data()
        system_info = self._load_system_info()
        
        print("Generating report sections...")
        
        # Generate all sections
        sections = [
            self._generate_executive_summary(analysis),
            self._generate_methodology_section(),
            self._generate_results_section(analysis),
            self._generate_plots_section(),
            self._generate_statistical_analysis_section(analysis),
            self._generate_system_specifications(system_info),
            self._generate_reproducibility_section(system_info)
        ]
        
        # Combine sections
        full_report = "\n\n".join(sections)
        
        # Add header and footer
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Paper Evaluation Report</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 10px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Academic Paper Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Results Directory:</strong> {self.latest_run_dir}</p>
        
        {full_report}
        
        <hr>
        <footer>
            <p><em>Report generated by Academic Paper Visualization System</em></p>
        </footer>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        html_path = self.reports_dir / 'evaluation_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved: {html_path}")
        return str(html_path)
    
    def generate_pdf_report(self) -> Optional[str]:
        """Generate PDF report from HTML"""
        if not WEASYPRINT_AVAILABLE:
            print("Warning: weasyprint not available. PDF generation skipped.")
            return None
        
        try:
            html_path = self.reports_dir / 'evaluation_report.html'
            pdf_path = self.reports_dir / 'evaluation_report.pdf'
            
            # Generate PDF from HTML
            weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            
            print(f"✓ PDF report saved: {pdf_path}")
            return str(pdf_path)
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return None
    
    def generate_markdown_report(self) -> str:
        """Generate Markdown report"""
        print("Loading analysis data...")
        analysis = self._load_analysis_data()
        system_info = self._load_system_info()
        
        print("Generating Markdown report...")
        
        # Generate all sections
        sections = [
            f"# Academic Paper Evaluation Report\n\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n**Results Directory:** {self.latest_run_dir}\n",
            self._generate_executive_summary(analysis),
            self._generate_methodology_section(),
            self._generate_results_section(analysis),
            self._generate_plots_section(),
            self._generate_statistical_analysis_section(analysis),
            self._generate_system_specifications(system_info),
            self._generate_reproducibility_section(system_info)
        ]
        
        # Combine sections
        markdown_content = "\n\n".join(sections)
        
        # Save Markdown report
        md_path = self.reports_dir / 'evaluation_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"✓ Markdown report saved: {md_path}")
        return str(md_path)
    
    def generate_all_reports(self) -> List[str]:
        """Generate all report formats"""
        print("Generating comprehensive reports...")
        
        generated_files = []
        
        # Generate HTML report
        try:
            html_file = self.generate_html_report()
            generated_files.append(html_file)
        except Exception as e:
            print(f"Error generating HTML report: {e}")
        
        # Generate PDF report
        try:
            pdf_file = self.generate_pdf_report()
            if pdf_file:
                generated_files.append(pdf_file)
        except Exception as e:
            print(f"Error generating PDF report: {e}")
        
        # Generate Markdown report
        try:
            md_file = self.generate_markdown_report()
            generated_files.append(md_file)
        except Exception as e:
            print(f"Error generating Markdown report: {e}")
        
        # Generate report index
        self._generate_report_index(generated_files)
        
        print(f"\nReport generation completed!")
        print(f"Reports saved to: {self.reports_dir}")
        
        return generated_files
    
    def _generate_report_index(self, report_files: List[str]):
        """Generate index of all reports"""
        index_path = self.reports_dir / 'report_index.txt'
        
        with open(index_path, 'w') as f:
            f.write("Generated Reports Index\n")
            f.write("=" * 50 + "\n\n")
            
            for i, report_file in enumerate(report_files, 1):
                f.write(f"Report {i}: {Path(report_file).name}\n")
                f.write(f"  File: {report_file}\n\n")
            
            f.write("\nUsage Instructions:\n")
            f.write("-" * 20 + "\n")
            f.write("1. HTML Report: Open in web browser for interactive viewing\n")
            f.write("2. PDF Report: Print-ready format for paper submission\n")
            f.write("3. Markdown Report: Source format for documentation\n")
            f.write("4. All reports contain the same content in different formats\n")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive reports')
    parser.add_argument('--run-dir', type=str, help='Path to the specific run directory to analyze')
    parser.add_argument('--results-dir', default='results', 
                       help='Base results directory (used if --run-dir is not provided)')
    parser.add_argument('--format', choices=['html', 'pdf', 'markdown', 'all'], 
                       default='all', help='Report format to generate')
    
    args = parser.parse_args()
    
    # Initialize report generator
    generator = ReportGenerator(results_dir=args.results_dir, run_dir=args.run_dir)
    
    # Generate reports based on format
    if args.format == 'all':
        reports = generator.generate_all_reports()
    elif args.format == 'html':
        reports = [generator.generate_html_report()]
    elif args.format == 'pdf':
        pdf_file = generator.generate_pdf_report()
        reports = [pdf_file] if pdf_file else []
    elif args.format == 'markdown':
        reports = [generator.generate_markdown_report()]
    
    print(f"\nReport generation completed!")
    print(f"Generated {len(reports)} reports")
    print(f"Next steps:")
    print(f"1. Review reports in: {generator.reports_dir}")
    print(f"2. Use HTML/PDF reports for paper submission")
    print(f"3. Use Markdown report for documentation")


if __name__ == "__main__":
    main()

