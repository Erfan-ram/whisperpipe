# Academic Paper Visualization System - Usage Guide

## Overview

This system provides comprehensive evaluation and visualization tools for academic paper submission. It automatically runs benchmarks, performs statistical analysis, generates publication-ready plots, creates LaTeX tables, and produces detailed reports.

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, whisper, matplotlib, seaborn; print('All dependencies installed!')"
```

### 2. Prepare Your Data

```bash
# Ensure audio files are in test_audio/ directory
ls test_audio/
# Should show: *.flac or *.wav files

# Example structure:
# test_audio/
# ├── audio1.flac
# ├── audio2.flac
# └── audio1.trans.txt  # Optional: reference transcriptions
```

### 3. Run Complete Evaluation

```bash
# Single command to run everything
python run_paper_evaluation.py

# Quick run (1 iteration, faster)
python run_paper_evaluation.py --quick

# Custom number of runs for statistics
python run_paper_evaluation.py --runs 5
```

### 4. View Results

```bash
# Open HTML report in browser
firefox results/run_YYYYMMDD_HHMMSS/reports/evaluation_report.html

# Or open interactive notebook
jupyter notebook results/run_YYYYMMDD_HHMMSS/interactive_analysis.ipynb
```

## 📊 What Gets Generated

The system creates a comprehensive evaluation with:

### 📈 Plots (10+ publication-ready figures)
- Main performance comparison (bar chart)
- Resource usage comparison (multi-panel)
- Memory usage over time (time series)
- Processing latency per chunk (time series)
- Stability index distribution (box/violin plots)
- WER vs audio duration (scatter plot)
- Computational efficiency (radar chart)
- Error analysis (heatmap)
- Memory growth rate analysis (line plot)
- Latency distribution (histogram)

### 📋 LaTeX Tables (5 ready-to-paste tables)
- Main results comparison
- Resource usage metrics
- Statistical significance tests
- Ablation study (if applicable)
- Detailed confidence intervals

### 📄 Reports (3 formats)
- **HTML Report**: Interactive web report with embedded plots
- **PDF Report**: Print-ready format for paper submission
- **Markdown Report**: Source format for documentation

### 📓 Interactive Notebook
- Real-time data exploration
- Customizable plotting
- Statistical analysis
- Export functionality

## 🔧 Configuration

### Basic Configuration

Edit `configs/default.yaml` to customize:

```yaml
benchmark:
  audio:
    data_dir: "test_audio"           # Audio files directory
    file_limit: 4                    # Max files to process
    max_chunk_duration_seconds: 30   # Max chunk duration
  
  model:
    name: "base"                     # Whisper model (tiny, base, small, medium, large)
    language: "en"                   # Language code
  
  runs:
    count: 3                         # Number of benchmark runs
    parallel: false                  # Run in parallel (requires more memory)

plots:
  style: "ieee"                      # ieee, nature, acl
  dpi: 300                          # Resolution for PNG output
  format: ["png", "pdf", "eps"]     # Output formats
```

### Advanced Configuration

```yaml
# Statistical analysis settings
statistics:
  confidence_level: 0.95
  tests:
    t_test: true
    wilcoxon: true
    effect_size: true

# Plot customization
plots:
  colors:
    primary: ["#1f77b4", "#ff7f0e"]  # IEEE blue/orange
    accent: ["#2ca02c", "#d62728"]  # Green, red accents
  
  sizes:
    single_column: [3.5, 2.5]        # IEEE single column
    double_column: [7.0, 4.0]      # IEEE double column
```

## 🎯 Usage Examples

### Example 1: Quick Evaluation

```bash
# Fast evaluation for testing
python run_paper_evaluation.py --quick

# Results: 1 benchmark run, all plots and tables generated
```

### Example 2: Statistical Significance

```bash
# Run 10 iterations for robust statistics
python run_paper_evaluation.py --runs 10

# Results: Statistical significance tests, confidence intervals
```

### Example 3: Custom Configuration

```bash
# Use custom config file
python run_paper_evaluation.py --config my_config.yaml

# Results: Evaluation with your specific settings
```

### Example 4: Individual Components

```bash
# Run only benchmark evaluation
python paper_evaluation/benchmark_runner.py

# Run only statistical analysis
python paper_evaluation/statistical_analysis.py

# Generate only plots
python paper_evaluation/plot_generator.py

# Create only LaTeX tables
python paper_evaluation/latex_generator.py

# Generate only reports
python paper_evaluation/report_generator.py
```

## 📊 Understanding the Results

### Key Metrics Explained

1. **WER (Word Error Rate)**: Lower is better
   - Measures transcription accuracy
   - Formula: (Substitutions + Deletions + Insertions) / Total Words × 100%

2. **Stability Index (SI)**: Higher is better
   - Novel metric for output consistency
   - Formula: (1 - avg_edit_distance / avg_length) × 100%
   - 100% = perfectly stable, 0% = completely unstable

3. **Latency**: Lower is better
   - End-to-end processing time in milliseconds

4. **Resource Usage**: Lower is better
   - GPU memory, RAM, CPU utilization
   - Measured in MB and percentage

### Statistical Significance

- **p < 0.05**: Significant difference (*)
- **p < 0.01**: Highly significant (**)
- **p < 0.001**: Very highly significant (***)
- **ns**: Not significant

### Effect Size (Cohen's d)

- **< 0.2**: Negligible effect
- **0.2-0.5**: Small effect
- **0.5-0.8**: Medium effect
- **> 0.8**: Large effect

## 🎨 Plot Customization

### IEEE Style Plots

```python
# In plot_generator.py, customize:
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.3
})
```

### Color Schemes

```python
# IEEE standard colors
colors = {
    'whisperpipe': '#1f77b4',  # Blue
    'baseline': '#ff7f0e',     # Orange
    'accent': ['#2ca02c', '#d62728', '#9467bd']  # Green, Red, Purple
}
```

### Figure Sizes

```python
# IEEE standard sizes
sizes = {
    'single_column': (3.5, 2.5),    # IEEE single column
    'double_column': (7.0, 4.0),    # IEEE double column
    'square': (4.0, 4.0)            # Square format
}
```

## 📋 LaTeX Integration

### Using Generated Tables

1. **Copy table code** from `results/run_*/tables/table_*.tex`
2. **Include required packages** in your LaTeX document:

```latex
\\usepackage{booktabs}
\\usepackage{multirow}
\\usepackage{array}
\\usepackage{threeparttable}
```

3. **Insert tables** in your document:

```latex
\\input{table_1_main_results.tex}
\\input{table_2_resource_usage.tex}
```

### Table Features

- **Bold formatting** for best results
- **Significance markers** (*, **, ***)
- **Confidence intervals** (95%)
- **Effect sizes** (Cohen's d)
- **IEEE-compliant formatting**

## 🔍 Troubleshooting

### Common Issues

#### 1. "No audio files found"

```bash
# Check audio directory
ls test_audio/

# Ensure files are .flac or .wav format
file test_audio/*.flac
```

#### 2. "CUDA out of memory"

```bash
# Use smaller model
python run_paper_evaluation.py --config configs/small_model.yaml

# Or force CPU
export CUDA_VISIBLE_DEVICES=""
```

#### 3. "Dependencies not found"

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install torch whisper matplotlib seaborn scipy pandas
```

#### 4. "Statistical analysis not found"

```bash
# Run statistical analysis first
python paper_evaluation/statistical_analysis.py

# Or run full pipeline
python run_paper_evaluation.py
```

### Performance Optimization

#### 1. Reduce Memory Usage

```yaml
# In configs/default.yaml
benchmark:
  audio:
    file_limit: 2                    # Process fewer files
    max_chunk_duration_seconds: 15   # Smaller chunks
  model:
    name: "tiny"                     # Smaller model
```

#### 2. Speed Up Evaluation

```bash
# Quick run with minimal iterations
python run_paper_evaluation.py --quick

# Skip certain steps
python run_paper_evaluation.py --skip-plots --skip-reports
```

#### 3. Parallel Processing

```yaml
# In configs/default.yaml
benchmark:
  runs:
    parallel: true                   # Run iterations in parallel
```

## 📚 Advanced Usage

### Custom Metrics

```python
# Add custom metrics in statistical_analysis.py
def calculate_custom_metric(data):
    # Your custom calculation
    return metric_value
```

### Custom Plots

```python
# Add custom plots in plot_generator.py
def plot_custom_analysis(ax, data):
    # Your custom plotting code
    ax.plot(data)
```

### Custom Reports

```python
# Add custom sections in report_generator.py
def generate_custom_section():
    # Your custom report content
    return markdown_content
```

## 📖 Citation

If you use this system in your research, please cite:

```bibtex
@software{academic_paper_visualization,
  title={Academic Paper Visualization System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/audio2text},
  note={Comprehensive evaluation framework for streaming ASR systems}
}
```

## 🤝 Contributing

### Adding New Metrics

1. **Add metric calculation** in `statistical_analysis.py`
2. **Update plot generation** in `plot_generator.py`
3. **Add to LaTeX tables** in `latex_generator.py`
4. **Include in reports** in `report_generator.py`

### Adding New Plots

1. **Create plot function** in `plot_generator.py`
2. **Add to plot list** in `generate_all_plots()`
3. **Update plot index** in `_generate_plot_index()`

### Adding New Tables

1. **Create table function** in `latex_generator.py`
2. **Add to table list** in `generate_all_tables()`
3. **Update table index** in `_generate_table_index()`

## 📞 Support

### Getting Help

1. **Check logs** in the results directory
2. **Review configuration** in `configs/default.yaml`
3. **Verify dependencies** with `pip list`
4. **Check data format** in `test_audio/` directory

### Reporting Issues

1. **Include error messages** from the logs
2. **Provide system information** (OS, Python version, etc.)
3. **Share configuration** if using custom settings
4. **Describe expected vs actual behavior**

### Feature Requests

1. **Describe the feature** you'd like to see
2. **Explain the use case** and benefits
3. **Provide examples** if possible
4. **Consider contributing** the implementation

## 🎯 Best Practices

### For Academic Papers

1. **Use IEEE style** for conference papers
2. **Include statistical significance** in all comparisons
3. **Report effect sizes** for meaningful differences
4. **Use consistent color schemes** across all plots
5. **Include error bars** and confidence intervals

### For Reproducibility

1. **Save configuration files** with your results
2. **Document system specifications** (hardware, software)
3. **Use version control** for your code
4. **Archive results** with git hashes
5. **Share data and code** when possible

### For Performance

1. **Start with quick runs** for testing
2. **Use appropriate model sizes** for your hardware
3. **Process data in chunks** for large datasets
4. **Monitor resource usage** during evaluation
5. **Save intermediate results** for debugging

---

**Happy evaluating! 🎉**

For more information, see the individual module documentation in the `paper_evaluation/` directory.

