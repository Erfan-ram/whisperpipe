# Fix Academic Paper Evaluation Data Flow

## Problem Analysis

The current system has a broken data flow where:

1. `group_benchmark.py` collects real benchmark data with detailed metrics
2. `benchmark_runner.py` doesn't properly save this data structure
3. `statistical_analysis.py` receives empty metric arrays
4. `plot_generator.py` falls back to generating mock/fake data

## Solution: Complete Data Flow Fix

### Phase 1: Fix Data Saving in benchmark_runner.py

**File**: `paper_evaluation/benchmark_runner.py`

The `_run_modified_benchmark()` method (line 193-202) calls `run_benchmark()` from `group_benchmark.py` which returns a rich data structure:

```python
{
    "whisperpipe": {
        "metrics": {wer, stability_index, avg_latency_ms, ...},
        "aggregated": {resource_summary, time_series, ...},
        "chunks": [...]
    },
    "baseline": {...},
    "comparison": {...},
    "metadata": {...}
}
```

**Changes needed**:

1. Update `_run_single_benchmark()` to properly extract and structure the returned data
2. Modify `_save_run_results()` to save the complete benchmark data structure
3. Ensure all metrics are flattened into the expected format for statistical_analysis.py

### Phase 2: Fix Data Loading in statistical_analysis.py

**File**: `paper_evaluation/statistical_analysis.py`

The `_extract_metrics()` method (lines 66-110) expects this structure in run results:

```python
{
    'whisperpipe': {'metrics': {wer, stability_index, ...}},
    'baseline': {'metrics': {...}}
}
```

**Changes needed**:

1. Update `_extract_metrics()` to handle the actual data structure from group_benchmark.py
2. Extract nested metrics properly (wer, stability_index, avg_latency_ms, etc.)
3. Extract resource metrics from `aggregated.resource_summary`
4. Map field names correctly (e.g., peak_gpu_memory_mb from gpu_memory.peak_mb)

### Phase 3: Fix Plot Generation to Use Real Data

**File**: `paper_evaluation/plot_generator.py`

Currently uses mock data in these functions:

- `plot_3_memory_usage_time_series()` - lines 262-264
- `plot_4_latency_time_series()` - lines 302-304  
- `plot_6_wer_vs_duration_scatter()` - lines 391-393
- `plot_7_computational_efficiency_radar()` - lines 443-444
- `plot_8_error_analysis_heatmap()` - lines 493-496
- `plot_9_memory_growth_rate()` - lines 537-539
- `plot_10_latency_distribution_histogram()` - lines 586-587

**Changes needed**:

1. Load actual time series data from run results for memory/latency plots
2. Calculate real WER vs duration from chunk-level data
3. Compute actual normalized metrics for radar chart
4. Extract real error breakdowns if available, or remove error heatmap
5. Use actual latency distributions from processing_times

### Phase 4: Enhance Data Structure

**Additional improvements**:

1. Save time series data per run (gpu_memory_mb, timestamps) for temporal plots
2. Save chunk-level metrics for per-chunk analysis
3. Add audio duration per chunk for WER vs duration scatter plot
4. Preserve intermediate outputs for stability analysis

## Implementation Order

1. **benchmark_runner.py** - Fix data extraction and saving
2. **statistical_analysis.py** - Fix data loading and metric extraction  
3. **plot_generator.py** - Replace all mock data with real data loading
4. **Test end-to-end** - Run full pipeline and verify plots use real data

## Key Files to Modify

- `paper_evaluation/benchmark_runner.py` (~50 lines changed)
- `paper_evaluation/statistical_analysis.py` (~30 lines changed)
- `paper_evaluation/plot_generator.py` (~200 lines changed across 10 plot functions)

## Expected Outcome

After these fixes:

- Benchmark results will contain all actual collected metrics
- Statistical analysis will process real data with correct statistics
- All plots will visualize actual benchmark performance
- No more mock/synthetic data in any visualization