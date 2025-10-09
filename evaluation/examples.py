#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Example: How to Use the Evaluation Framework

This script demonstrates the basic usage of the evaluation framework
for comparing WhisperPipe against the naive baseline.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_basic_comparison():
    """
    Example 1: Basic comparison between implementations
    
    This is the simplest way to compare WhisperPipe vs Naive
    """
    print("="*60)
    print("EXAMPLE 1: Basic Comparison")
    print("="*60)
    print("\nThis example shows how to run the comparison tool:")
    print()
    print("Command:")
    print("  python compare.py --duration 60 --model base")
    print()
    print("What it does:")
    print("  1. Runs naive baseline for 60 seconds")
    print("  2. Runs WhisperPipe for 60 seconds")
    print("  3. Compares metrics and prints results")
    print()
    print("Output includes:")
    print("  - Edit Overhead comparison")
    print("  - Stability comparison")
    print("  - Processing time comparison")
    print("  - Paper-ready statistics")
    print()


def example_file_based_evaluation():
    """
    Example 2: File-based evaluation (no microphone needed)
    
    Demonstrates how to evaluate using pre-recorded audio
    """
    print("="*60)
    print("EXAMPLE 2: File-Based Evaluation")
    print("="*60)
    print("\nThis example shows how to evaluate using audio files:")
    print()
    print("Step 1: Generate test audio")
    print("  python evaluate_audio.py --generate-sample")
    print()
    print("Step 2: Evaluate the audio file")
    print("  python evaluate_audio.py --audio test_audio.wav --model base")
    print()
    print("Benefits:")
    print("  - No microphone required")
    print("  - Reproducible results")
    print("  - Can test on specific audio samples")
    print()


def example_custom_metrics():
    """
    Example 3: Custom metrics collection
    
    Shows how to collect metrics programmatically
    """
    print("="*60)
    print("EXAMPLE 3: Custom Metrics Collection")
    print("="*60)
    print("\nCode example for collecting custom metrics:")
    print()
    print("""
from whisperpipe import pipeStream
from evaluation.metrics import MetricsTracker

# Create transcriber
pipe = pipeStream(model_name="base", language="en")

# Create metrics tracker
tracker = MetricsTracker()

# Wrap commit method to track metrics
original_commit = pipe._commit_to_stable_buffer

def tracked_commit(stable_text, end_time):
    # Record the commit
    tracker.record_stable_buffer_update(stable_text)
    tracker.record_commit_event(stable_text, end_time)
    # Call original
    return original_commit(stable_text, end_time)

pipe._commit_to_stable_buffer = tracked_commit

# Run transcription
tracker.start_session()
pipe.start_streaming()

# ... speak into microphone ...

pipe.stop_streaming()
tracker.end_session()

# Get metrics
metrics = tracker.get_comprehensive_metrics()
print(f"Edit Overhead: {metrics['edit_overhead']:.2f}×")
print(f"Stability: {metrics['stability_percentage']:.1f}%")
print(f"Commit Latency: {metrics['mean_commit_latency_ms']:.0f}ms")
    """)
    print()


def example_batch_testing():
    """
    Example 4: Batch testing for paper statistics
    
    Shows how to run multiple evaluations for statistical significance
    """
    print("="*60)
    print("EXAMPLE 4: Batch Testing")
    print("="*60)
    print("\nFor paper-quality results, run multiple evaluations:")
    print()
    print("Bash script:")
    print("""
#!/bin/bash

# Run 5 evaluations with base model
for i in {1..5}; do
    echo "Running evaluation $i/5..."
    python compare.py --duration 120 --model base > results_run${i}.txt
    sleep 10  # Brief pause between runs
done

# Extract metrics
echo "Edit Overhead Results:"
grep "WhisperPipe:" results_run*.txt | grep "×"

echo "\\nStability Results:"
grep "WhisperPipe:" results_run*.txt | grep "%"

echo "\\nCommit Latency Results:"
grep "mean commit latency" results_run*.txt

# Calculate averages in Python
python << END
import re
import numpy as np

# Parse edit overhead
overheads = []
for i in range(1, 6):
    with open(f'results_run{i}.txt') as f:
        content = f.read()
        match = re.search(r'WhisperPipe:\\s+([0-9.]+)×', content)
        if match:
            overheads.append(float(match.group(1)))

print(f"Average Edit Overhead: {np.mean(overheads):.2f}× (±{np.std(overheads):.2f})")
END
    """)
    print()


def example_troubleshooting():
    """
    Example 5: Common troubleshooting steps
    """
    print("="*60)
    print("EXAMPLE 5: Troubleshooting")
    print("="*60)
    print("\nCommon issues and solutions:")
    print()
    print("1. Import errors:")
    print("   python validate_syntax.py")
    print("   pip install -e .")
    print()
    print("2. No microphone input:")
    print("   # List devices:")
    print("   python -c \"import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]\"")
    print()
    print("3. Out of memory:")
    print("   python compare.py --model tiny --duration 30")
    print()
    print("4. Slow processing:")
    print("   # Use GPU if available")
    print("   python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"")
    print()


def main():
    """Main function - run all examples"""
    
    print("\n" + "="*60)
    print("WHISPERPIPE EVALUATION FRAMEWORK - USAGE EXAMPLES")
    print("="*60)
    print()
    
    examples = [
        ("Basic Comparison", example_basic_comparison),
        ("File-Based Evaluation", example_file_based_evaluation),
        ("Custom Metrics", example_custom_metrics),
        ("Batch Testing", example_batch_testing),
        ("Troubleshooting", example_troubleshooting),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        if i > 1:
            input("\nPress Enter for next example...")
        func()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print()
    print("1. Install dependencies:")
    print("   pip install -e .")
    print()
    print("2. Run a quick test:")
    print("   cd evaluation")
    print("   python compare.py --duration 30 --model tiny")
    print()
    print("3. For full evaluation:")
    print("   python compare.py --duration 120 --model base")
    print()
    print("4. Read detailed documentation:")
    print("   - evaluation/QUICKSTART.md")
    print("   - evaluation/README.md")
    print("   - INSTRUCTIONS.md")
    print()
    print("="*60)


if __name__ == "__main__":
    main()
