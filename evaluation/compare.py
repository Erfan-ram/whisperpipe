#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparison Evaluation Script

This script runs both WhisperPipe and the naive baseline implementation
side-by-side for comparison and generates metrics for the paper.

Usage:
    python compare.py --duration 60 --model base
"""

import argparse
import time
import sys
import os

# Add parent directory to path to import whisperpipe
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from whisperpipe import pipeStream
from evaluation.naive_whisper import NaiveWhisperStream
from evaluation.metrics import MetricsTracker


def compare_implementations(duration_seconds=60, model_name="base", language="en"):
    """
    Compare WhisperPipe vs Naive implementation
    
    Args:
        duration_seconds: How long to run each test
        model_name: Whisper model to use
        language: Language for transcription
    """
    
    print("="*80)
    print("WHISPERPIPE VS NAIVE WHISPER COMPARISON")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Language: {language}")
    print(f"  Duration: {duration_seconds}s per implementation")
    print(f"\nThis comparison will:")
    print(f"  1. Run naive baseline for {duration_seconds}s")
    print(f"  2. Run WhisperPipe for {duration_seconds}s")
    print(f"  3. Compare metrics")
    print("\n" + "="*80)
    
    # Test 1: Naive Implementation
    print("\n\n📍 PHASE 1: NAIVE BASELINE")
    print("-"*80)
    print("Running naive re-transcription approach...")
    print("Expected: High edit overhead, low stability, processing time growth")
    print("-"*80)
    
    naive = NaiveWhisperStream(
        model_name=model_name,
        language=language,
        processing_interval=1.0,
        debug_mode=False  # Less verbose for comparison
    )
    
    try:
        naive.start_streaming()
        print(f"\n[Recording for {duration_seconds}s - Speak into your microphone]")
        
        # Show countdown
        for i in range(duration_seconds, 0, -10):
            print(f"  {i}s remaining...")
            time.sleep(10 if i >= 10 else i)
        
        naive.stop_streaming()
        naive_metrics = naive.get_metrics()
        
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        naive.stop_streaming()
        naive_metrics = naive.get_metrics()
    except Exception as e:
        print(f"\nError in naive implementation: {e}")
        naive_metrics = None
    finally:
        naive.cleanup()
    
    # Brief pause between tests
    print("\n\nTaking 5 second break before next test...")
    time.sleep(5)
    
    # Test 2: WhisperPipe Implementation
    print("\n\n📍 PHASE 2: WHISPERPIPE")
    print("-"*80)
    print("Running WhisperPipe with dual-buffer architecture...")
    print("Expected: Low edit overhead, high stability, constant processing time")
    print("-"*80)
    
    # Create tracker for WhisperPipe
    pipe_tracker = MetricsTracker()
    
    pipe = pipeStream(
        model_name=model_name,
        language=language,
        finalization_delay=10.0,
        processing_interval=1.0,
        debug_mode=False
    )
    
    # Monkey-patch to track metrics
    original_commit = pipe._commit_to_stable_buffer
    
    def tracked_commit(stable_text, end_time):
        pipe_tracker.record_stable_buffer_update(stable_text)
        pipe_tracker.record_commit_event(stable_text, end_time)
        return original_commit(stable_text, end_time)
    
    pipe._commit_to_stable_buffer = tracked_commit
    
    try:
        pipe_tracker.start_session()
        pipe.start_streaming()
        print(f"\n[Recording for {duration_seconds}s - Speak into your microphone]")
        
        # Show countdown
        for i in range(duration_seconds, 0, -10):
            print(f"  {i}s remaining...")
            time.sleep(10 if i >= 10 else i)
        
        pipe.stop_streaming()
        pipe_tracker.end_session()
        pipe_metrics = pipe_tracker.get_comprehensive_metrics()
        
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        pipe.stop_streaming()
        pipe_tracker.end_session()
        pipe_metrics = pipe_tracker.get_comprehensive_metrics()
    except Exception as e:
        print(f"\nError in WhisperPipe: {e}")
        pipe_metrics = None
    
    # Print comparison results
    print("\n\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    if naive_metrics and pipe_metrics:
        print_comparison(naive_metrics, pipe_metrics)
    else:
        print("\n⚠️  Could not complete comparison (missing metrics)")
        if naive_metrics:
            print("\nNaive metrics:")
            print(naive_metrics)
        if pipe_metrics:
            print("\nWhisperPipe metrics:")
            print(pipe_metrics)


def print_comparison(naive_metrics, pipe_metrics):
    """Print formatted comparison of metrics"""
    
    print("\n📊 EDIT OVERHEAD COMPARISON")
    print("-"*80)
    naive_overhead = naive_metrics.get('edit_overhead', 0)
    pipe_overhead = pipe_metrics.get('edit_overhead', 0)
    
    print(f"  Naive Baseline:    {naive_overhead:.2f}×")
    print(f"  WhisperPipe:       {pipe_overhead:.2f}×")
    
    if naive_overhead > 0:
        reduction = ((naive_overhead - pipe_overhead) / naive_overhead) * 100
        improvement = naive_overhead / pipe_overhead if pipe_overhead > 0 else float('inf')
        print(f"  Improvement:       {reduction:.1f}% reduction ({improvement:.1f}× better)")
    
    print("\n✅ STABILITY COMPARISON")
    print("-"*80)
    naive_stability = naive_metrics.get('stability', 0)
    pipe_stability = pipe_metrics.get('stability_percentage', 0)
    
    print(f"  Naive Baseline:    {naive_stability:.1f}%")
    print(f"  WhisperPipe:       {pipe_stability:.1f}%")
    
    if naive_stability > 0:
        improvement = pipe_stability - naive_stability
        print(f"  Improvement:       +{improvement:.1f} percentage points")
    
    print("\n⏱️  COMMIT LATENCY")
    print("-"*80)
    pipe_latency = pipe_metrics.get('mean_commit_latency_ms', 0)
    print(f"  WhisperPipe:       {pipe_latency:.0f}ms mean commit latency")
    
    print("\n⚡ PROCESSING TIME COMPARISON")
    print("-"*80)
    naive_avg = naive_metrics.get('avg_processing_time', 0)
    naive_max = naive_metrics.get('max_processing_time', 0)
    pipe_avg = pipe_metrics.get('avg_processing_time', 0)
    pipe_max = pipe_metrics.get('max_processing_time', 0)
    
    print(f"  Naive Avg Time:    {naive_avg:.3f}s")
    print(f"  Naive Max Time:    {naive_max:.3f}s")
    print(f"  WhisperPipe Avg:   {pipe_avg:.3f}s")
    print(f"  WhisperPipe Max:   {pipe_max:.3f}s")
    
    if naive_avg > 0:
        speedup = naive_avg / pipe_avg if pipe_avg > 0 else float('inf')
        print(f"  Speedup:           {speedup:.2f}× faster average")
    
    print("\n📈 DETAILED METRICS")
    print("-"*80)
    print("\nNaive Baseline:")
    print(f"  Total transcriptions: {naive_metrics.get('transcription_count', 0)}")
    print(f"  Final word count:     {naive_metrics.get('final_word_count', 0)}")
    print(f"  Total edits:          {naive_metrics.get('edit_count', 0)}")
    
    print("\nWhisperPipe:")
    print(f"  Total commits:        {pipe_metrics.get('total_commits', 0)}")
    print(f"  Total transcriptions: {pipe_metrics.get('total_transcriptions', 0)}")
    print(f"  Final word count:     {pipe_metrics.get('final_word_count', 0)}")
    print(f"  Total edits:          {pipe_metrics.get('total_edits', 0)}")
    
    print("\n" + "="*80)
    print("\n💡 PAPER STATISTICS")
    print("="*80)
    print(f"""
Results demonstrate that WhisperPipe achieves {pipe_overhead:.2f}× edit overhead 
({((naive_overhead - pipe_overhead) / naive_overhead * 100):.0f}% reduction compared to naive re-transcription at {naive_overhead:.1f}×), 
with {pipe_latency:.0f}ms mean commit latency from speech onset to stable buffer.

Our stability analysis shows {pipe_stability:.0f}% transcription consistency, representing 
a {pipe_stability - naive_stability:.0f} percentage point improvement over naive streaming 
approaches ({naive_stability:.0f}% stability).

The dual-buffer architecture prevents exponential growth in processing time, maintaining 
near-constant computational cost per processing cycle while naive approaches exhibit 
linear growth proportional to audio duration.
    """)
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare WhisperPipe vs Naive Whisper implementations'
    )
    parser.add_argument(
        '--duration', 
        type=int, 
        default=60,
        help='Duration in seconds for each test (default: 60)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model to use (default: base)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language code (default: en)'
    )
    
    args = parser.parse_args()
    
    print("\n🎤 PREPARATION")
    print("="*80)
    print("This script will:")
    print("1. Test the naive baseline (re-transcription)")
    print("2. Test WhisperPipe (dual-buffer architecture)")
    print("\nYou will need to speak similar content for both tests")
    print("for a fair comparison.")
    print("\nPress Ctrl+C at any time to stop early.")
    print("="*80)
    
    input("\nPress Enter to begin...")
    
    compare_implementations(
        duration_seconds=args.duration,
        model_name=args.model,
        language=args.language
    )


if __name__ == "__main__":
    main()
