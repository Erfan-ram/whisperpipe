#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fair Comparison: WhisperPipe vs Naive Baseline
Both process every 1 second, but:
- WhisperPipe: Only processes new audio (dual buffer)
- Naive: Re-processes entire buffer (grows linearly)
"""

import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from whisperpipe import pipeStream
from baselines.naive_streaming import NaiveStreamingWhisper
from evaluation.metrics import StreamingMetrics
from evaluation.logger import TranscriptionLogger

def test_system(system_name, system, duration=10):
    """Test a single system"""
    print(f"\n{'='*60}")
    print(f"Testing: {system_name}")
    print(f"{'='*60}")
    print(f"Speak for {duration} seconds...")
    print("Starting in 3 seconds...\n")
    time.sleep(3)
    
    system.start_streaming()
    time.sleep(duration)
    system.stop_streaming()
    
    # Get results
    results = {}
    
    if system.logger:
        results['history'] = system.logger.get_transcription_history()
        results['commits'] = system.logger.get_stable_commits()
        results['summary'] = system.logger.get_summary()
    
    # Get resource stats
    if hasattr(system, 'get_resource_stats'):
        results['resources'] = system.get_resource_stats()
    
    return results

def main():
    print("\n" + "="*70)
    print(" FAIR COMPARISON: WhisperPipe vs Naive Baseline")
    print(" Both systems process every 1 second")
    print("="*70)
    
    reference = input("\nWhat will you say? (for WER calculation): ")
    duration = int(input("Recording duration in seconds (default 10): ") or "10")
    
    # Test 1: WhisperPipe
    print("\n\n" + "="*70)
    print(" TEST 1: WhisperPipe (Dual-Buffer System)")
    print(" → Processes only NEW audio chunks")
    print(" → Dual buffer prevents reprocessing stable content")
    print("="*70)
    
    whisperpipe = pipeStream(
        model_name="base",
        language="en",
        enable_evaluation=True,
        finalization_delay=5.0,
        processing_interval=1.0,  # Process every 1 second
        debug_mode=False
    )
    
    wp_results = test_system("WhisperPipe", whisperpipe, duration)
    whisperpipe.close()
    
    time.sleep(3)  # Brief pause between tests
    
    # Test 2: Naive Baseline
    print("\n\n" + "="*70)
    print(" TEST 2: Naive Streaming Baseline")
    print(" → Re-processes ENTIRE buffer every time")
    print(" → Buffer grows linearly, processing time increases")
    print("="*70)
    
    naive = NaiveStreamingWhisper(
        model_name="base",
        language="en",
        processing_interval=1.0  # Same interval as WhisperPipe
    )
    naive.logger = TranscriptionLogger()
    
    naive_results = test_system("Naive Baseline", naive, duration)
    naive.close()
    
    # Compare results
    print("\n\n" + "="*70)
    print(" COMPREHENSIVE COMPARISON RESULTS")
    print("="*70)
    
    if wp_results and naive_results:
        wp_final = wp_results['history'][-1] if wp_results['history'] else ""
        naive_final = naive_results['history'][-1] if naive_results['history'] else ""
        
        metrics = StreamingMetrics()
        
        # Calculate metrics for both
        wp_report = metrics.generate_report(
            reference=reference,
            hypothesis=wp_final,
            transcription_history=wp_results['history'],
            stable_commits=wp_results.get('commits', [])
        )
        
        naive_report = metrics.generate_report(
            reference=reference,
            hypothesis=naive_final,
            transcription_history=naive_results['history'],
            stable_commits=[]
        )
        
        # Print comparison
        print("\n" + "-"*70)
        print(" 1. TRANSCRIPTION QUALITY METRICS")
        print("-"*70)
        
        print(f"\n{'Metric':<30} {'WhisperPipe':<20} {'Naive Baseline':<20}")
        print("-"*70)
        print(f"{'Final WER':<30} {wp_report['final_wer']:<20.2f}% {naive_report['final_wer']:<20.2f}%")
        print(f"{'Edit Overhead':<30} {wp_report['edit_overhead']:<20.2f}x {naive_report['edit_overhead']:<20.2f}x")
        print(f"{'Stability Score':<30} {wp_report['stability_score']:<20.2f}% {naive_report['stability_score']:<20.2f}%")
        print(f"{'Transcription Changes':<30} {wp_report['transcription_changes']:<20} {naive_report['transcription_changes']:<20}")
        
        if 'mean_commit_latency_ms' in wp_report:
            print(f"{'Avg Commit Latency':<30} {wp_report['mean_commit_latency_ms']:<20.2f}ms {'N/A':<20}")
        
        # Resource comparison
        print("\n" + "-"*70)
        print(" 2. COMPUTATIONAL RESOURCE METRICS")
        print("-"*70)
        
        if 'resources' in wp_results and 'resources' in naive_results:
            wp_res = wp_results['resources']
            naive_res = naive_results['resources']
            
            print(f"\n{'Metric':<30} {'WhisperPipe':<20} {'Naive Baseline':<20}")
            print("-"*70)
            print(f"{'Audio Duration':<30} {wp_res['audio_duration']:<20.2f}s {naive_res['audio_duration']:<20.2f}s")
            print(f"{'Total Processing Time':<30} {wp_res['total_processing_time']:<20.2f}s {naive_res['total_processing_time']:<20.2f}s")
            print(f"{'Processing Overhead':<30} {wp_res['processing_overhead']:<20.2f}x {naive_res['processing_overhead']:<20.2f}x")
            print(f"{'Avg Processing/Cycle':<30} {wp_res['avg_processing_time']:<20.2f}s {naive_res['avg_processing_time']:<20.2f}s")
            print(f"{'Peak Processing Time':<30} {wp_res['peak_processing_time']:<20.2f}s {naive_res['peak_processing_time']:<20.2f}s")
        
        # Calculate improvements
        print("\n" + "-"*70)
        print(" 3. WhisperPipe IMPROVEMENTS")
        print("-"*70)
        
        edit_improvement = ((naive_report['edit_overhead'] - wp_report['edit_overhead']) / 
                           naive_report['edit_overhead'] * 100) if naive_report['edit_overhead'] > 0 else 0
        stability_improvement = wp_report['stability_score'] - naive_report['stability_score']
        
        print(f"\n✅ Edit Overhead Reduction: {edit_improvement:.1f}%")
        print(f"   ({naive_report['edit_overhead']:.2f}x → {wp_report['edit_overhead']:.2f}x)")
        
        print(f"\n✅ Stability Improvement: +{stability_improvement:.1f} percentage points")
        print(f"   ({naive_report['stability_score']:.1f}% → {wp_report['stability_score']:.1f}%)")
        
        if 'resources' in wp_results and 'resources' in naive_results:
            processing_improvement = ((naive_res['total_processing_time'] - wp_res['total_processing_time']) /
                                     naive_res['total_processing_time'] * 100) if naive_res['total_processing_time'] > 0 else 0
            
            print(f"\n✅ Processing Time Reduction: {processing_improvement:.1f}%")
            print(f"   ({naive_res['total_processing_time']:.2f}s → {wp_res['total_processing_time']:.2f}s)")
            
            overhead_improvement = ((naive_res['processing_overhead'] - wp_res['processing_overhead']) /
                                   naive_res['processing_overhead'] * 100) if naive_res['processing_overhead'] > 0 else 0
            
            print(f"\n✅ Computational Overhead Reduction: {overhead_improvement:.1f}%")
            print(f"   ({naive_res['processing_overhead']:.2f}x → {wp_res['processing_overhead']:.2f}x real-time)")
        
        print("\n" + "="*70)
        print(" SUMMARY FOR PAPER")
        print("="*70)
        print(f"\nEdit Overhead: {wp_report['edit_overhead']:.2f}x (vs {naive_report['edit_overhead']:.2f}x naive)")
        print(f"Stability Score: {wp_report['stability_score']:.1f}% (vs {naive_report['stability_score']:.1f}% naive)")
        if 'mean_commit_latency_ms' in wp_report:
            print(f"Commit Latency: {wp_report['mean_commit_latency_ms']:.0f}ms")
        if 'resources' in wp_results:
            print(f"Processing Overhead: {wp_res['processing_overhead']:.2f}x real-time")
        print(f"\nImprovement: {edit_improvement:.0f}% edit reduction, +{stability_improvement:.0f}pp stability")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()