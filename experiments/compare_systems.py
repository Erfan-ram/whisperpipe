#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare WhisperPipe vs Naive Baseline
Real-time microphone test
"""

import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from whisperpipe import pipeStream
from baselines.naive_streaming import NaiveStreamingWhisper
from evaluation.metrics import StreamingMetrics

def test_system(system_name, system, duration=10):
    """Test a single system"""
    print(f"\n{'='*60}")
    print(f"Testing: {system_name}")
    print(f"{'='*60}")
    print(f"Speak for {duration} seconds...")
    
    system.start_streaming()
    time.sleep(duration)
    system.stop_streaming()
    
    if system.logger:
        return {
            'history': system.logger.get_transcription_history(),
            'commits': system.logger.get_stable_commits(),
            'summary': system.logger.get_summary()
        }
    return None

def main():
    print("\n" + "="*70)
    print(" WHISPERPIPE vs NAIVE BASELINE COMPARISON")
    print("="*70)
    
    reference = input("\nWhat will you say? (for WER calculation): ")
    duration = int(input("Recording duration in seconds (default 10): ") or "10")
    
    # Test 1: WhisperPipe
    print("\n\n--- TEST 1: WhisperPipe (Your System) ---")
    whisperpipe = pipeStream(
        model_name="base",
        language="en",
        enable_evaluation=True,
        finalization_delay=5.0,
        debug_mode=False
    )
    wp_results = test_system("WhisperPipe", whisperpipe, duration)
    whisperpipe.close()
    
    time.sleep(2)  # Brief pause
    
    # Test 2: Naive Baseline
    print("\n\n--- TEST 2: Naive Streaming Baseline ---")
    naive = NaiveStreamingWhisper(
        model_name="base",
        language="en"
    )
    from evaluation.logger import TranscriptionLogger
    naive.logger = TranscriptionLogger()
    
    naive_results = test_system("Naive Baseline", naive, duration)
    naive.close()
    
    # Compare results
    print("\n\n" + "="*70)
    print(" COMPARISON RESULTS")
    print("="*70)
    
    if wp_results and naive_results:
        wp_final = wp_results['history'][-1] if wp_results['history'] else ""
        naive_final = naive_results['history'][-1] if naive_results['history'] else ""
        
        metrics = StreamingMetrics()
        
        # WhisperPipe metrics
        wp_report = metrics.generate_report(
            reference=reference,
            hypothesis=wp_final,
            transcription_history=wp_results['history'],
            stable_commits=wp_results['commits']
        )
        
        # Naive metrics
        naive_report = metrics.generate_report(
            reference=reference,
            hypothesis=naive_final,
            transcription_history=naive_results['history'],
            stable_commits=[]
        )
        
        print("\n--- WhisperPipe (Your System) ---")
        print(f"  Final WER: {wp_report['final_wer']:.2f}%")
        print(f"  Edit Overhead: {wp_report['edit_overhead']:.2f}x")
        print(f"  Stability Score: {wp_report['stability_score']:.2f}%")
        print(f"  Transcription Changes: {wp_report['transcription_changes']}")
        if 'mean_commit_latency_ms' in wp_report:
            print(f"  Avg Commit Latency: {wp_report['mean_commit_latency_ms']:.2f} ms")
        
        print("\n--- Naive Baseline ---")
        print(f"  Final WER: {naive_report['final_wer']:.2f}%")
        print(f"  Edit Overhead: {naive_report['edit_overhead']:.2f}x")
        print(f"  Stability Score: {naive_report['stability_score']:.2f}%")
        print(f"  Transcription Changes: {naive_report['transcription_changes']}")
        
        print("\n--- Improvements (WhisperPipe vs Naive) ---")
        edit_improvement = ((naive_report['edit_overhead'] - wp_report['edit_overhead']) / 
                           naive_report['edit_overhead'] * 100) if naive_report['edit_overhead'] > 0 else 0
        stability_improvement = wp_report['stability_score'] - naive_report['stability_score']
        
        print(f"  Edit Overhead Reduction: {edit_improvement:.1f}%")
        print(f"  Stability Improvement: +{stability_improvement:.1f}%")
        
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()