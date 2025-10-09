#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real Audio Test - Test WhisperPipe with actual microphone input
"""

from whisperpipe import pipeStream
from evaluation.metrics import StreamingMetrics
import time

def main():
    print("\n" + "="*60)
    print(" REAL AUDIO TEST - WhisperPipe Evaluation")
    print("="*60)
    
    # What you're going to say (for WER calculation)
    reference = input("\nWhat will you say? (e.g., 'the quick brown fox jumps over the lazy dog'): ")
    duration = int(input("How many seconds to record? (default 10): ") or "10")
    
    # Create transcriber with evaluation enabled
    transcriber = pipeStream(
        model_name="base",
        language="en",
        enable_evaluation=True,  # This enables logging
        finalization_delay=5.0,  # Shorter delay for testing
        debug_mode=False  # Less verbose output
    )
    
    print(f"\n🎤 Speak into your microphone for {duration} seconds...")
    print(f"📝 Say: '{reference}'")
    print("▶️  Starting in 3 seconds...\n")
    time.sleep(3)
    
    # Start recording
    transcriber.start_streaming()
    time.sleep(duration)
    transcriber.stop_streaming()
    
    # Get results from logger
    if transcriber.logger:
        history = transcriber.logger.get_transcription_history()
        commits = transcriber.logger.get_stable_commits()
        
        print("\n" + "="*60)
        print("📊 TRANSCRIPTION HISTORY (All Intermediate Outputs):")
        print("="*60)
        for i, text in enumerate(history, 1):
            print(f"  {i}. {text}")
        
        print("\n" + "="*60)
        print("✅ STABLE COMMITS (Text Committed to Buffer):")
        print("="*60)
        for i, commit in enumerate(commits, 1):
            latency_ms = commit['commit_latency'] * 1000
            print(f"  {i}. [{latency_ms:.0f}ms latency] {commit['text']}")
        
        # Calculate metrics
        if history:
            final_output = history[-1]
            
            print("\n" + "="*60)
            print("📈 METRICS:")
            print("="*60)
            
            report = StreamingMetrics.generate_report(
                reference=reference,
                hypothesis=final_output,
                transcription_history=history,
                stable_commits=commits
            )
            
            print(f"  Final WER: {report['final_wer']:.2f}%")
            print(f"  Edit Overhead: {report['edit_overhead']:.2f}x")
            print(f"  Stability Score: {report['stability_score']:.2f}%")
            print(f"  Transcription Changes: {report['transcription_changes']}")
            
            if 'mean_commit_latency_ms' in report:
                print(f"  Avg Commit Latency: {report['mean_commit_latency_ms']:.2f} ms")
            
            print(f"\n  Reference:  '{reference}'")
            print(f"  Your Speech: '{final_output}'")
            print("="*60 + "\n")
    else:
        print("⚠️  No logger available - evaluation not enabled")
    
    transcriber.close()

if __name__ == "__main__":
    main()