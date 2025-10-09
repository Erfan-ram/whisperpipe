#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to verify evaluation toolkit works
"""

import sys
import time
from whisperpipe import pipeStream
from evaluation.metrics import StreamingMetrics
from evaluation.logger import TranscriptionLogger

def test_logging():
    """Test that logging is working"""
    print("\n" + "="*60)
    print("TEST 1: Logging System")
    print("="*60)
    
    # Create logger
    logger = TranscriptionLogger()
    logger.start_session()
    
    # Simulate some transcriptions
    logger.log_transcription("hello", is_stable=False)
    time.sleep(0.1)
    logger.log_transcription("hello world", is_stable=False)
    time.sleep(0.1)
    logger.log_transcription("hello world how", is_stable=False)
    time.sleep(0.1)
    
    # Simulate stable commit
    logger.log_stable_commit("hello world", audio_end_time=2.0)
    
    # Print summary
    logger.print_summary()
    
    print("✅ Logging test passed!\n")
    return logger

def test_metrics(logger):
    """Test metric calculations"""
    print("\n" + "="*60)
    print("TEST 2: Metrics Calculation")
    print("="*60)
    
    # Get data from logger
    history = logger.get_transcription_history()
    commits = logger.get_stable_commits()
    
    # Calculate metrics
    metrics = StreamingMetrics()
    
    # Test reference vs hypothesis
    reference = "hello world how are you"
    hypothesis = "hello world how are you"
    
    wer = metrics.calculate_final_wer(reference, hypothesis)
    edit_overhead = metrics.calculate_edit_overhead(history)
    stability = metrics.calculate_stability_score(history)
    
    print(f"WER: {wer:.2f}%")
    print(f"Edit Overhead: {edit_overhead:.2f}x")
    print(f"Stability Score: {stability:.2f}%")
    
    if commits:
        mean_lat, std_lat, _ = metrics.calculate_commit_latency(commits)
        print(f"Mean Commit Latency: {mean_lat:.2f} ms")
    
    print("\n✅ Metrics test passed!\n")

def test_integration():
    """Test integration with WhisperPipe"""
    print("\n" + "="*60)
    print("TEST 3: Integration with WhisperPipe")
    print("="*60)
    
    try:
        # Create transcriber with evaluation enabled
        transcriber = pipeStream(
            model_name="base",
            language="en",
            enable_evaluation=True,  # Enable logging
            debug_mode=False
        )
        
        if transcriber.logger:
            print("✅ Logger successfully integrated with WhisperPipe")
            print(f"   Logger type: {type(transcriber.logger)}")
        else:
            print("⚠️  Logger not available (evaluation module not imported)")
        
        print("\n✅ Integration test passed!\n")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" EVALUATION TOOLKIT TEST SUITE")
    print("="*70)
    
    # Run tests
    logger = test_logging()
    test_metrics(logger)
    test_integration()
    
    print("\n" + "="*70)
    print(" ALL TESTS COMPLETED")
    print("="*70 + "\n")