#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test to verify the evaluation framework without requiring Whisper model
Tests the metrics calculations with mock data
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import (
    calculate_wer,
    calculate_stability_index,
    calculate_prefix_stability,
    calculate_average_latency,
    calculate_metrics_summary
)


def test_wer():
    """Test WER calculation"""
    print("\n" + "="*80)
    print("TEST: Word Error Rate (WER)")
    print("="*80)
    
    test_cases = [
        {
            'reference': "the quick brown fox jumps over the lazy dog",
            'hypothesis': "the quick brown fox jumps over the lazy dog",
            'expected_wer': 0.0
        },
        {
            'reference': "the quick brown fox",
            'hypothesis': "the slow brown fox",
            'expected_wer': 25.0  # 1 substitution out of 4 words
        },
        {
            'reference': "hello world",
            'hypothesis': "hello",
            'expected_wer': 50.0  # 1 deletion out of 2 words
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        wer = calculate_wer(case['reference'], case['hypothesis'])
        print(f"\nTest {i}:")
        print(f"  Reference: '{case['reference']}'")
        print(f"  Hypothesis: '{case['hypothesis']}'")
        print(f"  Expected WER: {case['expected_wer']:.1f}%")
        print(f"  Calculated WER: {wer:.1f}%")
        print(f"  Status: {'✓ PASS' if abs(wer - case['expected_wer']) < 1.0 else '✗ FAIL'}")


def test_stability_index():
    """Test Stability Index calculation"""
    print("\n" + "="*80)
    print("TEST: Stability Index (SI)")
    print("="*80)
    
    # Test case 1: Perfectly stable (identical outputs)
    stable_outputs = [
        "the quick brown fox",
        "the quick brown fox",
        "the quick brown fox"
    ]
    si_stable = calculate_stability_index(stable_outputs)
    print(f"\nTest 1: Perfectly stable outputs")
    print(f"  Outputs: {stable_outputs}")
    print(f"  SI: {si_stable:.2f}%")
    print(f"  Expected: ~100%")
    print(f"  Status: {'✓ PASS' if si_stable > 95 else '✗ FAIL'}")
    
    # Test case 2: Growing prefix (stable start, growing end)
    growing_outputs = [
        "the quick",
        "the quick brown",
        "the quick brown fox",
        "the quick brown fox jumps"
    ]
    si_growing = calculate_stability_index(growing_outputs)
    print(f"\nTest 2: Growing outputs (stable prefix)")
    print(f"  Outputs: {growing_outputs}")
    print(f"  SI: {si_growing:.2f}%")
    print(f"  Expected: ~70-85% (prefix stable, length changes)")
    print(f"  Status: {'✓ PASS' if 60 < si_growing < 90 else '✗ FAIL'}")
    
    # Test case 3: Highly unstable (completely different each time)
    unstable_outputs = [
        "hello world",
        "foo bar baz",
        "testing one two three",
        "completely different text"
    ]
    si_unstable = calculate_stability_index(unstable_outputs)
    print(f"\nTest 3: Highly unstable outputs")
    print(f"  Outputs: {unstable_outputs}")
    print(f"  SI: {si_unstable:.2f}%")
    print(f"  Expected: <50% (high instability)")
    print(f"  Status: {'✓ PASS' if si_unstable < 60 else '✗ FAIL'}")


def test_prefix_stability():
    """Test prefix stability calculation"""
    print("\n" + "="*80)
    print("TEST: Prefix Stability")
    print("="*80)
    
    outputs = [
        "the quick brown",
        "the quick brown fox",
        "the quick brown fox jumps",
    ]
    ps = calculate_prefix_stability(outputs)
    print(f"\nOutputs: {outputs}")
    print(f"Prefix Stability: {ps:.2f}%")
    print(f"Expected: >90% (prefix is preserved)")
    print(f"Status: {'✓ PASS' if ps > 85 else '✗ FAIL'}")


def test_latency():
    """Test latency calculation"""
    print("\n" + "="*80)
    print("TEST: Average Latency")
    print("="*80)
    
    processing_times = [0.150, 0.145, 0.152, 0.148, 0.151]  # seconds
    avg_latency = calculate_average_latency(processing_times)
    expected = 149.2  # ms
    
    print(f"\nProcessing times: {processing_times}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Expected: ~{expected:.1f} ms")
    print(f"Status: {'✓ PASS' if abs(avg_latency - expected) < 1.0 else '✗ FAIL'}")


def test_metrics_summary():
    """Test complete metrics summary"""
    print("\n" + "="*80)
    print("TEST: Complete Metrics Summary")
    print("="*80)
    
    reference = "the quick brown fox jumps over the lazy dog"
    hypothesis = "the quick brown fox jumped over the lazy dog"
    
    intermediate_outputs = [
        "the quick",
        "the quick brown",
        "the quick brown fox",
        "the quick brown fox jumped",
        "the quick brown fox jumped over",
        "the quick brown fox jumped over the",
        "the quick brown fox jumped over the lazy dog"
    ]
    
    processing_times = [0.15, 0.14, 0.15, 0.16, 0.15, 0.14, 0.15]
    
    metrics = calculate_metrics_summary(
        reference,
        hypothesis,
        intermediate_outputs,
        processing_times
    )
    
    print(f"\nReference: '{reference}'")
    print(f"Hypothesis: '{hypothesis}'")
    print(f"\nMetrics:")
    print(f"  WER: {metrics['wer']:.2f}%")
    print(f"  Stability Index: {metrics['stability_index']:.2f}%")
    print(f"  Prefix Stability: {metrics['prefix_stability']:.2f}%")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f} ms")
    print(f"  Num Outputs: {metrics['num_intermediate_outputs']}")
    
    # Validate
    checks = [
        ("WER", metrics['wer'] > 0 and metrics['wer'] < 20),
        ("SI", metrics['stability_index'] > 50),
        ("Prefix Stability", metrics['prefix_stability'] > 80),
        ("Latency", metrics['avg_latency_ms'] > 100 and metrics['avg_latency_ms'] < 200),
        ("Num Outputs", metrics['num_intermediate_outputs'] == 7)
    ]
    
    print(f"\nValidation:")
    for name, passed in checks:
        print(f"  {name}: {'✓ PASS' if passed else '✗ FAIL'}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*80)
    print("EVALUATION FRAMEWORK TEST SUITE")
    print("#"*80)
    
    test_wer()
    test_stability_index()
    test_prefix_stability()
    test_latency()
    test_metrics_summary()
    
    print("\n" + "#"*80)
    print("ALL TESTS COMPLETED")
    print("#"*80 + "\n")


if __name__ == "__main__":
    run_all_tests()
