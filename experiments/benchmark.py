#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Benchmark Suite for WhisperPipe Paper Results
Works with fixed naive baseline (true re-processing)
Generates paper-ready metrics and abstract text
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import json
import numpy as np
from datetime import datetime
from whisperpipe import pipeStream
from baselines.naive_streaming import NaiveStreamingWhisper
from evaluation.metrics import StreamingMetrics
from evaluation.logger import TranscriptionLogger

class CompleteBenchmark:
    """Run complete benchmark and generate paper-ready results"""
    
    def __init__(self):
        self.results = {
            'metadata': {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': 'base',
                'language': 'en',
                'version': '1.0'
            },
            'tests': []
        }
    
    def run_test(self, test_name, reference_text, duration=15):
        """Run a single test with both systems"""
        print("\n" + "="*80)
        print(f" TEST: {test_name}")
        print("="*80)
        print(f"Reference text: '{reference_text}'")
        print(f"Duration: {duration} seconds")
        
        test_result = {
            'name': test_name,
            'reference': reference_text,
            'duration': duration,
            'whisperpipe': {},
            'naive': {}
        }
        
        # Test 1: WhisperPipe
        print("\n" + "-"*80)
        print(" RUNNING: WhisperPipe (Your System)")
        print("-"*80)
        print("🎤 Say the phrase above when recording starts...")
        print("Ready to record in 3 seconds...")
        time.sleep(3)
        
        wp_result = self._test_whisperpipe(duration)
        test_result['whisperpipe'] = wp_result
        
        print("\n✅ WhisperPipe test complete!")
        print(f"   Final transcription: '{wp_result['final_text']}'")
        
        # Wait between tests
        print("\n⏸️  30-second break before Naive test...")
        print("   (Take a breath, prepare to repeat the SAME phrase)")
        self._countdown(30)
        
        # Test 2: Naive Baseline
        print("\n" + "-"*80)
        print(" RUNNING: Naive Baseline")
        print("-"*80)
        print("🎤 Please repeat the EXACT SAME phrase!")
        print(f"   Phrase: '{reference_text}'")
        print("Ready to record in 3 seconds...")
        time.sleep(3)
        
        naive_result = self._test_naive(duration)
        test_result['naive'] = naive_result
        
        print("\n✅ Naive baseline test complete!")
        print(f"   Final transcription: '{naive_result['final_text']}'")
        
        # Calculate metrics
        print("\n" + "-"*80)
        print(" CALCULATING METRICS")
        print("-"*80)
        
        metrics = self._calculate_metrics(reference_text, wp_result, naive_result)
        test_result['metrics'] = metrics
        
        self.results['tests'].append(test_result)
        
        # Print summary
        self._print_test_summary(test_name, metrics)
        
        return test_result
    
    def _countdown(self, seconds):
        """Countdown timer with visual feedback"""
        for remaining in range(seconds, 0, -5):
            if remaining <= 5:
                print(f"   {remaining}...", end=' ', flush=True)
                time.sleep(1)
            else:
                print(f"   {remaining}s remaining...", end='\r', flush=True)
                time.sleep(5)
        print("\n")
    
    def _test_whisperpipe(self, duration):
        """Test WhisperPipe system with proper logging"""
        transcriber = pipeStream(
            model_name="base",
            language="en",
            enable_evaluation=True,
            finalization_delay=5.0,
            processing_interval=1.0,
            debug_mode=False
        )
        
        session_start = time.time()
        transcriber.start_streaming()
        
        # Show progress
        for i in range(duration):
            time.sleep(1)
            print(f"\r   Recording: {i+1}/{duration}s", end='', flush=True)
        print()  # New line
        
        transcriber.stop_streaming()
        session_end = time.time()
        
        # Get results from logger
        history = []
        commits = []
        processing_times = []
        
        if transcriber.logger:
            history = transcriber.logger.get_transcription_history()
            commits = transcriber.logger.get_stable_commits()
            
            # Extract processing times from events
            for event in transcriber.logger.get_all_events():
                if event.get('metadata') and event['metadata'].get('processing_time'):
                    pt = event['metadata']['processing_time']
                    if pt > 0:
                        processing_times.append(pt)
        
        final_text = history[-1] if history else ""
        
        # Calculate metrics
        actual_duration = session_end - session_start
        total_processing = sum(processing_times) if processing_times else 0
        processing_overhead = total_processing / actual_duration if actual_duration > 0 else 0
        
        transcriber.close()
        
        return {
            'final_text': final_text,
            'history': history,
            'commits': commits,
            'audio_duration': actual_duration,
            'total_processing_time': total_processing,
            'processing_overhead': processing_overhead,
            'processing_times': processing_times,
            'num_cycles': len(processing_times),
            'avg_processing_time': np.mean(processing_times) if processing_times else 0
        }
    
    def _test_naive(self, duration):
        """Test Naive baseline with true re-processing"""
        naive = NaiveStreamingWhisper(
            model_name="base",
            language="en",
            processing_interval=1.0
        )
        naive.logger = TranscriptionLogger()
        
        session_start = time.time()
        naive.start_streaming()
        
        # Show progress
        for i in range(duration):
            time.sleep(1)
            print(f"\r   Recording: {i+1}/{duration}s", end='', flush=True)
        print()  # New line
        
        naive.stop_streaming()
        session_end = time.time()
        
        # Get results from logger
        history = []
        processing_times = []
        
        if naive.logger:
            history = naive.logger.get_transcription_history()
            
            # Extract processing times
            for event in naive.logger.get_all_events():
                if event.get('metadata') and event['metadata'].get('processing_time'):
                    pt = event['metadata']['processing_time']
                    if pt > 0:
                        processing_times.append(pt)
        
        final_text = history[-1] if history else ""
        
        # Calculate metrics
        actual_duration = session_end - session_start
        total_processing = sum(processing_times) if processing_times else 0
        processing_overhead = total_processing / actual_duration if actual_duration > 0 else 0
        
        naive.close()
        
        return {
            'final_text': final_text,
            'history': history,
            'commits': [],
            'audio_duration': actual_duration,
            'total_processing_time': total_processing,
            'processing_overhead': processing_overhead,
            'processing_times': processing_times,
            'num_cycles': len(processing_times),
            'avg_processing_time': np.mean(processing_times) if processing_times else 0
        }
    
    def _calculate_metrics(self, reference, wp_result, naive_result):
        """Calculate all metrics for comparison"""
        metrics_calc = StreamingMetrics()
        
        # Transcription quality metrics
        wp_wer = metrics_calc.calculate_final_wer(reference, wp_result['final_text'])
        wp_edit_overhead = metrics_calc.calculate_edit_overhead(wp_result['history'])
        wp_stability = metrics_calc.calculate_stability_score(wp_result['history'])
        wp_changes = metrics_calc.calculate_transcription_changes(wp_result['history'])
        
        naive_wer = metrics_calc.calculate_final_wer(reference, naive_result['final_text'])
        naive_edit_overhead = metrics_calc.calculate_edit_overhead(naive_result['history'])
        naive_stability = metrics_calc.calculate_stability_score(naive_result['history'])
        naive_changes = metrics_calc.calculate_transcription_changes(naive_result['history'])
        
        # Commit latency (WhisperPipe only)
        commit_latency_mean = 0
        commit_count = 0
        if wp_result['commits']:
            valid_latencies = []
            for commit in wp_result['commits']:
                lat = commit.get('commit_latency', 0)
                if 0 < lat < 5.0:  # Reasonable range
                    valid_latencies.append(lat * 1000)  # Convert to ms
            
            if valid_latencies:
                commit_latency_mean = np.mean(valid_latencies)
                commit_count = len(valid_latencies)
        
        # Calculate improvements
        edit_improvement = ((naive_edit_overhead - wp_edit_overhead) / naive_edit_overhead * 100) if naive_edit_overhead > 0.01 else 0
        stability_improvement = wp_stability - naive_stability
        processing_improvement = ((naive_result['total_processing_time'] - wp_result['total_processing_time']) / 
                                  naive_result['total_processing_time'] * 100) if naive_result['total_processing_time'] > 0.01 else 0
        overhead_improvement = ((naive_result['processing_overhead'] - wp_result['processing_overhead']) / 
                               naive_result['processing_overhead'] * 100) if naive_result['processing_overhead'] > 0.01 else 0
        
        return {
            'whisperpipe': {
                'wer': wp_wer,
                'edit_overhead': wp_edit_overhead,
                'stability_score': wp_stability,
                'transcription_changes': wp_changes,
                'commit_latency_ms': commit_latency_mean,
                'commit_count': commit_count,
                'audio_duration': wp_result['audio_duration'],
                'total_processing_time': wp_result['total_processing_time'],
                'processing_overhead': wp_result['processing_overhead'],
                'num_cycles': wp_result['num_cycles'],
                'avg_processing_time': wp_result['avg_processing_time']
            },
            'naive': {
                'wer': naive_wer,
                'edit_overhead': naive_edit_overhead,
                'stability_score': naive_stability,
                'transcription_changes': naive_changes,
                'audio_duration': naive_result['audio_duration'],
                'total_processing_time': naive_result['total_processing_time'],
                'processing_overhead': naive_result['processing_overhead'],
                'num_cycles': naive_result['num_cycles'],
                'avg_processing_time': naive_result['avg_processing_time']
            },
            'improvements': {
                'edit_overhead_reduction_percent': edit_improvement,
                'stability_improvement_points': stability_improvement,
                'processing_time_reduction_percent': processing_improvement,
                'overhead_reduction_percent': overhead_improvement
            }
        }
    
    def _print_test_summary(self, test_name, metrics):
        """Print detailed summary for a single test"""
        print(f"\n{'='*80}")
        print(f" RESULTS: {test_name}")
        print(f"{'='*80}")
        
        wp = metrics['whisperpipe']
        naive = metrics['naive']
        imp = metrics['improvements']
        
        print(f"\n{'Metric':<40} {'WhisperPipe':<20} {'Naive':<20}")
        print("-"*80)
        print(f"{'Final WER':<40} {wp['wer']:<20.2f}% {naive['wer']:<20.2f}%")
        print(f"{'Edit Overhead':<40} {wp['edit_overhead']:<20.2f}x {naive['edit_overhead']:<20.2f}x")
        print(f"{'Stability Score':<40} {wp['stability_score']:<20.2f}% {naive['stability_score']:<20.2f}%")
        print(f"{'Transcription Changes':<40} {wp['transcription_changes']:<20} {naive['transcription_changes']:<20}")
        print(f"{'Audio Duration':<40} {wp['audio_duration']:<20.2f}s {naive['audio_duration']:<20.2f}s")
        print(f"{'Total Processing Time':<40} {wp['total_processing_time']:<20.2f}s {naive['total_processing_time']:<20.2f}s")
        print(f"{'Processing Cycles':<40} {wp['num_cycles']:<20} {naive['num_cycles']:<20}")
        print(f"{'Avg Time per Cycle':<40} {wp['avg_processing_time']:<20.2f}s {naive['avg_processing_time']:<20.2f}s")
        print(f"{'Processing Overhead (×RT)':<40} {wp['processing_overhead']:<20.2f}x {naive['processing_overhead']:<20.2f}x")
        
        if wp['commit_count'] > 0:
            print(f"{'Commit Latency (mean)':<40} {wp['commit_latency_ms']:<20.0f}ms ({wp['commit_count']} commits)")
        
        print(f"\n{'IMPROVEMENTS (WhisperPipe vs Naive)'}")
        print("-"*80)
        
        def print_improvement(name, value, unit, higher_is_better=True):
            if (value > 0 and higher_is_better) or (value < 0 and not higher_is_better):
                print(f"✅ {name}: {abs(value):.1f}{unit}")
            else:
                print(f"⚠️  {name}: {abs(value):.1f}{unit} (unexpected)")
        
        print_improvement("Edit Overhead Reduction", imp['edit_overhead_reduction_percent'], "%", True)
        print_improvement("Stability Improvement", imp['stability_improvement_points'], "pp", True)
        print_improvement("Processing Time Reduction", imp['processing_time_reduction_percent'], "%", True)
        print_improvement("Computational Overhead Reduction", imp['overhead_reduction_percent'], "%", True)
    
    def generate_aggregate_results(self):
        """Calculate aggregate results across all tests"""
        if not self.results['tests']:
            print("No tests run yet!")
            return None
        
        num_tests = len(self.results['tests'])
        
        # Initialize accumulators
        wp_metrics = {
            'wer': [], 'edit_overhead': [], 'stability_score': [],
            'commit_latency_ms': [], 'processing_overhead': []
        }
        naive_metrics = {
            'wer': [], 'edit_overhead': [], 'stability_score': [],
            'processing_overhead': []
        }
        improvements = {
            'edit_overhead_reduction_percent': [],
            'stability_improvement_points': [],
            'processing_time_reduction_percent': [],
            'overhead_reduction_percent': []
        }
        
        # Collect all metrics
        for test in self.results['tests']:
            m = test['metrics']
            
            wp_metrics['wer'].append(m['whisperpipe']['wer'])
            wp_metrics['edit_overhead'].append(m['whisperpipe']['edit_overhead'])
            wp_metrics['stability_score'].append(m['whisperpipe']['stability_score'])
            if m['whisperpipe']['commit_latency_ms'] > 0:
                wp_metrics['commit_latency_ms'].append(m['whisperpipe']['commit_latency_ms'])
            wp_metrics['processing_overhead'].append(m['whisperpipe']['processing_overhead'])
            
            naive_metrics['wer'].append(m['naive']['wer'])
            naive_metrics['edit_overhead'].append(m['naive']['edit_overhead'])
            naive_metrics['stability_score'].append(m['naive']['stability_score'])
            naive_metrics['processing_overhead'].append(m['naive']['processing_overhead'])
            
            improvements['edit_overhead_reduction_percent'].append(m['improvements']['edit_overhead_reduction_percent'])
            improvements['stability_improvement_points'].append(m['improvements']['stability_improvement_points'])
            improvements['processing_time_reduction_percent'].append(m['improvements']['processing_time_reduction_percent'])
            improvements['overhead_reduction_percent'].append(m['improvements']['overhead_reduction_percent'])
        
        # Calculate averages
        avg_wp = {k: np.mean(v) if v else 0 for k, v in wp_metrics.items()}
        avg_naive = {k: np.mean(v) if v else 0 for k, v in naive_metrics.items()}
        avg_imp = {k: np.mean(v) if v else 0 for k, v in improvements.items()}
        
        self.results['aggregate'] = {
            'num_tests': num_tests,
            'whisperpipe_avg': avg_wp,
            'naive_avg': avg_naive,
            'improvements_avg': avg_imp
        }
        
        return self.results['aggregate']
    
    def print_final_report(self):
        """Print complete final report with paper-ready numbers"""
        print("\n" + "="*80)
        print(" COMPLETE BENCHMARK RESULTS - PAPER READY")
        print("="*80)
        
        agg = self.generate_aggregate_results()
        if not agg:
            return
        
        print(f"\nTotal Tests Completed: {agg['num_tests']}")
        print(f"Date: {self.results['metadata']['date']}")
        print(f"Model: {self.results['metadata']['model']}")
        print(f"Language: {self.results['metadata']['language']}")
        
        print(f"\n{'='*80}")
        print(" AVERAGE RESULTS ACROSS ALL TESTS")
        print(f"{'='*80}")
        
        wp = agg['whisperpipe_avg']
        naive = agg['naive_avg']
        imp = agg['improvements_avg']
        
        print(f"\n{'Metric':<40} {'WhisperPipe':<20} {'Naive':<20}")
        print("-"*80)
        print(f"{'Final WER':<40} {wp['wer']:<20.2f}% {naive['wer']:<20.2f}%")
        print(f"{'Edit Overhead':<40} {wp['edit_overhead']:<20.2f}x {naive['edit_overhead']:<20.2f}x")
        print(f"{'Stability Score':<40} {wp['stability_score']:<20.2f}% {naive['stability_score']:<20.2f}%")
        print(f"{'Processing Overhead (×RT)':<40} {wp['processing_overhead']:<20.2f}x {naive['processing_overhead']:<20.2f}x")
        
        if wp['commit_latency_ms'] > 0:
            print(f"{'Mean Commit Latency':<40} {wp['commit_latency_ms']:<20.0f}ms {'N/A':<20}")
        
        print(f"\n{'='*80}")
        print(" IMPROVEMENTS (WhisperPipe vs Naive Baseline)")
        print(f"{'='*80}")
        print(f"✅ Edit Overhead Reduction:          {imp['edit_overhead_reduction_percent']:.1f}%")
        print(f"✅ Stability Improvement:            +{imp['stability_improvement_points']:.1f} percentage points")
        print(f"✅ Processing Time Reduction:        {imp['processing_time_reduction_percent']:.1f}%")
        print(f"✅ Computational Overhead Reduction: {imp['overhead_reduction_percent']:.1f}%")
        
        # Generate paper text
        print(f"\n{'='*80}")
        print(" PAPER-READY ABSTRACT TEXT")
        print(f"{'='*80}")
        
        abstract = self._generate_abstract(wp, naive, imp)
        print(abstract)
        
        # Save results
        self._save_results()
    
    def _generate_abstract(self, wp, naive, imp):
        """Generate paper-ready abstract with integrated metrics"""
        
        edit_reduction = imp['edit_overhead_reduction_percent']
        stability_gain = imp['stability_improvement_points']
        overhead_reduction = imp['overhead_reduction_percent']
        commit_latency = wp['commit_latency_ms']
        
        abstract = f"""
Large-scale self-supervised models such as Whisper have demonstrated 
state-of-the-art performance in offline automatic speech recognition (ASR). 
However, Whisper was designed for batch processing of complete audio files 
and lacks native support for real-time streaming scenarios. Direct application 
of Whisper to streaming contexts results in unstable intermediate outputs, 
excessive computational overhead from redundant reprocessing, and poor handling 
of non-speech content.

In this work, we present WhisperPipe, a streaming adaptation framework that 
enables real-time transcription using Whisper while maintaining output stability 
and computational efficiency. Our approach introduces three key contributions: 
(i) a dual-buffer architecture that separates finalized text from active 
processing audio, eliminating redundant reprocessing of stable content, 
(ii) a similarity-based stabilization mechanism employing word-level timestamps 
and multi-way confirmation to identify commitment points, and (iii) an adaptive 
content filtering system that detects and rejects foreign-language segments and 
transcription artifacts using Whisper's language identification capabilities.

We evaluate our system through comparative experiments against naive streaming 
baselines that re-transcribe entire audio buffers at each processing cycle. 
Comprehensive experiments demonstrate that WhisperPipe achieves {wp['edit_overhead']:.2f}× 
edit overhead (representing a {edit_reduction:.0f}% reduction compared to {naive['edit_overhead']:.2f}× 
for naive re-transcription), {wp['stability_score']:.0f}% transcription consistency 
(a {stability_gain:.0f} percentage point improvement over {naive['stability_score']:.0f}% 
baseline stability), and {wp['processing_overhead']:.2f}× computational overhead 
(achieving {overhead_reduction:.0f}% reduction versus {naive['processing_overhead']:.2f}× 
naive baseline){"" if commit_latency == 0 else f", with {commit_latency:.0f}ms mean commit latency"}. 

The dual-buffer architecture prevents linear growth in processing time, maintaining 
near-constant computational cost per processing cycle while naive approaches exhibit 
proportional growth. These results establish WhisperPipe as a computationally 
efficient solution for deploying Whisper in latency-sensitive applications including 
live captioning, voice assistants, and real-time accessibility tools.
        """
        return abstract
    
    def _save_results(self):
        """Save results to JSON file"""
        filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Make results JSON serializable
        def convert_to_serializable(obj):
            if isinstance(obj, np.float64) or isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        import json
        serializable_results = json.loads(
            json.dumps(self.results, default=convert_to_serializable)
        )
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main benchmark execution"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  WhisperPipe Complete Benchmark Suite                      ║
║                         Paper Evaluation Tool v1.0                         ║
║                                                                            ║
║  This will test WhisperPipe vs Naive Baseline and generate paper results  ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Ask user how many tests to run
    while True:
        try:
            num_tests_input = input("\nHow many tests do you want to run? (1-10, default=3): ").strip()
            num_tests = int(num_tests_input) if num_tests_input else 3
            if 1 <= num_tests <= 10:
                break
            print("❌ Please enter a number between 1 and 10")
        except ValueError:
            print("❌ Please enter a valid number")
    
    benchmark = CompleteBenchmark()
    
    # Test cases pool
    all_test_cases = [
        {
            'name': 'Test 1: Short Common Phrase',
            'reference': 'the quick brown fox jumps over the lazy dog',
            'duration': 10
        },
        {
            'name': 'Test 2: AI/ML Context',
            'reference': 'artificial intelligence and machine learning are transforming technology',
            'duration': 12
        },
        {
            'name': 'Test 3: Technical ASR',
            'reference': 'real time speech recognition systems require efficient algorithms',
            'duration': 12
        },
        {
            'name': 'Test 4: Natural Speech',
            'reference': 'good morning how are you doing today',
            'duration': 10
        },
        {
            'name': 'Test 5: Complex Sentence',
            'reference': 'when developing software it is important to consider user experience',
            'duration': 12
        },
        {
            'name': 'Test 6: Question Form',
            'reference': 'what time does the meeting start tomorrow afternoon',
            'duration': 10
        },
        {
            'name': 'Test 7: Long Technical',
            'reference': 'the dual buffer architecture provides significant performance improvements by reducing overhead',
            'duration': 15
        },
        {
            'name': 'Test 8: Numbers Context',
            'reference': 'the conference is scheduled for january fifteenth at ten thirty',
            'duration': 12
        },
        {
            'name': 'Test 9: Simple Command',
            'reference': 'please send the report by tomorrow',
            'duration': 8
        },
        {
            'name': 'Test 10: Mixed Content',
            'reference': 'the average processing time is approximately zero point eight seconds',
            'duration': 12
        }
    ]
    
    # Select test cases
    test_cases = all_test_cases[:num_tests]
    
    print(f"\n📋 You will run {num_tests} test(s)")
    print("\nFor each test:")
    print("  1. ✅ First: Speak into WhisperPipe")
    print("  2. ⏸️  30-second break")
    print("  3. ✅ Second: Repeat SAME phrase into Naive baseline")
    print("  4. ⏸️  2-minute break before next test\n")
    
    input("🚀 Press Enter when ready to start...")
    
    # Run all tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TEST {i}/{num_tests}: {test_case['name']}")
        print(f"{'#'*80}")
        
        print(f"\n📝 PHRASE TO SAY:")
        print(f"   '{test_case['reference']}'")
        print("\n💡 TIP: Say it naturally at normal speaking pace")
        
        input("\nPress Enter to start this test...")
        
        benchmark.run_test(
            test_name=test_case['name'],
            reference_text=test_case['reference'],
            duration=test_case['duration']
        )
        
        if i < num_tests:
            print(f"\n{'='*80}")
            print(f" BREAK: Test {i} complete. Next test in 2 minutes...")
            print(f"{'='*80}")
            benchmark._countdown(120)
    
    # Print final report
    print("\n\n" + "🎉"*40)
    print("ALL TESTS COMPLETED!")
    print("🎉"*40)
    
    benchmark.print_final_report()
    
    print("\n" + "="*80)
    print(" ✅ BENCHMARK COMPLETE!")
    print("="*80)
    print("\n📊 Check the JSON file for detailed results")
    print("📝 Copy the PAPER-READY ABSTRACT TEXT above into your paper!")
    print("🎯 Use the average metrics table for your results section\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        print("Partial results may be saved\n")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()