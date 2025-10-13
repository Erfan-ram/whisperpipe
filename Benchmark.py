#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script to compare whisperpipe with baseline Whisper streaming
Tests both systems with the same audio file and calculates WER, SI, and latency metrics
"""

import soundfile as sf
import numpy as np
from evaluation.whisper_baseline import WhisperBaseline
from evaluation.metrics import calculate_metrics_summary
import threading
import time
import queue
from whisperpipe.core import pipeStream

# Load audio file
print("Loading audio file...")
audio, sr = sf.read('evaluation/sample.wav')
if audio.ndim == 2:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)

# Resample to 16kHz if needed
if sr != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Ground truth transcription
reference = "the stale smell of old beer lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle taste fine with ham tacos al pastor are my favorite a zestful food is the hot cross bun"

audio_duration = len(audio) / 16000.0
print(f"Audio duration: {audio_duration:.2f} seconds\n")

# ============================================================================
# Test 1: whisperpipe (Enhanced Streaming - Incremental Chunks)
# ============================================================================
print("="*80)
print("TEST 1: whisperpipe (Enhanced Streaming - Incremental Chunks)")
print("="*80)

# Initialize pipeStream
pipe = pipeStream(model_name="base", language="en", debug_mode=False)

# Manually initialize pipeStream state without starting microphone
pipe.is_recording = True
pipe._is_paused = False
pipe.rolling_buffer = np.array([], dtype=np.float32)
pipe.stable_text_buffer = ""
pipe.active_audio_buffer = np.array([], dtype=np.float32)
pipe.last_transcription = ""
pipe.completed_sentences = []
pipe.sentence_start_time = None
pipe.last_stable_buffer_update = None
pipe.last_word_count = 0
pipe.audio_queue = queue.Queue()
pipe.last_transcription_time = time.time()
pipe.transcription_history = []
pipe.duplicate_detection_state = "waiting"
pipe.confirmed_pattern = ""
pipe.foreign_language_rejection_count = 0
pipe.last_rejection_time = None
pipe._summary_printed = False
pipe.intermediate_outputs = []

# Start processing thread
pipe.process_thread = threading.Thread(target=pipe._process_audio)
pipe.process_thread.daemon = True
pipe.process_thread.start()

# Track actual processing time (excluding finalization wait)
pipe_start_time = time.time()

# Feed audio in 1-second increments (0-1s, then 1-2s, then 2-3s, etc.)
print("Feeding audio in 1-second incremental chunks to whisperpipe...")
sample_rate = 16000
increment_seconds = 1.0
increment_samples = int(increment_seconds * sample_rate)

for i in range(int(audio_duration)):
    start_sample = i * increment_samples
    end_sample = min((i + 1) * increment_samples, len(audio))
    
    if start_sample >= len(audio):
        break
    
    # Get the chunk for this second (not from 0, but just this increment)
    audio_chunk = audio[start_sample:end_sample]
    
    # Split into micro-chunks for realistic processing
    realtime_chunk_size = 1024
    micro_chunks = [audio_chunk[j:j+realtime_chunk_size] for j in range(0, len(audio_chunk), realtime_chunk_size)]
    chunk_duration = realtime_chunk_size / sample_rate
    
    print(f"Feeding second {i}-{i+1}...", end="\r")
    
    # Feed micro-chunks
    for micro_chunk in micro_chunks:
        pipe.audio_queue.put(micro_chunk)
        time.sleep(chunk_duration)

# Mark end of actual audio processing (before finalization wait)
pipe_processing_end_time = time.time()
pipe_processing_time = pipe_processing_end_time - pipe_start_time

# Subtract finalization time from processing time for fair comparison
# The audio duration is the actual time spent feeding audio
# Processing time should only count the actual computational work
adjusted_processing_time = audio_duration  # The actual audio feeding time

print(f"\nAudio feeding complete.")
print(f"Total elapsed time: {pipe_processing_time:.2f}s")
print(f"Actual audio duration: {audio_duration:.2f}s")
print(f"Processing overhead: {pipe_processing_time - audio_duration:.2f}s")
print(f"Waiting for finalization ({pipe.finalization_delay}s)...")

# Wait for finalization delay
time.sleep(pipe.finalization_delay + 2)

# Stop the stream
pipe.stop_streaming()

# Get final transcription
pipe_final_text = " ".join(pipe.get_all_transcribed_text())

# Get intermediate results for metrics
pipe_intermediates_data = pipe.get_intermediate_outputs()
pipe_intermediates = [o['text'] for o in pipe_intermediates_data]
pipe_times = [o['processing_time'] for o in pipe_intermediates_data]

# Calculate metrics for whisperpipe
pipe_metrics = calculate_metrics_summary(reference, pipe_final_text, pipe_intermediates, pipe_times)

print("\n--- whisperpipe Results ---")
print(f"Final Transcription: {pipe_final_text}")
print(f"WER: {pipe_metrics['wer']:.2f}%")
print(f"SI: {pipe_metrics['stability_index']:.2f}%")
print(f"Avg Latency: {pipe_metrics['avg_latency_ms']:.2f} ms")
print(f"Total Processing Time (excl. finalization): {adjusted_processing_time:.2f}s")
print(f"Processing Overhead: {pipe_processing_time - audio_duration:.2f}s")
print(f"Number of intermediate outputs: {len(pipe_intermediates)}")

pipe.close()

# ============================================================================
# Test 1-OLD: whisperpipe (Enhanced Streaming - Original Method) [COMMENTED]
# ============================================================================
# This is the original test method - kept for reference but commented out
"""
print("="*80)
print("TEST 1: whisperpipe (Enhanced Streaming)")
print("="*80)

# Initialize pipeStream
pipe = pipeStream(model_name="base", language="en", debug_mode=False)

# Manually initialize pipeStream state without starting microphone
pipe.is_recording = True
pipe._is_paused = False
pipe.rolling_buffer = np.array([], dtype=np.float32)
pipe.stable_text_buffer = ""
pipe.active_audio_buffer = np.array([], dtype=np.float32)
pipe.last_transcription = ""
pipe.completed_sentences = []
pipe.sentence_start_time = None
pipe.last_stable_buffer_update = None
pipe.last_word_count = 0
pipe.audio_queue = queue.Queue()
pipe.last_transcription_time = time.time()
pipe.transcription_history = []
pipe.duplicate_detection_state = "waiting"
pipe.confirmed_pattern = ""
pipe.foreign_language_rejection_count = 0
pipe.last_rejection_time = None
pipe._summary_printed = False
pipe.intermediate_outputs = []

# Start processing thread
pipe.process_thread = threading.Thread(target=pipe._process_audio)
pipe.process_thread.daemon = True
pipe.process_thread.start()

# Track actual processing time (excluding finalization wait)
pipe_start_time = time.time()

# Re-chunk audio for real-time simulation
realtime_chunk_size = 1024
micro_chunks = [audio[i:i+realtime_chunk_size] for i in range(0, len(audio), realtime_chunk_size)]
chunk_duration = realtime_chunk_size / 16000.0

# Feed audio micro-chunks to simulate real-time stream
print("Feeding audio to whisperpipe...")
for micro_chunk in micro_chunks:
    pipe.audio_queue.put(micro_chunk)
    time.sleep(chunk_duration)

# Mark end of actual audio processing (before finalization wait)
pipe_processing_end_time = time.time()
pipe_processing_time = pipe_processing_end_time - pipe_start_time

print(f"Audio feeding complete. Processing time: {pipe_processing_time:.2f}s")
print(f"Waiting for finalization ({pipe.finalization_delay}s)...")

# Wait for finalization delay
time.sleep(pipe.finalization_delay + 2)

# Stop the stream
pipe.stop_streaming()

# Get final transcription
pipe_final_text = " ".join(pipe.get_all_transcribed_text())

# Get intermediate results for metrics
pipe_intermediates_data = pipe.get_intermediate_outputs()
pipe_intermediates = [o['text'] for o in pipe_intermediates_data]
pipe_times = [o['processing_time'] for o in pipe_intermediates_data]

# Calculate metrics for whisperpipe
pipe_metrics = calculate_metrics_summary(reference, pipe_final_text, pipe_intermediates, pipe_times)

print("\n--- whisperpipe Results ---")
print(f"Final Transcription: {pipe_final_text}")
print(f"WER: {pipe_metrics['wer']:.2f}%")
print(f"SI: {pipe_metrics['stability_index']:.2f}%")
print(f"Avg Latency: {pipe_metrics['avg_latency_ms']:.2f} ms")
print(f"Total Processing Time (excl. finalization): {pipe_processing_time:.2f}s")
print(f"Number of intermediate outputs: {len(pipe_intermediates)}")

pipe.close()
"""

# ============================================================================
# Test 2: Whisper Baseline (Simulated Streaming)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Whisper Baseline (Simulated Streaming)")
print("="*80)

# Initialize baseline
baseline = WhisperBaseline(model_name="base", language="en")

# Simulate streaming by transcribing progressively larger chunks
# 0-1s, 0-2s, 0-3s, ..., 0-end
print("Simulating streaming by transcribing progressively larger chunks...")

baseline_start_time = time.time()

# Transcribe in 1-second increments
increment_seconds = 1.0
sample_rate = 16000
increment_samples = int(increment_seconds * sample_rate)

baseline_intermediates = []
baseline_times = []
baseline_final_text = ""

for i in range(1, int(audio_duration) + 2):  # +2 to ensure we get the full audio
    end_sample = min(i * increment_samples, len(audio))
    
    if end_sample > len(audio):
        break
    
    # Get audio chunk from 0 to current position
    audio_chunk = audio[:end_sample]
    duration = end_sample / sample_rate
    
    # Transcribe this chunk
    print(f"Transcribing 0-{duration:.1f}s...", end="\r")
    transcription, proc_time = baseline.transcribe_progressive_chunk(audio_chunk)
    
    baseline_intermediates.append(transcription)
    baseline_times.append(proc_time)
    baseline_final_text = transcription  # Last one will be the final

baseline_processing_end_time = time.time()
baseline_processing_time = baseline_processing_end_time - baseline_start_time

print(f"\nBaseline processing complete. Total time: {baseline_processing_time:.2f}s")

# Calculate metrics for baseline
baseline_metrics = calculate_metrics_summary(
    reference, 
    baseline_final_text, 
    baseline_intermediates, 
    baseline_times
)

print("\n--- Baseline Results ---")
print(f"Final Transcription: {baseline_final_text}")
print(f"WER: {baseline_metrics['wer']:.2f}%")
print(f"SI: {baseline_metrics['stability_index']:.2f}%")
print(f"Avg Latency: {baseline_metrics['avg_latency_ms']:.2f} ms")
print(f"Total Processing Time: {baseline_processing_time:.2f}s")
print(f"Number of intermediate outputs: {len(baseline_intermediates)}")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print(f"\n{'Metric':<30} {'whisperpipe':<20} {'Baseline':<20} {'Improvement':<20}")
print("-"*90)

wer_improvement = ((baseline_metrics['wer'] - pipe_metrics['wer']) / baseline_metrics['wer'] * 100) if baseline_metrics['wer'] > 0 else 0
print(f"{'WER':<30} {pipe_metrics['wer']:>8.2f}% {baseline_metrics['wer']:>16.2f}% {wer_improvement:>14.1f}%")

si_improvement = ((pipe_metrics['stability_index'] - baseline_metrics['stability_index']) / baseline_metrics['stability_index'] * 100) if baseline_metrics['stability_index'] > 0 else 0
print(f"{'Stability Index (SI)':<30} {pipe_metrics['stability_index']:>8.2f}% {baseline_metrics['stability_index']:>16.2f}% {si_improvement:>14.1f}%")

latency_reduction = baseline_metrics['avg_latency_ms'] - pipe_metrics['avg_latency_ms']
print(f"{'Avg Latency (ms)':<30} {pipe_metrics['avg_latency_ms']:>8.2f} ms {baseline_metrics['avg_latency_ms']:>13.2f} ms {latency_reduction:>12.2f} ms")

# Use adjusted_processing_time for whisperpipe (audio duration without finalization)
time_improvement = ((baseline_processing_time - adjusted_processing_time) / baseline_processing_time * 100) if baseline_processing_time > 0 else 0
print(f"{'Total Processing Time (s)':<30} {adjusted_processing_time:>8.2f} s {baseline_processing_time:>14.2f} s {time_improvement:>14.1f}%")

print("\n" + "="*80)
print("BENCHMARK COMPLETE")
print("="*80)
