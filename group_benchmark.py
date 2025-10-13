#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Group benchmark script to compare whisperpipe with baseline Whisper streaming across multiple audio files.
"""

import soundfile as sf
import numpy as np
import os
import threading
import time
import queue
from tqdm import tqdm

from whisperpipe.core import pipeStream
from evaluation.whisper_baseline import WhisperBaseline
from evaluation.metrics import calculate_metrics_summary
from evaluation.resource_monitor import ResourceMonitor, print_resource_summary

def get_test_files(data_dir, limit=None):
    """
    Scans a directory for audio files and their transcriptions.
    Assumes a structure like LibriSpeech:
    - <speaker>-<chapter>.trans.txt
    - <speaker>-<chapter>-<utterance>.flac
    """
    test_files = []
    trans_file = ""
    for f in os.listdir(data_dir):
        if f.endswith(".trans.txt"):
            trans_file = os.path.join(data_dir, f)
            break

    if not trans_file:
        raise FileNotFoundError(f"No .trans.txt file found in {data_dir}")

    transcriptions = {}
    with open(trans_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                transcriptions[parts[0]] = parts[1]

    audio_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".flac")])
    
    if limit:
        audio_files = audio_files[:limit]

    for audio_file in audio_files:
        utterance_id = os.path.splitext(audio_file)[0]
        if utterance_id in transcriptions:
            test_files.append({
                "audio_path": os.path.join(data_dir, audio_file),
                "reference": transcriptions[utterance_id]
            })
            
    return test_files

def run_benchmark(test_files):
    """
    Runs the benchmark on a list of test files and aggregates the results.
    """
    all_pipe_results = []
    all_baseline_results = []

    # Concatenate all audio files and references
    full_audio = []
    full_reference = []
    
    print("Loading and concatenating audio files...")
    for test_file in tqdm(test_files):
        audio, sr = sf.read(test_file["audio_path"])
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        full_audio.append(audio.astype(np.float32))
        full_reference.append(test_file["reference"])

    audio = np.concatenate(full_audio)
    reference = " ".join(full_reference)
    audio_duration = len(audio) / 16000.0
    
    print(f"Total audio duration: {audio_duration:.2f} seconds")
    print(f"Full reference text: {reference}")

    # ============================================================================
    # Test 1: whisperpipe
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 1: whisperpipe (Enhanced Streaming - Incremental Chunks)")
    print("="*80)

    pipe_monitor = ResourceMonitor(interval=0.5)
    pipe = pipeStream(model_name="base", language="en", debug_mode=False)
    pipe.is_recording = True
    pipe._is_paused = False
    pipe.rolling_buffer = np.array([], dtype=np.float32)
    pipe.audio_queue = queue.Queue()
    pipe.process_thread = threading.Thread(target=pipe._process_audio)
    pipe.process_thread.daemon = True
    
    pipe_monitor.start()
    pipe.process_thread.start()

    pipe_start_time = time.time()
    
    print("Feeding audio to whisperpipe...")
    sample_rate = 16000
    increment_seconds = 2.0
    increment_samples = int(increment_seconds * sample_rate)

    for i in tqdm(range(int(audio_duration / increment_seconds) + 1)):
        start_sample = i * increment_samples
        end_sample = min((i + 1) * increment_samples, len(audio))
        if start_sample >= len(audio):
            break
        audio_chunk = audio[start_sample:end_sample]
        
        realtime_chunk_size = 1024
        micro_chunks = [audio_chunk[j:j+realtime_chunk_size] for j in range(0, len(audio_chunk), realtime_chunk_size)]
        chunk_duration = realtime_chunk_size / sample_rate
        
        for micro_chunk in micro_chunks:
            pipe.audio_queue.put(micro_chunk)
            time.sleep(chunk_duration)

    pipe_processing_end_time = time.time()
    pipe_processing_time = pipe_processing_end_time - pipe_start_time
    adjusted_processing_time = audio_duration

    print(f"\nAudio feeding complete.")
    print(f"Waiting for finalization ({pipe.finalization_delay}s)...")
    time.sleep(pipe.finalization_delay + 2)

    pipe_monitor.stop()
    pipe.stop_streaming()

    pipe_resource_summary = pipe_monitor.get_summary()
    pipe_final_text = " ".join(pipe.get_all_transcribed_text())
    pipe_intermediates_data = pipe.get_intermediate_outputs()
    pipe_intermediates = [o['text'] for o in pipe_intermediates_data]
    pipe_times = [o['processing_time'] for o in pipe_intermediates_data]
    pipe_metrics = calculate_metrics_summary(reference, pipe_final_text, pipe_intermediates, pipe_times)

    print("\n--- whisperpipe Results ---")
    print(f"Final Transcription: {pipe_final_text}")
    print(f"WER: {pipe_metrics['wer']:.2f}%")
    print_resource_summary("whisperpipe", pipe_resource_summary, audio_duration)
    pipe.close()

    # ============================================================================
    # Test 2: Whisper Baseline
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 2: Whisper Baseline (Simulated Streaming)")
    print("="*80)

    baseline = WhisperBaseline(model_name="base", language="en")
    baseline_monitor = ResourceMonitor(interval=0.5)
    
    baseline_monitor.start()
    baseline_start_time = time.time()

    baseline_intermediates = []
    baseline_times = []
    baseline_final_text = ""

    print("Simulating streaming for baseline...")
    for i in tqdm(range(1, int(audio_duration) + 2)):
        end_sample = min(i * increment_samples, len(audio))
        if end_sample > len(audio):
            break
        audio_chunk = audio[:end_sample]
        transcription, proc_time = baseline.transcribe_progressive_chunk(audio_chunk)
        baseline_intermediates.append(transcription)
        baseline_times.append(proc_time)
        baseline_final_text = transcription

    baseline_processing_end_time = time.time()
    baseline_processing_time = baseline_processing_end_time - baseline_start_time
    
    baseline_monitor.stop()
    baseline_resource_summary = baseline_monitor.get_summary()

    baseline_metrics = calculate_metrics_summary(reference, baseline_final_text, baseline_intermediates, baseline_times)

    print("\n--- Baseline Results ---")
    print(f"Final Transcription: {baseline_final_text}")
    print(f"WER: {baseline_metrics['wer']:.2f}%")
    print_resource_summary("Baseline", baseline_resource_summary, audio_duration)

    # ============================================================================
    # Comparison Summary
    # ============================================================================
    print("\n" + "="*80)
    print("OVERALL COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'whisperpipe':<20} {'Baseline':<20}")
    print("-"*70)
    print(f"{'WER':<30} {pipe_metrics['wer']:>8.2f}% {baseline_metrics['wer']:>16.2f}%")
    print(f"{'Stability Index (SI)':<30} {pipe_metrics['stability_index']:>8.2f}% {baseline_metrics['stability_index']:>16.2f}%")
    print(f"{'Avg Latency (ms)':<30} {pipe_metrics['avg_latency_ms']:>8.2f} ms {baseline_metrics['avg_latency_ms']:>13.2f} ms")
    print(f"{'Total Processing Time (s)':<30} {adjusted_processing_time:>8.2f} s {baseline_processing_time:>14.2f} s")
    
    print("\n" + "="*80)
    print("RESOURCE USAGE COMPARISON")
    print("="*80)
    print(f"\n{'Resource Metric':<30} {'whisperpipe':<20} {'Baseline':<20}")
    print("-"*70)
    print(f"{'Peak GPU Memory (MB)':<30} {pipe_resource_summary['gpu_memory']['peak_mb']:>8.1f} MB {baseline_resource_summary['gpu_memory']['peak_mb']:>14.1f} MB")
    print(f"{'Peak RAM (MB)':<30} {pipe_resource_summary['ram']['peak_mb']:>8.1f} MB {baseline_resource_summary['ram']['peak_mb']:>14.1f} MB")
    print(f"{'Mean GPU Utilization (%)':<30} {pipe_resource_summary['gpu_utilization']['mean_pct']:>8.1f}% {baseline_resource_summary['gpu_utilization']['mean_pct']:>15.1f}%")


if __name__ == "__main__":
    # As requested, we'll start by testing with just four files.
    # Set limit=None to run on all files in the directory.
    test_files = get_test_files("test_audio", limit=3)
    if not test_files:
        print("No test files found. Please check the 'test_audio' directory.")
    else:
        run_benchmark(test_files)
