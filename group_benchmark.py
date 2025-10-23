#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Group benchmark script to compare whisperpipe with baseline Whisper streaming across multiple audio files.
Enhanced with chunked testing for realistic live streaming simulation.
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
from evaluation.metrics import calculate_metrics_summary, calculate_resource_efficiency_index, calculate_memory_growth_rate, calculate_computational_intensity
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

def split_audio_into_chunks(audio, reference, max_chunk_duration_seconds=180):
    """
    Split concatenated audio into fixed-duration chunks for realistic live streaming simulation.
    
    Args:
        audio: Full concatenated audio array
        reference: Full concatenated reference text
        max_chunk_duration_seconds: Maximum duration per chunk (default: 180s = 3 minutes)
    
    Returns:
        List of (audio_chunk, reference_chunk) tuples
    """
    sample_rate = 16000
    max_chunk_samples = int(max_chunk_duration_seconds * sample_rate)
    
    chunks = []
    total_samples = len(audio)
    
    # Split reference text proportionally (rough estimation)
    reference_words = reference.split() if reference else []
    total_duration = total_samples / sample_rate
    
    chunk_start = 0
    chunk_index = 0
    
    while chunk_start < total_samples:
        chunk_end = min(chunk_start + max_chunk_samples, total_samples)
        audio_chunk = audio[chunk_start:chunk_end]
        
        # Calculate proportional reference text for this chunk
        chunk_duration = len(audio_chunk) / sample_rate
        start_ratio = (chunk_start / sample_rate) / total_duration if total_duration > 0 else 0
        end_ratio = (chunk_end / sample_rate) / total_duration if total_duration > 0 else 0
        
        start_word_idx = int(start_ratio * len(reference_words))
        end_word_idx = int(end_ratio * len(reference_words))
        
        reference_chunk = " ".join(reference_words[start_word_idx:end_word_idx]) if reference_words else ""
        
        chunks.append({
            'audio': audio_chunk,
            'reference': reference_chunk,
            'duration': chunk_duration,
            'chunk_index': chunk_index + 1
        })
        
        chunk_start = chunk_end
        chunk_index += 1
    
    return chunks

def create_default_resource_summary():
    """Create a default resource summary structure with all required fields."""
    return {
        'duration_s': 0.0,
        'samples': 0,  # Added samples field
        'gpu_memory': {
            'peak_mb': 0.0,
            'mean_mb': 0.0,
            'std_mb': 0.0,
            'min_mb': 0.0
        },
        'ram': {
            'peak_mb': 0.0,
            'mean_mb': 0.0,
            'std_mb': 0.0,
            'min_mb': 0.0
        },
        'gpu_utilization': {
            'mean_pct': 0.0,
            'peak_pct': 0.0,
            'std_pct': 0.0,
            'min_pct': 0.0
        },
        'cpu': {
            'mean_pct': 0.0,
            'peak_pct': 0.0,
            'std_pct': 0.0,
            'min_pct': 0.0
        }
    }

def validate_and_fix_resource_summary(resource_summary, duration=0.0):
    """Validate and fix resource summary to ensure all required fields are present."""
    if not resource_summary:
        return create_default_resource_summary()
    
    # Ensure duration_s is present
    if 'duration_s' not in resource_summary:
        resource_summary['duration_s'] = duration
    
    # Ensure samples field is present (required by print_resource_summary)
    if 'samples' not in resource_summary:
        # Estimate samples from duration if not present
        estimated_samples = int(duration / 0.5) if duration > 0 else 0  # Assuming 0.5s intervals
        resource_summary['samples'] = estimated_samples
    
    # Define required structure
    required_structure = create_default_resource_summary()
    
    # Fix each section
    for section_name, section_default in required_structure.items():
        if section_name in ['duration_s', 'samples']:
            continue
            
        if section_name not in resource_summary:
            resource_summary[section_name] = section_default.copy()
        else:
            # Ensure all fields in this section exist
            for field_name, default_value in section_default.items():
                if field_name not in resource_summary[section_name]:
                    resource_summary[section_name][field_name] = default_value
    
    return resource_summary

def test_single_chunk_whisperpipe(audio_chunk, chunk_info):
    """Test whisperpipe on a single audio chunk."""
    print(f"\n--- whisperpipe Chunk {chunk_info['chunk_index']} (Duration: {chunk_info['duration']:.1f}s) ---")
    
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
    
    sample_rate = 16000
    pipe_increment_seconds = 2.0
    pipe_increment_samples = int(pipe_increment_seconds * sample_rate)
    audio_duration = len(audio_chunk) / sample_rate

    for i in tqdm(range(int(audio_duration / pipe_increment_seconds) + 1), desc="Processing chunk"):
        start_sample = i * pipe_increment_samples
        end_sample = min((i + 1) * pipe_increment_samples, len(audio_chunk))
        if start_sample >= len(audio_chunk):
            break
        audio_segment = audio_chunk[start_sample:end_sample]
        
        realtime_chunk_size = 1024
        micro_chunks = [audio_segment[j:j+realtime_chunk_size] for j in range(0, len(audio_segment), realtime_chunk_size)]
        chunk_duration = realtime_chunk_size / sample_rate
        
        for micro_chunk in micro_chunks:
            pipe.audio_queue.put(micro_chunk)
            time.sleep(chunk_duration)

    pipe_processing_end_time = time.time()
    pipe_processing_time = pipe_processing_end_time - pipe_start_time

    print(f"Audio feeding complete. Waiting for finalization ({pipe.finalization_delay}s)...")
    time.sleep(pipe.finalization_delay + 2)

    pipe_monitor.stop()
    pipe.stop_streaming()

    pipe_resource_summary = pipe_monitor.get_summary()
    pipe_time_series = pipe_monitor.get_time_series()
    pipe_final_text = " ".join(pipe.get_all_transcribed_text())
    pipe_intermediates_data = pipe.get_intermediate_outputs()
    pipe_intermediates = [o['text'] for o in pipe_intermediates_data] if pipe_intermediates_data else []
    pipe_times = [o['processing_time'] for o in pipe_intermediates_data] if pipe_intermediates_data else []
    
    # Validate and fix resource summary
    pipe_resource_summary = validate_and_fix_resource_summary(pipe_resource_summary, audio_duration)
    
    pipe.close()
    
    return {
        'final_text': pipe_final_text,
        'intermediates': pipe_intermediates,
        'processing_times': pipe_times,
        'resource_summary': pipe_resource_summary,
        'time_series': pipe_time_series,
        'processing_time': audio_duration,
        'chunk_info': chunk_info
    }

def test_single_chunk_baseline(audio_chunk, chunk_info):
    """Test baseline Whisper on a single audio chunk."""
    print(f"\n--- Baseline Chunk {chunk_info['chunk_index']} (Duration: {chunk_info['duration']:.1f}s) ---")
    
    baseline = WhisperBaseline(model_name="base", language="en")
    baseline_monitor = ResourceMonitor(interval=0.5)
    
    baseline_monitor.start()
    baseline_start_time = time.time()

    baseline_intermediates = []
    baseline_times = []
    baseline_final_text = ""
    
    sample_rate = 16000
    baseline_increment_seconds = 1.0
    baseline_increment_samples = int(baseline_increment_seconds * sample_rate)
    audio_duration = len(audio_chunk) / sample_rate

    for i in tqdm(range(1, int(audio_duration / baseline_increment_seconds) + 2), desc="Processing chunk"):
        end_sample = min(i * baseline_increment_samples, len(audio_chunk))
        if end_sample <= 0:
            continue
        if end_sample > len(audio_chunk):
            end_sample = len(audio_chunk)

        audio_segment = audio_chunk[:end_sample]
        transcription, proc_time = baseline.transcribe_progressive_chunk(audio_segment)
        baseline_intermediates.append(transcription)
        baseline_times.append(proc_time)
        baseline_final_text = transcription
        
        if end_sample == len(audio_chunk):
            break

    baseline_processing_end_time = time.time()
    baseline_processing_time = baseline_processing_end_time - baseline_start_time
    
    baseline_monitor.stop()
    baseline_resource_summary = baseline_monitor.get_summary()
    baseline_time_series = baseline_monitor.get_time_series()

    # Validate and fix resource summary
    baseline_resource_summary = validate_and_fix_resource_summary(baseline_resource_summary, audio_duration)

    return {
        'final_text': baseline_final_text,
        'intermediates': baseline_intermediates,
        'processing_times': baseline_times,
        'resource_summary': baseline_resource_summary,
        'time_series': baseline_time_series,
        'processing_time': baseline_processing_time,
        'chunk_info': chunk_info
    }

def aggregate_results(chunk_results, full_reference):
    """Aggregate results from all chunks into comprehensive metrics."""
    if not chunk_results:
        raise ValueError("No chunk results to aggregate")
    
    # Combine all final texts
    combined_final_text = " ".join([result.get('final_text', '') for result in chunk_results])
    
    # Combine all intermediates and processing times
    all_intermediates = []
    all_processing_times = []
    
    for result in chunk_results:
        intermediates = result.get('intermediates', [])
        proc_times = result.get('processing_times', [])
        if isinstance(intermediates, list):
            all_intermediates.extend(intermediates)
        if isinstance(proc_times, list):
            all_processing_times.extend(proc_times)
    
    # Aggregate processing times
    total_processing_time = sum([result.get('processing_time', 0) for result in chunk_results])
    
    # Initialize aggregated structure with all required fields
    aggregated_resources = create_default_resource_summary()
    
    # Collect all values for proper statistics calculation
    chunk_count = len(chunk_results)
    duration_sum = 0.0
    samples_sum = 0
    
    gpu_memory_values = []
    ram_values = []
    gpu_util_values = []
    cpu_values = []
    
    for result in chunk_results:
        res = result.get('resource_summary', {})
        
        # Duration and samples
        duration_sum += res.get('duration_s', 0.0)
        samples_sum += res.get('samples', 0)
        
        # GPU Memory
        gpu_mem = res.get('gpu_memory', {})
        gpu_memory_values.append({
            'peak_mb': gpu_mem.get('peak_mb', 0.0),
            'mean_mb': gpu_mem.get('mean_mb', 0.0),
            'std_mb': gpu_mem.get('std_mb', 0.0),
            'min_mb': gpu_mem.get('min_mb', 0.0)
        })
        
        # RAM
        ram = res.get('ram', {})
        ram_values.append({
            'peak_mb': ram.get('peak_mb', 0.0),
            'mean_mb': ram.get('mean_mb', 0.0),
            'std_mb': ram.get('std_mb', 0.0),
            'min_mb': ram.get('min_mb', 0.0)
        })
        
        # GPU Utilization
        gpu_util = res.get('gpu_utilization', {})
        gpu_util_values.append({
            'mean_pct': gpu_util.get('mean_pct', 0.0),
            'peak_pct': gpu_util.get('peak_pct', 0.0),
            'std_pct': gpu_util.get('std_pct', 0.0),
            'min_pct': gpu_util.get('min_pct', 0.0)
        })
        
        # CPU
        cpu = res.get('cpu', {})
        cpu_values.append({
            'mean_pct': cpu.get('mean_pct', 0.0),
            'peak_pct': cpu.get('peak_pct', 0.0),
            'std_pct': cpu.get('std_pct', 0.0),
            'min_pct': cpu.get('min_pct', 0.0)
        })
    
    # Calculate aggregated statistics
    def safe_max(values_list, field):
        valid_values = [v[field] for v in values_list if v[field] > 0]
        return max(valid_values) if valid_values else 0.0
    
    def safe_mean(values_list, field):
        valid_values = [v[field] for v in values_list if v[field] > 0]
        return sum(valid_values) / len(valid_values) if valid_values else 0.0
    
    def safe_std(values_list, field):
        valid_values = [v[field] for v in values_list if v[field] > 0]
        if len(valid_values) < 2:
            return 0.0
        mean_val = sum(valid_values) / len(valid_values)
        variance = sum((x - mean_val) ** 2 for x in valid_values) / len(valid_values)
        return variance ** 0.5
    
    def safe_min(values_list, field):
        valid_values = [v[field] for v in values_list if v[field] > 0]
        return min(valid_values) if valid_values else 0.0
    
    # Set duration and samples
    aggregated_resources['duration_s'] = duration_sum
    aggregated_resources['samples'] = samples_sum
    
    # GPU Memory
    aggregated_resources['gpu_memory']['peak_mb'] = safe_max(gpu_memory_values, 'peak_mb')
    aggregated_resources['gpu_memory']['mean_mb'] = safe_mean(gpu_memory_values, 'mean_mb')
    aggregated_resources['gpu_memory']['std_mb'] = safe_std(gpu_memory_values, 'mean_mb')
    aggregated_resources['gpu_memory']['min_mb'] = safe_min(gpu_memory_values, 'min_mb')
    
    # RAM
    aggregated_resources['ram']['peak_mb'] = safe_max(ram_values, 'peak_mb')
    aggregated_resources['ram']['mean_mb'] = safe_mean(ram_values, 'mean_mb')
    aggregated_resources['ram']['std_mb'] = safe_std(ram_values, 'mean_mb')
    aggregated_resources['ram']['min_mb'] = safe_min(ram_values, 'min_mb')
    
    # GPU Utilization
    aggregated_resources['gpu_utilization']['mean_pct'] = safe_mean(gpu_util_values, 'mean_pct')
    aggregated_resources['gpu_utilization']['peak_pct'] = safe_max(gpu_util_values, 'peak_pct')
    aggregated_resources['gpu_utilization']['std_pct'] = safe_std(gpu_util_values, 'mean_pct')
    aggregated_resources['gpu_utilization']['min_pct'] = safe_min(gpu_util_values, 'min_pct')
    
    # CPU
    aggregated_resources['cpu']['mean_pct'] = safe_mean(cpu_values, 'mean_pct')
    aggregated_resources['cpu']['peak_pct'] = safe_max(cpu_values, 'peak_pct')
    aggregated_resources['cpu']['std_pct'] = safe_std(cpu_values, 'mean_pct')
    aggregated_resources['cpu']['min_pct'] = safe_min(cpu_values, 'min_pct')
    
    # Combine time series data with enhanced structure
    combined_time_series = {
        'timestamps': [],
        'gpu_memory': {'samples': [], 'unit': 'MB'},
        'gpu_utilization': {'samples': [], 'unit': '%'},
        'cpu': {'samples': [], 'unit': '%'},
        'ram': {'samples': [], 'unit': 'MB'},
        # Legacy format for backward compatibility
        'gpu_memory_mb': [],
        'gpu_util_pct': [],
        'cpu_pct': [],
        'ram_mb': []
    }
    
    for result in chunk_results:
        ts = result.get('time_series', {})
        if 'timestamps' in ts:
            timestamp_data = ts['timestamps']
            if isinstance(timestamp_data, list):
                combined_time_series['timestamps'].extend(timestamp_data)
                
                # Handle structured format
                for metric in ['gpu_memory', 'gpu_utilization', 'cpu', 'ram']:
                    if metric in ts and 'samples' in ts[metric]:
                        samples = ts[metric]['samples']
                        if isinstance(samples, list):
                            combined_time_series[metric]['samples'].extend(samples)
                
                # Handle legacy format
                for legacy_key in ['gpu_memory_mb', 'gpu_util_pct', 'cpu_pct', 'ram_mb']:
                    if legacy_key in ts:
                        legacy_data = ts[legacy_key]
                        if isinstance(legacy_data, list):
                            combined_time_series[legacy_key].extend(legacy_data)
    
    return {
        'final_text': combined_final_text,
        'intermediates': all_intermediates,
        'processing_times': all_processing_times,
        'resource_summary': aggregated_resources,
        'time_series': combined_time_series,
        'total_processing_time': total_processing_time,
        'chunk_count': chunk_count
    }

def run_benchmark(test_files, max_chunk_duration_seconds=180):
    """
    Runs the benchmark on a list of test files with chunked processing.
    
    Args:
        test_files: List of test files
        max_chunk_duration_seconds: Maximum duration per chunk (default: 180s = 3 minutes)
    """
    # Concatenate all audio files and references
    full_audio = []
    full_reference = []
    
    print("Loading and concatenating audio files...")
    for test_file in tqdm(test_files):
        try:
            audio, sr = sf.read(test_file["audio_path"])
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            full_audio.append(audio.astype(np.float32))
            full_reference.append(test_file["reference"])
        except Exception as e:
            print(f"Error loading {test_file['audio_path']}: {e}")
            continue

    if not full_audio:
        print("No audio files could be loaded successfully.")
        return

    audio = np.concatenate(full_audio)
    reference = " ".join(full_reference)
    total_audio_duration = len(audio) / 16000.0
    
    print(f"Total audio duration: {total_audio_duration:.2f} seconds")
    
    # Split audio into chunks
    audio_chunks = split_audio_into_chunks(audio, reference, max_chunk_duration_seconds)
    print(f"Split into {len(audio_chunks)} chunks of max {max_chunk_duration_seconds}s each")
    
    for i, chunk in enumerate(audio_chunks):
        print(f"Chunk {i+1}: {chunk['duration']:.1f}s")

    # ============================================================================
    # Test 1: whisperpipe (chunked)
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 1: whisperpipe (Enhanced Streaming - Chunked Processing)")
    print("="*80)

    pipe_chunk_results = []
    for chunk in audio_chunks:
        try:
            result = test_single_chunk_whisperpipe(chunk['audio'], chunk)
            pipe_chunk_results.append(result)
            print(f"Chunk {chunk['chunk_index']} completed: {len(result.get('final_text', ''))} characters transcribed")
        except Exception as e:
            print(f"Error processing whisperpipe chunk {chunk['chunk_index']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not pipe_chunk_results:
        print("ERROR: No successful whisperpipe chunk results. Cannot proceed with aggregation.")
        return

    # Aggregate whisperpipe results
    try:
        pipe_aggregated = aggregate_results(pipe_chunk_results, reference)
        pipe_metrics = calculate_metrics_summary(
            reference, 
            pipe_aggregated['final_text'], 
            pipe_aggregated['intermediates'], 
            pipe_aggregated['processing_times']
        )

        print("\n--- whisperpipe Aggregated Results ---")
        print(f"Total Chunks Processed: {pipe_aggregated['chunk_count']}")
        print(f"Final Transcription: {pipe_aggregated['final_text']}")
        print(f"WER: {pipe_metrics['wer']:.2f}%")
        print_resource_summary("whisperpipe", pipe_aggregated['resource_summary'], total_audio_duration)
    except Exception as e:
        print(f"ERROR: Failed to aggregate whisperpipe results: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================================
    # Test 2: Baseline Whisper (chunked)
    # ============================================================================
    print("\n" + "="*80)
    print("TEST 2: Whisper Baseline (Chunked Simulated Streaming)")
    print("="*80)

    baseline_chunk_results = []
    for chunk in audio_chunks:
        try:
            result = test_single_chunk_baseline(chunk['audio'], chunk)
            baseline_chunk_results.append(result)
            print(f"Chunk {chunk['chunk_index']} completed: {len(result.get('final_text', ''))} characters transcribed")
        except Exception as e:
            print(f"Error processing baseline chunk {chunk['chunk_index']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not baseline_chunk_results:
        print("ERROR: No successful baseline chunk results. Cannot proceed with aggregation.")
        return

    # Aggregate baseline results
    try:
        baseline_aggregated = aggregate_results(baseline_chunk_results, reference)
        baseline_metrics = calculate_metrics_summary(
            reference, 
            baseline_aggregated['final_text'],
            baseline_aggregated['intermediates'], 
            baseline_aggregated['processing_times']
        )

        print("\n--- Baseline Aggregated Results ---")
        print(f"Total Chunks Processed: {baseline_aggregated['chunk_count']}")
        print(f"Final Transcription: {baseline_aggregated['final_text']}")
        print(f"WER: {baseline_metrics['wer']:.2f}%")
        print_resource_summary("Baseline", baseline_aggregated['resource_summary'], total_audio_duration)
    except Exception as e:
        print(f"ERROR: Failed to aggregate baseline results: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================================================================
    # Comparison
    # ============================================================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY (Aggregated from All Chunks)")
    print("="*80)

    comparison_results = {}
    try:
        print(f"\n{'Metric':<30} {'whisperpipe':<20} {'Baseline':<20} {'Improvement':<20}")
        print("-"*90)

        wer_improvement = ((baseline_metrics['wer'] - pipe_metrics['wer']) / baseline_metrics['wer'] * 100) if baseline_metrics['wer'] > 0 else 0
        print(f"{'WER':<30} {pipe_metrics['wer']:>8.2f}% {baseline_metrics['wer']:>16.2f}% {wer_improvement:>14.1f}%")

        si_improvement = ((pipe_metrics['stability_index'] - baseline_metrics['stability_index']) / baseline_metrics['stability_index'] * 100) if baseline_metrics['stability_index'] > 0 else 0
        print(f"{'Stability Index (SI)':<30} {pipe_metrics['stability_index']:>8.2f}% {baseline_metrics['stability_index']:>16.2f}% {si_improvement:>14.1f}%")

        latency_reduction = baseline_metrics['avg_latency_ms'] - pipe_metrics['avg_latency_ms']
        print(f"{'Avg Latency (ms)':<30} {pipe_metrics['avg_latency_ms']:>8.2f} ms {baseline_metrics['avg_latency_ms']:>13.2f} ms {latency_reduction:>12.2f} ms")

        time_improvement = ((baseline_aggregated['total_processing_time'] - pipe_aggregated['total_processing_time']) / baseline_aggregated['total_processing_time'] * 100) if baseline_aggregated['total_processing_time'] > 0 else 0
        print(f"{'Total Processing Time (s)':<30} {pipe_aggregated['total_processing_time']:>8.2f} s {baseline_aggregated['total_processing_time']:>14.2f} s {time_improvement:>14.1f}%")

        # ============================================================================
        # Resource Usage Comparison
        # ============================================================================
        print("\n" + "="*80)
        print("RESOURCE USAGE COMPARISON (Aggregated)")
        print("="*80)

        print(f"\n{'Resource Metric':<30} {'whisperpipe':<20} {'Baseline':<20} {'Improvement':<20}")
        print("-"*90)

        gpu_mem_improvement = ((baseline_aggregated['resource_summary']['gpu_memory']['peak_mb'] - pipe_aggregated['resource_summary']['gpu_memory']['peak_mb']) / baseline_aggregated['resource_summary']['gpu_memory']['peak_mb'] * 100) if baseline_aggregated['resource_summary']['gpu_memory']['peak_mb'] > 0 else 0
        print(f"{'Peak GPU Memory (MB)':<30} {pipe_aggregated['resource_summary']['gpu_memory']['peak_mb']:>8.1f} MB {baseline_aggregated['resource_summary']['gpu_memory']['peak_mb']:>14.1f} MB {gpu_mem_improvement:>14.1f}%")

        ram_improvement = ((baseline_aggregated['resource_summary']['ram']['peak_mb'] - pipe_aggregated['resource_summary']['ram']['peak_mb']) / baseline_aggregated['resource_summary']['ram']['peak_mb'] * 100) if baseline_aggregated['resource_summary']['ram']['peak_mb'] > 0 else 0
        print(f"{'Peak RAM (MB)':<30} {pipe_aggregated['resource_summary']['ram']['peak_mb']:>8.1f} MB {baseline_aggregated['resource_summary']['ram']['peak_mb']:>14.1f} MB {ram_improvement:>14.1f}%")

        gpu_util_reduction = baseline_aggregated['resource_summary']['gpu_utilization']['mean_pct'] - pipe_aggregated['resource_summary']['gpu_utilization']['mean_pct']
        print(f"{'Mean GPU Utilization (%)':<30} {pipe_aggregated['resource_summary']['gpu_utilization']['mean_pct']:>8.1f}% {baseline_aggregated['resource_summary']['gpu_utilization']['mean_pct']:>15.1f}% {gpu_util_reduction:>13.1f}%")

        pipe_rei = calculate_resource_efficiency_index(pipe_aggregated['resource_summary']['gpu_memory']['peak_mb'], total_audio_duration)
        baseline_rei = calculate_resource_efficiency_index(baseline_aggregated['resource_summary']['gpu_memory']['peak_mb'], total_audio_duration)
        rei_improvement = ((baseline_rei - pipe_rei) / baseline_rei * 100) if baseline_rei > 0 else 0
        print(f"{'Resource Efficiency (MB/s)':<30} {pipe_rei:>8.2f} {baseline_rei:>19.2f} {rei_improvement:>14.1f}%")

        pipe_growth = calculate_memory_growth_rate(pipe_aggregated['time_series']['gpu_memory_mb'], pipe_aggregated['time_series']['timestamps'])
        baseline_growth = calculate_memory_growth_rate(baseline_aggregated['time_series']['gpu_memory_mb'], baseline_aggregated['time_series']['timestamps'])
        print(f"{'Memory Growth Rate (MB/s)':<30} {pipe_growth:>8.3f} {baseline_growth:>19.3f} {'N/A':>14}")

        pipe_ci = calculate_computational_intensity(
            pipe_aggregated['resource_summary']['gpu_utilization']['mean_pct'], 
            pipe_aggregated['total_processing_time'], 
            total_audio_duration,
            pipe_aggregated['resource_summary']['cpu']['mean_pct']
        )
        baseline_ci = calculate_computational_intensity(
            baseline_aggregated['resource_summary']['gpu_utilization']['mean_pct'], 
            baseline_aggregated['total_processing_time'], 
            total_audio_duration,
            baseline_aggregated['resource_summary']['cpu']['mean_pct']
        )
        ci_improvement = ((baseline_ci - pipe_ci) / baseline_ci * 100) if baseline_ci > 0 else 0
        print(f"{'Computational Intensity':<30} {pipe_ci:>8.3f} {baseline_ci:>19.3f} {ci_improvement:>14.1f}%")

        comparison_results = {
            'wer_improvement': wer_improvement,
            'si_improvement': si_improvement,
            'latency_reduction': latency_reduction,
            'time_improvement': time_improvement,
            'gpu_mem_improvement': gpu_mem_improvement,
            'ram_improvement': ram_improvement
        }

    except Exception as e:
        print(f"ERROR in comparison calculations: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "="*80)
    print("CHUNKED TESTING SUMMARY")
    print("="*80)
    print(f"Total Audio Duration: {total_audio_duration:.1f}s")
    print(f"Number of Chunks: {len(audio_chunks)}")
    print(f"Max Chunk Duration: {max_chunk_duration_seconds}s")
    chunk_durations = [f"{chunk['duration']:.1f}s" for chunk in audio_chunks]
    print(f"Chunk Durations: {chunk_durations}")
    
    print("\n" + "="*80)
    print("RESOURCE USAGE ANALYSIS")
    print("="*80)
    print("\nKey Findings:")
    print("1. Resource Efficiency Index (REI): Lower is better - indicates MB per second of audio")
    print("2. Memory Growth Rate: Shows if memory increases over time (leak detection)")
    print("3. Computational Intensity: Lower is better - efficiency of resource usage")
    print("   Note: Uses CPU utilization when GPU utilization is not available")
    print("\nwhisperpipe advantages:")
    print("- Consistent memory usage due to dual-buffer architecture")
    print("- Avoids reprocessing, reducing computational load")
    print("- Stable buffer prevents memory growth as audio increases")
    print("- Chunked processing simulates realistic live streaming scenarios")

    return {
        "whisperpipe": {
            "metrics": pipe_metrics,
            "aggregated": pipe_aggregated,
            "chunks": pipe_chunk_results
        },
        "baseline": {
            "metrics": baseline_metrics,
            "aggregated": baseline_aggregated,
            "chunks": baseline_chunk_results
        },
        "comparison": comparison_results,
        "metadata": {
            "total_audio_duration": total_audio_duration,
            "chunk_count": len(audio_chunks),
            "max_chunk_duration": max_chunk_duration_seconds
        }
    }



if __name__ == "__main__":
    # Enhanced with chunked testing
    # max_chunk_duration_seconds: Maximum duration per chunk (e.g., 180s = 3 minutes)
    MAX_CHUNK_DURATION_SECONDS = 30  # 30 seconds per chunk

    test_files = get_test_files("test_audio", limit=4)  # Your example: 4 files
    if not test_files:
        print("No test files found. Please check the 'test_audio' directory.")
    else:
        print(f"Testing with maximum chunk duration: {MAX_CHUNK_DURATION_SECONDS}s ({MAX_CHUNK_DURATION_SECONDS/60:.1f} minutes)")
        run_benchmark(test_files, max_chunk_duration_seconds=MAX_CHUNK_DURATION_SECONDS)