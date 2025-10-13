#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for streaming ASR systems
Includes WER, latency, and Stability Index (SI)
"""
import numpy as np
from typing import List, Tuple
import difflib


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER)
    
    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = total words in reference
    
    Args:
        reference: Ground truth transcription
        hypothesis: Predicted transcription
        
    Returns:
        WER as a percentage (0-100)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 100.0
    
    # Use difflib to compute edit operations
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    
    # Count operations
    substitutions = 0
    deletions = 0
    insertions = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            deletions += i2 - i1
        elif tag == 'insert':
            insertions += j2 - j1
    
    # Calculate WER
    wer = (substitutions + deletions + insertions) / len(ref_words) * 100
    
    return wer


def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings
    
    Args:
        s1, s2: Input strings
        
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_stability_index(intermediate_outputs: List[str]) -> float:
    """
    Calculate Stability Index (SI) for streaming ASR
    
    SI measures how consistent intermediate outputs are as new audio is processed.
    Higher SI means more stable outputs (less revision of previous text).
    
    SI = 1 - (average_edit_distance / average_length) * 100
    
    Args:
        intermediate_outputs: List of intermediate transcriptions in chronological order
        
    Returns:
        Stability Index as a percentage (0-100)
        Higher is better (100 = perfectly stable, 0 = completely unstable)
    """
    if len(intermediate_outputs) < 2:
        return 100.0  # Single output is perfectly stable
    
    edit_distances = []
    lengths = []
    
    # Compare consecutive outputs
    for i in range(len(intermediate_outputs) - 1):
        prev = intermediate_outputs[i].lower()
        curr = intermediate_outputs[i + 1].lower()
        
        # Calculate edit distance
        dist = calculate_levenshtein_distance(prev, curr)
        edit_distances.append(dist)
        
        # Track lengths for normalization
        avg_len = (len(prev) + len(curr)) / 2
        lengths.append(avg_len)
    
    # Calculate average normalized edit distance
    if not lengths or sum(lengths) == 0:
        return 100.0
    
    avg_edit_distance = np.mean(edit_distances)
    avg_length = np.mean(lengths)
    
    # Normalize by average length
    normalized_instability = avg_edit_distance / avg_length if avg_length > 0 else 0
    
    # Convert to stability (higher is better)
    stability = max(0, min(100, (1 - normalized_instability) * 100))
    
    return stability


def calculate_prefix_stability(intermediate_outputs: List[str]) -> float:
    """
    Calculate prefix stability - measures how much of the output remains unchanged
    
    This is an alternative stability metric that focuses on prefix preservation.
    
    Args:
        intermediate_outputs: List of intermediate transcriptions
        
    Returns:
        Prefix stability percentage (0-100)
    """
    if len(intermediate_outputs) < 2:
        return 100.0
    
    stability_scores = []
    
    for i in range(len(intermediate_outputs) - 1):
        prev_words = intermediate_outputs[i].lower().split()
        curr_words = intermediate_outputs[i + 1].lower().split()
        
        if not prev_words:
            continue
        
        # Count how many prefix words are preserved
        common_prefix_len = 0
        for j in range(min(len(prev_words), len(curr_words))):
            if prev_words[j] == curr_words[j]:
                common_prefix_len += 1
            else:
                break
        
        # Calculate stability for this transition
        if len(prev_words) > 0:
            score = common_prefix_len / len(prev_words) * 100
            stability_scores.append(score)
    
    if not stability_scores:
        return 100.0
    
    return np.mean(stability_scores)


def calculate_average_latency(processing_times: List[float]) -> float:
    """
    Calculate average processing latency
    
    Args:
        processing_times: List of processing times in seconds
        
    Returns:
        Average latency in milliseconds
    """
    if not processing_times:
        return 0.0
    
    return np.mean(processing_times) * 1000  # Convert to ms


def calculate_metrics_summary(
    reference: str,
    hypothesis: str,
    intermediate_outputs: List[str],
    processing_times: List[float]
) -> dict:
    """
    Calculate all evaluation metrics
    
    Args:
        reference: Ground truth transcription
        hypothesis: Final predicted transcription
        intermediate_outputs: All intermediate outputs
        processing_times: Processing time for each output
        
    Returns:
        Dictionary with all metrics
    """
    wer = calculate_wer(reference, hypothesis)
    si = calculate_stability_index(intermediate_outputs)
    prefix_si = calculate_prefix_stability(intermediate_outputs)
    avg_latency = calculate_average_latency(processing_times)
    
    return {
        'wer': wer,
        'stability_index': si,
        'prefix_stability': prefix_si,
        'avg_latency_ms': avg_latency,
        'num_intermediate_outputs': len(intermediate_outputs)
    }


def calculate_resource_efficiency_index(peak_memory_mb: float, audio_duration_s: float) -> float:
    """
    Calculate Resource Efficiency Index (REI)
    
    REI measures memory efficiency relative to audio duration.
    Lower is better - indicates less memory needed per second of audio.
    
    REI = Peak_Memory_MB / Audio_Duration_s
    
    Args:
        peak_memory_mb: Peak memory usage in megabytes
        audio_duration_s: Duration of audio in seconds
        
    Returns:
        Resource Efficiency Index (MB/s)
    """
    if audio_duration_s <= 0:
        return float('inf')
    
    return peak_memory_mb / audio_duration_s


def calculate_memory_growth_rate(memory_samples: list, timestamps: list) -> float:
    """
    Calculate memory growth rate over time
    
    This metric identifies memory leaks or unbounded growth.
    Uses linear regression on memory usage over time.
    
    Args:
        memory_samples: List of memory measurements in MB
        timestamps: List of corresponding timestamps in seconds
        
    Returns:
        Memory growth rate in MB/second (0 indicates stable, >0 indicates growth)
    """
    if len(memory_samples) < 2 or len(timestamps) < 2:
        return 0.0
    
    import numpy as np
    
    # Simple linear regression: y = mx + b
    # where y = memory, x = time, m = growth rate
    x = np.array(timestamps)
    y = np.array(memory_samples)
    
    # Calculate slope (growth rate)
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    
    return max(0, slope)  # Return 0 if negative (memory decreasing)


def calculate_computational_intensity(gpu_util_pct: float, processing_time: float, audio_duration: float, cpu_util_pct: float = 0.0) -> float:
    """
    Calculate Computational Intensity (CI)
    
    CI measures how much computational resource is used relative to real-time.
    CI = (GPU_Utilization% / 100) * (Processing_Time / Audio_Duration)
    
    If GPU utilization is not available (0%), falls back to CPU utilization.
    
    A value of 1.0 means using 100% GPU for real-time processing.
    Lower is better - indicates more efficient use of resources.
    
    Args:
        gpu_util_pct: Average GPU utilization percentage
        processing_time: Total processing time in seconds
        audio_duration: Audio duration in seconds
        cpu_util_pct: Average CPU utilization percentage (fallback if GPU is 0)
        
    Returns:
        Computational Intensity (dimensionless)
    """
    if audio_duration <= 0:
        return float('inf')
    
    # Use GPU utilization if available, otherwise fall back to CPU
    util_pct = gpu_util_pct if gpu_util_pct > 0 else cpu_util_pct
    
    # Normalize utilization to 0-1
    util_factor = util_pct / 100.0
    
    # Time factor (>1 means slower than real-time, <1 means faster)
    time_factor = processing_time / audio_duration
    
    return util_factor * time_factor

