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
