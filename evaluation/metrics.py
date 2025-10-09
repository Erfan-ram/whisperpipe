#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streaming ASR Evaluation Metrics
Specialized metrics for real-time transcription systems
"""

import numpy as np
from typing import List, Dict, Tuple
import re

try:
    from jiwer import wer as jiwer_wer, cer as jiwer_cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("[WARNING] jiwer not installed. Install with: pip install jiwer")
    print("[INFO] Using fallback WER calculation")


class StreamingMetrics:
    """
    Calculate metrics specific to streaming ASR systems
    
    Metrics include:
    - Final WER (Word Error Rate)
    - Edit Overhead (how much reprocessing occurred)
    - Stability Score (consistency of intermediate outputs)
    - Commit Latency (time to stabilization)
    """
    
    @staticmethod
    def calculate_final_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate on final transcription
        
        Args:
            reference: Ground truth text
            hypothesis: Transcribed text
            
        Returns:
            WER as percentage (0-100)
        """
        if JIWER_AVAILABLE:
            try:
                error_rate = jiwer_wer(reference, hypothesis)
                return error_rate * 100
            except Exception as e:
                print(f"[WARNING] jiwer failed: {e}, using fallback")
        
        # Fallback implementation
        return StreamingMetrics._calculate_wer_fallback(reference, hypothesis) * 100
    
    @staticmethod
    def _calculate_wer_fallback(reference: str, hypothesis: str) -> float:
        """Fallback WER calculation using Levenshtein distance"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0
        
        distance = StreamingMetrics._levenshtein_distance(ref_words, hyp_words)
        return distance / len(ref_words)
    
    @staticmethod
    def calculate_edit_overhead(transcription_history: List[str]) -> float:
        """
        Calculate edit overhead: how many edits occurred before finalization
        
        Lower is better - indicates more stable transcriptions
        
        Args:
            transcription_history: List of intermediate transcriptions
            
        Returns:
            Edit overhead ratio (e.g., 1.5 = 50% more edits than final length)
        """
        if len(transcription_history) < 2:
            return 0.0
        
        final_text = transcription_history[-1]
        final_word_count = len(final_text.split())
        
        if final_word_count == 0:
            return 0.0
        
        total_edits = 0
        
        # Calculate edits between consecutive transcriptions
        for i in range(len(transcription_history) - 1):
            current = transcription_history[i]
            next_text = transcription_history[i + 1]
            
            current_words = current.split()
            next_words = next_text.split()
            
            edits = StreamingMetrics._levenshtein_distance(current_words, next_words)
            total_edits += edits
        
        # Normalize by final word count
        overhead = total_edits / final_word_count if final_word_count > 0 else 0
        return overhead
    
    @staticmethod
    def calculate_stability_score(transcription_history: List[str]) -> float:
        """
        Calculate stability score: how consistent are intermediate outputs?
        
        Higher is better - indicates words stay stable once transcribed
        
        Args:
            transcription_history: List of intermediate transcriptions
            
        Returns:
            Stability percentage (0-100)
        """
        if len(transcription_history) < 2:
            return 100.0
        
        final_words = transcription_history[-1].split()
        
        if len(final_words) == 0:
            return 0.0
        
        stable_count = 0
        total_checks = 0
        
        # For each intermediate transcription, check if words remained stable
        for intermediate in transcription_history[:-1]:
            intermediate_words = intermediate.split()
            
            # Check position-wise matching with final output
            for i, word in enumerate(intermediate_words):
                if i < len(final_words):
                    if word == final_words[i]:
                        stable_count += 1
                    total_checks += 1
        
        stability = (stable_count / total_checks * 100) if total_checks > 0 else 0
        return stability
    
    @staticmethod
    def calculate_commit_latency(stable_commits: List[Dict]) -> Tuple[float, float, List[float]]:
        """
        Calculate commit latency statistics
        
        Args:
            stable_commits: List of commit events from logger
            
        Returns:
            Tuple of (mean_latency_ms, std_latency_ms, all_latencies_ms)
        """
        if not stable_commits:
            return 0.0, 0.0, []
        
        latencies_ms = []
        
        for commit in stable_commits:
            latency = commit.get('commit_latency', 0) * 1000  # Convert to ms
            latencies_ms.append(latency)
        
        mean_latency = np.mean(latencies_ms) if latencies_ms else 0
        std_latency = np.std(latencies_ms) if latencies_ms else 0
        
        return mean_latency, std_latency, latencies_ms
    
    @staticmethod
    def calculate_transcription_changes(transcription_history: List[str]) -> int:
        """
        Count how many times the transcription changed
        
        Args:
            transcription_history: List of intermediate transcriptions
            
        Returns:
            Number of changes
        """
        changes = 0
        for i in range(len(transcription_history) - 1):
            if transcription_history[i] != transcription_history[i + 1]:
                changes += 1
        return changes
    
    @staticmethod
    def _levenshtein_distance(seq1: List, seq2: List) -> int:
        """
        Calculate Levenshtein distance between two sequences
        
        Args:
            seq1, seq2: Lists of items (words, characters, etc.)
            
        Returns:
            Edit distance
        """
        if len(seq1) < len(seq2):
            return StreamingMetrics._levenshtein_distance(seq2, seq1)
        
        if len(seq2) == 0:
            return len(seq1)
        
        previous_row = list(range(len(seq2) + 1))
        
        for i, item1 in enumerate(seq1):
            current_row = [i + 1]
            for j, item2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (item1 != item2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def generate_report(reference: str, hypothesis: str, transcription_history: List[str], 
                       stable_commits: List[Dict]) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            reference: Ground truth text
            hypothesis: Final transcription
            transcription_history: List of all intermediate transcriptions
            stable_commits: List of stable commit events
            
        Returns:
            Dictionary with all metrics
        """
        report = {
            'final_wer': StreamingMetrics.calculate_final_wer(reference, hypothesis),
            'edit_overhead': StreamingMetrics.calculate_edit_overhead(transcription_history),
            'stability_score': StreamingMetrics.calculate_stability_score(transcription_history),
            'transcription_changes': StreamingMetrics.calculate_transcription_changes(transcription_history),
            'total_transcriptions': len(transcription_history),
            'reference_text': reference,
            'hypothesis_text': hypothesis,
        }
        
        # Add commit latency if available
        if stable_commits:
            mean_latency, std_latency, all_latencies = StreamingMetrics.calculate_commit_latency(stable_commits)
            report['mean_commit_latency_ms'] = mean_latency
            report['std_commit_latency_ms'] = std_latency
            report['all_commit_latencies_ms'] = all_latencies
            report['total_commits'] = len(stable_commits)
        
        return report
    
    @staticmethod
    def print_report(report: Dict):
        """Print formatted evaluation report"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        print(f"Final WER: {report['final_wer']:.2f}%")
        print(f"Edit Overhead: {report['edit_overhead']:.2f}x")
        print(f"Stability Score: {report['stability_score']:.2f}%")
        print(f"Transcription Changes: {report['transcription_changes']}")
        print(f"Total Transcriptions: {report['total_transcriptions']}")
        
        if 'mean_commit_latency_ms' in report:
            print(f"Mean Commit Latency: {report['mean_commit_latency_ms']:.2f} ms")
            print(f"Std Commit Latency: {report['std_commit_latency_ms']:.2f} ms")
            print(f"Total Commits: {report['total_commits']}")
        
        print(f"\nReference: {report['reference_text']}")
        print(f"Hypothesis: {report['hypothesis_text']}")
        print("="*60 + "\n")
