#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics Wrapper for WhisperPipe

This module wraps the WhisperPipe pipeStream class to track detailed
metrics for evaluation purposes, including:
- Edit overhead (number of text changes)
- Commit latency (time from speech to stable buffer)
- Transcription stability/consistency
- Processing time analysis
"""

import time
import numpy as np
from typing import List, Dict, Any


class MetricsTracker:
    """
    Tracks metrics for WhisperPipe evaluation
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        # Edit overhead tracking
        self.stable_buffer_updates = []  # List of (timestamp, text, word_count)
        self.active_transcriptions = []  # List of all intermediate transcriptions
        self.total_word_edits = 0
        self.final_word_count = 0
        
        # Commit latency tracking
        self.commit_events = []  # List of (timestamp, committed_text, audio_duration)
        self.speech_onset_times = []  # When speech starts
        
        # Stability tracking
        self.stability_samples = []  # Track text consistency over time
        self.last_active_text = ""
        
        # Processing metrics
        self.processing_times = []
        self.buffer_sizes = []
        
        # Session info
        self.session_start = None
        self.session_end = None
    
    def start_session(self):
        """Mark session start"""
        self.session_start = time.time()
    
    def end_session(self):
        """Mark session end"""
        self.session_end = time.time()
    
    def record_stable_buffer_update(self, text: str, timestamp: float = None):
        """
        Record when text is committed to stable buffer
        
        Args:
            text: Text committed to stable buffer
            timestamp: Time of commit (defaults to now)
        """
        if timestamp is None:
            timestamp = time.time()
        
        word_count = len(text.split())
        self.stable_buffer_updates.append({
            'timestamp': timestamp,
            'text': text,
            'word_count': word_count
        })
        
        # Calculate edits from last stable update
        if len(self.stable_buffer_updates) > 1:
            prev = self.stable_buffer_updates[-2]['text']
            curr = text
            
            # Count word-level changes
            prev_words = set(prev.split())
            curr_words = set(curr.split())
            added = curr_words - prev_words
            removed = prev_words - curr_words
            
            self.total_word_edits += len(added) + len(removed)
    
    def record_active_transcription(self, text: str, timestamp: float = None):
        """
        Record intermediate transcription (before commitment)
        
        Args:
            text: Current active transcription
            timestamp: Time of transcription
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.active_transcriptions.append({
            'timestamp': timestamp,
            'text': text
        })
        
        # Track stability - how often does active text change?
        if self.last_active_text and text != self.last_active_text:
            # Text changed - less stable
            self.stability_samples.append(0)
        elif self.last_active_text:
            # Text stayed the same - more stable
            self.stability_samples.append(1)
        
        self.last_active_text = text
    
    def record_commit_event(self, committed_text: str, audio_end_time: float, 
                           speech_onset: float = None):
        """
        Record a commit event for latency calculation
        
        Args:
            committed_text: Text being committed
            audio_end_time: End time of audio segment being committed
            speech_onset: When speech actually started (for latency calculation)
        """
        timestamp = time.time()
        
        event = {
            'timestamp': timestamp,
            'committed_text': committed_text,
            'audio_end_time': audio_end_time
        }
        
        if speech_onset is not None:
            event['speech_onset'] = speech_onset
            event['commit_latency'] = timestamp - speech_onset
        
        self.commit_events.append(event)
    
    def record_processing_time(self, duration: float, buffer_size_seconds: float = None):
        """
        Record processing time for a transcription cycle
        
        Args:
            duration: Processing time in seconds
            buffer_size_seconds: Size of audio buffer processed
        """
        self.processing_times.append(duration)
        if buffer_size_seconds is not None:
            self.buffer_sizes.append(buffer_size_seconds)
    
    def record_speech_onset(self, timestamp: float = None):
        """
        Record when speech begins (for latency calculation)
        
        Args:
            timestamp: Time when speech started
        """
        if timestamp is None:
            timestamp = time.time()
        self.speech_onset_times.append(timestamp)
    
    def calculate_edit_overhead(self) -> float:
        """
        Calculate edit overhead: total edits / final word count
        
        Returns:
            float: Edit overhead ratio
        """
        # Get final text from last stable buffer update
        if not self.stable_buffer_updates:
            return 0.0
        
        final_text = self.stable_buffer_updates[-1]['text']
        self.final_word_count = len(final_text.split())
        
        if self.final_word_count == 0:
            return 0.0
        
        # Count all word-level changes in stable buffer
        total_edits = 0
        for i in range(1, len(self.stable_buffer_updates)):
            prev_words = set(self.stable_buffer_updates[i-1]['text'].split())
            curr_words = set(self.stable_buffer_updates[i]['text'].split())
            
            added = curr_words - prev_words
            removed = prev_words - curr_words
            total_edits += len(added) + len(removed)
        
        return total_edits / self.final_word_count
    
    def calculate_mean_commit_latency(self) -> float:
        """
        Calculate mean commit latency in milliseconds
        
        Returns:
            float: Mean latency in ms, or 0 if no data
        """
        latencies = [e['commit_latency'] for e in self.commit_events 
                    if 'commit_latency' in e]
        
        if not latencies:
            return 0.0
        
        return np.mean(latencies) * 1000  # Convert to milliseconds
    
    def calculate_stability(self) -> float:
        """
        Calculate transcription stability/consistency percentage
        
        Returns:
            float: Stability percentage (0-100)
        """
        if not self.stability_samples:
            return 0.0
        
        # Percentage of times text remained stable
        return (sum(self.stability_samples) / len(self.stability_samples)) * 100
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics in a comprehensive dictionary
        
        Returns:
            dict: All calculated metrics
        """
        return {
            'edit_overhead': self.calculate_edit_overhead(),
            'total_edits': sum(len(self.stable_buffer_updates[i]['text'].split()) - 
                             len(self.stable_buffer_updates[i-1]['text'].split())
                             for i in range(1, len(self.stable_buffer_updates))),
            'final_word_count': self.final_word_count,
            'mean_commit_latency_ms': self.calculate_mean_commit_latency(),
            'stability_percentage': self.calculate_stability(),
            'total_commits': len(self.stable_buffer_updates),
            'total_transcriptions': len(self.active_transcriptions),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0,
            'avg_buffer_size': np.mean(self.buffer_sizes) if self.buffer_sizes else 0,
            'session_duration': (self.session_end - self.session_start) 
                              if self.session_end and self.session_start else 0
        }
    
    def print_metrics_report(self):
        """Print a formatted metrics report"""
        metrics = self.get_comprehensive_metrics()
        
        print("\n" + "="*60)
        print("WHISPERPIPE METRICS REPORT")
        print("="*60)
        
        print(f"\n📊 EDIT OVERHEAD")
        print(f"  Edit Overhead Ratio: {metrics['edit_overhead']:.2f}×")
        print(f"  Total Edits: {metrics['total_edits']}")
        print(f"  Final Word Count: {metrics['final_word_count']}")
        
        print(f"\n⏱️  COMMIT LATENCY")
        print(f"  Mean Commit Latency: {metrics['mean_commit_latency_ms']:.0f}ms")
        print(f"  Total Commits: {metrics['total_commits']}")
        
        print(f"\n✅ STABILITY")
        print(f"  Transcription Consistency: {metrics['stability_percentage']:.1f}%")
        print(f"  Total Transcriptions: {metrics['total_transcriptions']}")
        
        print(f"\n⚡ PROCESSING")
        print(f"  Avg Processing Time: {metrics['avg_processing_time']:.3f}s")
        print(f"  Max Processing Time: {metrics['max_processing_time']:.3f}s")
        print(f"  Avg Buffer Size: {metrics['avg_buffer_size']:.1f}s")
        
        if metrics['session_duration'] > 0:
            print(f"\n🕐 SESSION")
            print(f"  Duration: {metrics['session_duration']:.1f}s")
        
        print("="*60 + "\n")


class WhisperPipeWithMetrics:
    """
    Wrapper around pipeStream that tracks metrics
    """
    
    def __init__(self, pipestream_instance):
        """
        Initialize with a pipeStream instance
        
        Args:
            pipestream_instance: Instance of pipeStream to wrap
        """
        self.pipe = pipestream_instance
        self.metrics = MetricsTracker()
        
        # Monkey-patch the commit method to track metrics
        self._original_commit = self.pipe._commit_to_stable_buffer
        self.pipe._commit_to_stable_buffer = self._tracked_commit
        
        # Track finalization
        self._original_finalize = self.pipe._finalize_sentence
        self.pipe._finalize_sentence = self._tracked_finalize
    
    def _tracked_commit(self, stable_text, end_time):
        """Wrapped commit method that tracks metrics"""
        # Record commit event
        self.metrics.record_stable_buffer_update(stable_text)
        self.metrics.record_commit_event(stable_text, end_time)
        
        # Call original method
        return self._original_commit(stable_text, end_time)
    
    def _tracked_finalize(self, final_text=None):
        """Wrapped finalize method that tracks session end"""
        self.metrics.end_session()
        return self._original_finalize(final_text)
    
    def start_streaming(self):
        """Start streaming with metrics tracking"""
        self.metrics.reset()
        self.metrics.start_session()
        return self.pipe.start_streaming()
    
    def stop_streaming(self):
        """Stop streaming and print metrics"""
        result = self.pipe.stop_streaming()
        self.metrics.end_session()
        self.metrics.print_metrics_report()
        return result
    
    def get_metrics(self):
        """Get current metrics"""
        return self.metrics.get_comprehensive_metrics()
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped pipeStream"""
        return getattr(self.pipe, name)
