#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transcription Logger for Streaming ASR Evaluation
Tracks all transcription events, stable commits, and timing data
"""

import time
from typing import List, Dict, Optional

class TranscriptionLogger:
    """
    Logger to track streaming transcription events for evaluation
    
    Tracks:
    - All intermediate transcriptions
    - Stable buffer commits
    - Timestamps for latency calculation
    - Word-level changes
    """
    
    def __init__(self):
        """Initialize empty logging structures"""
        self.events = []
        self.transcription_history = []
        self.stable_commits = []
        self.audio_chunks = []
        self.start_time = None
        self.enabled = True
        
    def start_session(self):
        """Mark the start of a transcription session"""
        self.start_time = time.time()
        self.events = []
        self.transcription_history = []
        self.stable_commits = []
        self.audio_chunks = []
        
    def log_transcription(self, text: str, is_stable: bool = False, metadata: Optional[Dict] = None):
        """
        Log a transcription event
        
        Args:
            text: Transcribed text
            is_stable: Whether this is a stable/finalized transcription
            metadata: Additional metadata (e.g., similarity score, word count)
        """
        if not self.enabled:
            return
            
        timestamp = time.time()
        
        event = {
            'timestamp': timestamp,
            'relative_time': timestamp - self.start_time if self.start_time else 0,
            'text': text,
            'is_stable': is_stable,
            'word_count': len(text.split()) if text else 0,
            'metadata': metadata or {}
        }
        
        self.events.append(event)
        
        # Track full history of transcriptions
        if text and text not in self.transcription_history:
            self.transcription_history.append(text)
        elif text:  # Even if duplicate, track it for stability analysis
            self.transcription_history.append(text)
    
    def log_stable_commit(self, committed_text: str, audio_end_time: float, metadata: Optional[Dict] = None):
        """
        Log when text is committed to stable buffer
        
        Args:
            committed_text: Text being committed
            audio_end_time: Timestamp in audio where this text ends (in seconds)
            metadata: Additional metadata
        """
        if not self.enabled:
            return
            
        timestamp = time.time()
        
        commit = {
            'timestamp': timestamp,
            'relative_time': timestamp - self.start_time if self.start_time else 0,
            'text': committed_text,
            'audio_end_time': audio_end_time,
            'commit_latency': timestamp - (self.start_time + audio_end_time) if self.start_time else 0,
            'metadata': metadata or {}
        }
        
        self.stable_commits.append(commit)
    
    def log_audio_chunk(self, chunk_duration: float, chunk_size: int):
        """
        Log audio chunk processing
        
        Args:
            chunk_duration: Duration of audio chunk in seconds
            chunk_size: Size of audio chunk in samples
        """
        if not self.enabled:
            return
            
        timestamp = time.time()
        
        self.audio_chunks.append({
            'timestamp': timestamp,
            'relative_time': timestamp - self.start_time if self.start_time else 0,
            'duration': chunk_duration,
            'size': chunk_size
        })
    
    def get_transcription_history(self) -> List[str]:
        """Get list of all transcribed texts in order"""
        return self.transcription_history.copy()
    
    def get_stable_commits(self) -> List[Dict]:
        """Get list of all stable commits with timing data"""
        return self.stable_commits.copy()
    
    def get_all_events(self) -> List[Dict]:
        """Get all logged events"""
        return self.events.copy()
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of the logging session
        
        Returns:
            Dictionary with session statistics
        """
        total_transcriptions = len(self.transcription_history)
        total_stable_commits = len(self.stable_commits)
        
        # Calculate average commit latency
        commit_latencies = [c['commit_latency'] for c in self.stable_commits]
        avg_commit_latency = sum(commit_latencies) / len(commit_latencies) if commit_latencies else 0
        
        # Get final transcription
        final_text = self.transcription_history[-1] if self.transcription_history else ""
        
        return {
            'total_transcriptions': total_transcriptions,
            'total_stable_commits': total_stable_commits,
            'avg_commit_latency_ms': avg_commit_latency * 1000,
            'final_text': final_text,
            'final_word_count': len(final_text.split()) if final_text else 0,
            'session_duration': time.time() - self.start_time if self.start_time else 0
        }
    
    def print_summary(self):
        """Print human-readable summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("TRANSCRIPTION SESSION SUMMARY")
        print("="*60)
        print(f"Total Transcriptions: {summary['total_transcriptions']}")
        print(f"Stable Commits: {summary['total_stable_commits']}")
        print(f"Avg Commit Latency: {summary['avg_commit_latency_ms']:.2f} ms")
        print(f"Final Word Count: {summary['final_word_count']}")
        print(f"Session Duration: {summary['session_duration']:.2f} seconds")
        print(f"Final Text: {summary['final_text']}")
        print("="*60 + "\n")
    
    def reset(self):
        """Reset all logged data"""
        self.__init__()
    
    def enable(self):
        """Enable logging"""
        self.enabled = True
    
    def disable(self):
        """Disable logging (for production use)"""
        self.enabled = False
