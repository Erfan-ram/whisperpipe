#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Naive Whisper Streaming Implementation (Baseline for Comparison)

This is a baseline implementation that demonstrates the naive approach to 
streaming transcription with Whisper. Unlike WhisperPipe, this implementation:
- Re-transcribes the entire audio buffer on each processing cycle
- Has no dual-buffer architecture
- Has no stability mechanisms or confirmation logic
- Has no content filtering
- Results in unstable outputs and redundant reprocessing

This serves as a comparison baseline to demonstrate WhisperPipe's improvements.
"""

import numpy as np
import pyaudio
import threading
import time
import queue
import whisper
import torch


class NaiveWhisperStream:
    """
    Naive streaming implementation of Whisper that re-transcribes 
    entire audio buffer on each cycle without optimization.
    """
    
    def __init__(self, model_name="base", language="en", processing_interval=1.0, 
                 buffer_duration_seconds=30.0, debug_mode=True):
        """
        Initialize naive Whisper streaming transcriber
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            language: Language code for transcription (e.g., "en", "es", "fr")
            processing_interval: Interval in seconds between processing cycles
            buffer_duration_seconds: Maximum audio buffer duration
            debug_mode: Enable debug logging
        """
        self._debug_mode_enabled = debug_mode
        self.language = language
        print(f"[NAIVE] Loading Whisper model: {model_name}")
        
        try:
            self.model = whisper.load_model(model_name)
            print("[NAIVE] Model loaded successfully!")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[NAIVE] Using device: {self.device}")
            
        except Exception as e:
            print(f"[NAIVE] Error loading model: {e}")
            raise
        
        # Audio parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper expects 16kHz
        self.CHUNK = 1024
        
        # Buffer parameters
        self.buffer_duration_seconds = buffer_duration_seconds
        self.max_buffer_size = int(self.RATE * buffer_duration_seconds)
        self.processing_interval = processing_interval
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_recording = False
        
        # Transcription tracking (for metrics)
        self.transcription_history = []  # Track all transcriptions for edit overhead
        self.last_transcription = ""
        self.edit_count = 0
        self.transcription_count = 0
        
        # PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.process_thread = None
        
        # Metrics tracking
        self.processing_times = []
        self.buffer_sizes = []
        
    def _debug_print(self, message):
        """Print debug message if debug mode enabled"""
        if self._debug_mode_enabled:
            print(f"[NAIVE DEBUG] {message}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio stream"""
        if status:
            self._debug_print(f"PyAudio status: {status}")
        
        try:
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_queue.put(audio_data)
        except Exception as e:
            self._debug_print(f"Error in audio callback: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """
        Naive audio processing: re-transcribes entire buffer each cycle
        This is the inefficient baseline approach.
        """
        while self.is_recording:
            try:
                start_time = time.time()
                
                # Collect audio from queue
                chunk_count = 0
                while not self.audio_queue.empty() and chunk_count < 50:
                    chunk = self.audio_queue.get(block=False)
                    self.audio_buffer = np.append(self.audio_buffer, chunk)
                    chunk_count += 1
                
                # Limit buffer size
                if len(self.audio_buffer) > self.max_buffer_size:
                    self.audio_buffer = self.audio_buffer[-self.max_buffer_size:]
                
                # Process if we have enough audio (at least 1 second)
                if len(self.audio_buffer) >= self.RATE:
                    
                    # NAIVE APPROACH: Re-transcribe ENTIRE buffer every time
                    # This causes exponential growth in processing time
                    buffer_duration = len(self.audio_buffer) / self.RATE
                    self._debug_print(f"Re-transcribing entire buffer: {buffer_duration:.1f}s")
                    
                    # Record buffer size for metrics
                    self.buffer_sizes.append(len(self.audio_buffer))
                    
                    # Transcribe entire buffer
                    transcribe_start = time.time()
                    result = self.model.transcribe(
                        self.audio_buffer,
                        language=self.language,
                        fp16=False
                    )
                    transcribe_time = time.time() - transcribe_start
                    self.processing_times.append(transcribe_time)
                    
                    new_text = result['text'].strip()
                    
                    # Track transcription changes for edit overhead metric
                    if new_text:
                        self.transcription_count += 1
                        
                        # Count edits (word-level changes)
                        if self.last_transcription != new_text:
                            old_words = set(self.last_transcription.split())
                            new_words = set(new_text.split())
                            
                            # Words that changed
                            added = new_words - old_words
                            removed = old_words - new_words
                            self.edit_count += len(added) + len(removed)
                            
                            self._debug_print(f"Edit overhead: +{len(added)} -{len(removed)} words")
                        
                        # Store in history for stability analysis
                        self.transcription_history.append({
                            'text': new_text,
                            'timestamp': time.time(),
                            'buffer_size': len(self.audio_buffer),
                            'processing_time': transcribe_time
                        })
                        
                        # Display output (unstable - changes frequently)
                        print(f"\n[NAIVE OUTPUT] {new_text}")
                        
                        self.last_transcription = new_text
                
                # Sleep until next processing interval
                elapsed = time.time() - start_time
                sleep_time = max(0.01, self.processing_interval - elapsed)
                time.sleep(sleep_time)
                
            except queue.Empty:
                time.sleep(0.05)
            except Exception as e:
                self._debug_print(f"Error in processing: {e}")
                time.sleep(0.1)
    
    def start_streaming(self):
        """Start audio streaming and transcription"""
        if self.is_recording:
            print("[NAIVE] Already recording")
            return False
        
        self.is_recording = True
        self.transcription_history = []
        self.last_transcription = ""
        self.edit_count = 0
        self.transcription_count = 0
        self.processing_times = []
        self.buffer_sizes = []
        
        try:
            # Open audio stream
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self._audio_callback
            )
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self._process_audio)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            print("[NAIVE] Streaming started (Naive Re-transcription Mode)")
            print("[NAIVE] - Re-transcribes entire buffer each cycle")
            print("[NAIVE] - No stability mechanisms")
            print("[NAIVE] - No dual-buffer architecture")
            print("[NAIVE] - High edit overhead expected")
            
            return True
            
        except Exception as e:
            print(f"[NAIVE] Error starting: {e}")
            self.is_recording = False
            return False
    
    def stop_streaming(self):
        """Stop audio streaming"""
        if not self.is_recording:
            return False
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        print("\n[NAIVE] Streaming stopped")
        self._print_metrics()
        
        return True
    
    def _print_metrics(self):
        """Print collected metrics"""
        if not self.transcription_history:
            return
        
        print("\n" + "="*60)
        print("NAIVE IMPLEMENTATION METRICS")
        print("="*60)
        
        # Edit overhead
        final_word_count = len(self.last_transcription.split())
        if final_word_count > 0:
            edit_overhead = self.edit_count / final_word_count
            print(f"Edit Overhead: {edit_overhead:.2f}× ({self.edit_count} edits / {final_word_count} words)")
        
        # Processing time growth
        if len(self.processing_times) > 1:
            avg_processing_time = np.mean(self.processing_times)
            max_processing_time = np.max(self.processing_times)
            print(f"Avg Processing Time: {avg_processing_time:.3f}s")
            print(f"Max Processing Time: {max_processing_time:.3f}s")
            
            # Show processing time growth (linear with buffer size)
            if len(self.buffer_sizes) > 1:
                avg_buffer_size = np.mean(self.buffer_sizes) / self.RATE
                print(f"Avg Buffer Size: {avg_buffer_size:.1f}s")
        
        # Transcription stability (how often text changes)
        if len(self.transcription_history) > 1:
            changes = 0
            for i in range(1, len(self.transcription_history)):
                if self.transcription_history[i]['text'] != self.transcription_history[i-1]['text']:
                    changes += 1
            
            stability = (1 - changes / (len(self.transcription_history) - 1)) * 100
            print(f"Transcription Stability: {stability:.1f}% (lower is worse)")
            print(f"  Total transcriptions: {len(self.transcription_history)}")
            print(f"  Text changes: {changes}")
        
        print("="*60 + "\n")
    
    def get_metrics(self):
        """
        Get metrics for evaluation
        
        Returns:
            dict: Metrics including edit_overhead, stability, processing_times
        """
        final_word_count = len(self.last_transcription.split())
        edit_overhead = self.edit_count / final_word_count if final_word_count > 0 else 0
        
        # Calculate stability
        stability = 0.0
        if len(self.transcription_history) > 1:
            changes = 0
            for i in range(1, len(self.transcription_history)):
                if self.transcription_history[i]['text'] != self.transcription_history[i-1]['text']:
                    changes += 1
            stability = (1 - changes / (len(self.transcription_history) - 1)) * 100
        
        return {
            'edit_overhead': edit_overhead,
            'edit_count': self.edit_count,
            'final_word_count': final_word_count,
            'stability': stability,
            'transcription_count': len(self.transcription_history),
            'processing_times': self.processing_times.copy(),
            'buffer_sizes': [bs / self.RATE for bs in self.buffer_sizes],
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_streaming()
        if self.p:
            self.p.terminate()


if __name__ == "__main__":
    """Simple test of naive implementation"""
    print("Naive Whisper Streaming Test")
    print("This will show unstable, frequently changing output")
    print("Press Ctrl+C to stop\n")
    
    transcriber = NaiveWhisperStream(
        model_name="base",
        language="en",
        processing_interval=1.0,
        debug_mode=True
    )
    
    try:
        transcriber.start_streaming()
        
        # Run for a while
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        transcriber.stop_streaming()
        transcriber.cleanup()
