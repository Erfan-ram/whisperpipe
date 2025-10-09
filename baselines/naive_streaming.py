#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Naive Streaming Baseline - Fair Real-Time Comparison
Processes every 1 second like WhisperPipe, but re-transcribes ENTIRE buffer each time
Tracks processing time and resource usage for fair comparison
"""

import whisper
import numpy as np
import time
import queue
import threading
import pyaudio

class NaiveStreamingWhisper:
    """
    Baseline: Re-transcribe entire buffer every processing cycle
    
    Key difference from WhisperPipe:
    - WhisperPipe: Only processes NEW audio (dual buffer prevents reprocessing)
    - Naive: Re-processes ALL audio EVERY time (grows linearly with time)
    
    This demonstrates:
    - Higher computational cost (processing time increases over time)
    - Lower efficiency (re-processing stable content)
    - Similar output quality but worse resource usage
    """
    
    def __init__(self, model_name="base", language="en", processing_interval=1.0):
        print(f"Loading Whisper model for Naive Baseline: {model_name}")
        self.model = whisper.load_model(model_name)
        self.language = language
        self.processing_interval = processing_interval
        
        self.audio_buffer = np.array([], dtype=np.float32)
        self.RATE = 16000
        self.CHUNK = 1024
        self.logger = None
        self.is_recording = False
        
        self.audio_queue = queue.Queue()
        self.last_transcription = ""
        
        # Resource tracking
        self.total_processing_time = 0.0
        self.processing_times = []  # Track each processing duration
        self.buffer_sizes = []  # Track buffer size at each processing
        self.transcription_count = 0
        
        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.process_thread = None
        self.session_start_time = None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """
        Processing thread - re-transcribe entire buffer every processing_interval
        This simulates real-time streaming but with naive re-processing
        """
        last_process_time = time.time()
        
        while self.is_recording:
            try:
                # Collect audio chunks (same as WhisperPipe)
                chunk_count = 0
                while not self.audio_queue.empty() and chunk_count < 15:
                    chunk = self.audio_queue.get(block=False)
                    self.audio_buffer = np.append(self.audio_buffer, chunk)
                    chunk_count += 1
                
                # Process at regular intervals (SAME as WhisperPipe)
                current_time = time.time()
                elapsed = current_time - last_process_time
                
                if elapsed >= self.processing_interval:
                    buffer_duration = len(self.audio_buffer) / self.RATE
                    
                    if buffer_duration > 0.5:  # At least 0.5s of audio
                        # KEY DIFFERENCE: Re-transcribe ENTIRE buffer every time
                        process_start = time.time()
                        
                        result = self.model.transcribe(
                            self.audio_buffer,  # ← ENTIRE buffer (grows over time)
                            fp16=False,
                            language=self.language
                        )
                        
                        process_end = time.time()
                        processing_time = process_end - process_start
                        
                        # Track resource usage
                        self.total_processing_time += processing_time
                        self.processing_times.append(processing_time)
                        self.buffer_sizes.append(buffer_duration)
                        self.transcription_count += 1
                        
                        text = result["text"].strip()
                        
                        if text:
                            # Calculate real-time factor
                            rtf = processing_time / buffer_duration if buffer_duration > 0 else 0
                            
                            print(f"\033[91m[NAIVE] Buffer: {buffer_duration:.1f}s | "
                                  f"Process: {processing_time:.2f}s | RTF: {rtf:.2f}x | "
                                  f"Text: {text}\033[0m")
                            
                            self.last_transcription = text
                            
                            if self.logger:
                                self.logger.log_transcription(text, is_stable=False, metadata={
                                    'processing_time': processing_time,
                                    'buffer_duration': buffer_duration,
                                    'rtf': rtf,
                                    'buffer_size_samples': len(self.audio_buffer)
                                })
                    
                    last_process_time = current_time
                
                time.sleep(0.05)
                
            except queue.Empty:
                time.sleep(0.05)
            except Exception as e:
                print(f"Error in naive processing: {e}")
                time.sleep(0.1)
    
    def start_streaming(self):
        """Start audio capture and transcription"""
        self.is_recording = True
        self.audio_buffer = np.array([], dtype=np.float32)
        self.session_start_time = time.time()
        
        # Reset tracking
        self.total_processing_time = 0.0
        self.processing_times = []
        self.buffer_sizes = []
        self.transcription_count = 0
        
        if self.logger:
            self.logger.start_session()
        
        # Open audio stream
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self._audio_callback
        )
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        print(f"Naive streaming started (processes every {self.processing_interval}s)...")
    
    def stop_streaming(self):
        """Stop streaming"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.process_thread:
            self.process_thread.join(timeout=2)
        
        # Final transcription
        if len(self.audio_buffer) > 0:
            process_start = time.time()
            
            result = self.model.transcribe(
                self.audio_buffer,
                fp16=False,
                language=self.language
            )
            
            process_end = time.time()
            processing_time = process_end - process_start
            self.total_processing_time += processing_time
            
            final_text = result["text"].strip()
            
            if self.logger:
                self.logger.log_transcription(final_text, is_stable=True, metadata={
                    'processing_time': processing_time,
                    'is_final': True
                })
            
            print(f"\n[NAIVE FINAL] {final_text}")
        
        # Print resource usage summary
        self._print_resource_summary()
    
    def _print_resource_summary(self):
        """Print resource usage statistics"""
        if self.session_start_time:
            session_duration = time.time() - self.session_start_time
            audio_duration = len(self.audio_buffer) / self.RATE
            
            print(f"\n{'='*60}")
            print("NAIVE BASELINE - RESOURCE USAGE SUMMARY")
            print(f"{'='*60}")
            print(f"Total Audio Duration: {audio_duration:.2f}s")
            print(f"Total Processing Time: {self.total_processing_time:.2f}s")
            print(f"Processing Overhead: {(self.total_processing_time/audio_duration):.2f}x real-time")
            print(f"Transcription Count: {self.transcription_count}")
            print(f"Avg Processing Time: {np.mean(self.processing_times):.2f}s" if self.processing_times else "N/A")
            print(f"Peak Processing Time: {max(self.processing_times):.2f}s" if self.processing_times else "N/A")
            print(f"{'='*60}\n")
    
    def get_resource_stats(self):
        """Get resource usage statistics"""
        audio_duration = len(self.audio_buffer) / self.RATE if len(self.audio_buffer) > 0 else 0
        
        return {
            'total_processing_time': self.total_processing_time,
            'audio_duration': audio_duration,
            'processing_overhead': self.total_processing_time / audio_duration if audio_duration > 0 else 0,
            'transcription_count': self.transcription_count,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'peak_processing_time': max(self.processing_times) if self.processing_times else 0,
            'processing_times': self.processing_times.copy(),
            'buffer_sizes': self.buffer_sizes.copy()
        }
    
    def close(self):
        """Clean up"""
        self.stop_streaming()
        self.p.terminate()