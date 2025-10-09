#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Naive Streaming Baseline
Re-transcribes entire buffer each time (no smart buffering)
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
    No stable buffer, no pattern detection, no smart commits
    """
    
    def __init__(self, model_name="base", language="en", processing_interval=1.0):
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
        
        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.process_thread = None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """Processing thread - re-transcribe everything each time"""
        last_process_time = time.time()
        
        while self.is_recording:
            try:
                # Collect audio chunks
                chunk_count = 0
                while not self.audio_queue.empty() and chunk_count < 15:
                    chunk = self.audio_queue.get(block=False)
                    self.audio_buffer = np.append(self.audio_buffer, chunk)
                    chunk_count += 1
                
                # Process at intervals
                current_time = time.time()
                if current_time - last_process_time >= self.processing_interval:
                    if len(self.audio_buffer) > self.RATE * 0.5:
                        # Re-transcribe ENTIRE buffer (naive approach)
                        result = self.model.transcribe(
                            self.audio_buffer,
                            fp16=False,
                            language=self.language
                        )
                        
                        text = result["text"].strip()
                        
                        if text:
                            print(f"\033[91m[NAIVE] {text}\033[0m")
                            self.last_transcription = text
                            
                            if self.logger:
                                self.logger.log_transcription(text, is_stable=False)
                    
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
        
        print("Naive streaming started...")
    
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
            result = self.model.transcribe(
                self.audio_buffer,
                fp16=False,
                language=self.language
            )
            final_text = result["text"].strip()
            
            if self.logger:
                self.logger.log_transcription(final_text, is_stable=True)
            
            print(f"\n[NAIVE FINAL] {final_text}")
    
    def close(self):
        """Clean up"""
        self.stop_streaming()
        self.p.terminate()