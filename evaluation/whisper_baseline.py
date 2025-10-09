#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline Whisper streaming implementation for comparison
This is a simple sliding window approach without the enhancements:
- No dual-buffer architecture
- No similarity-based prefix stabilization
- No noise/foreign-language rejection
"""
import numpy as np
import whisper
import torch
import time
from typing import List, Tuple, Optional


class WhisperBaseline:
    """Baseline Whisper streaming with simple sliding window"""
    
    def __init__(self, model_name="base", language="en", window_duration=5.0):
        """
        Initialize baseline Whisper model
        
        Args:
            model_name: Whisper model name
            language: Language code
            window_duration: Sliding window duration in seconds
        """
        print(f"Loading Whisper baseline model: {model_name}")
        self.model = whisper.load_model(model_name)
        self.language = language
        self.window_duration = window_duration
        self.RATE = 16000  # Whisper expects 16kHz
        
        # Check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Simple buffer for audio
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Track outputs
        self.intermediate_outputs = []  # Store all intermediate transcriptions
        self.final_output = ""
        
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Tuple[str, float]:
        """
        Process an audio chunk and return transcription
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            Tuple of (transcription, processing_time)
        """
        start_time = time.time()
        
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Keep only the last window_duration seconds
        max_samples = int(self.window_duration * self.RATE)
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]
        
        # Transcribe current buffer
        if len(self.audio_buffer) > self.RATE * 0.5:  # At least 0.5s of audio
            try:
                result = self.model.transcribe(
                    self.audio_buffer,
                    language=self.language,
                    fp16=False if self.device == "cpu" else True,
                    task="transcribe"
                )
                transcription = result.get("text", "").strip()
            except Exception as e:
                print(f"Error in transcription: {e}")
                transcription = ""
        else:
            transcription = ""
        
        processing_time = time.time() - start_time
        
        # Store intermediate output
        self.intermediate_outputs.append({
            'timestamp': time.time(),
            'text': transcription,
            'processing_time': processing_time
        })
        
        return transcription, processing_time
    
    def finalize(self) -> str:
        """
        Finalize and return the last transcription
        
        Returns:
            Final transcription text
        """
        if self.intermediate_outputs:
            self.final_output = self.intermediate_outputs[-1]['text']
        return self.final_output
    
    def get_intermediate_outputs(self) -> List[dict]:
        """Return all intermediate outputs for stability analysis"""
        return self.intermediate_outputs
    
    def reset(self):
        """Reset the baseline model state"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.intermediate_outputs = []
        self.final_output = ""
