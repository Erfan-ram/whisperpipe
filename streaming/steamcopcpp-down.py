#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time speech-to-text streaming with pywhispercpp
"""
import numpy as np
import pyaudio
import threading
import time
from pynput import keyboard
from pywhispercpp.model import Model
import pywhispercpp.constants as constants

class WhisperCppStreamingTranscriber:
    def __init__(self, model_name="tiny.en"):
        """
        Initialize the transcriber with pywhispercpp
        
        Args:
            model_name: Model name (tiny.en, base.en, small.en, medium.en)
                        The smaller models are faster but less accurate.
        """
        print(f"Loading Whisper.cpp model: {model_name}...")
        
        # Initialize the pywhispercpp model
        # single_segment=True ensures better streaming behavior
        # n_threads=4 provides good performance on CPU
        self.model = Model(model_name, 
                          single_segment=True, 
                          print_progress=False,
                          n_threads=4)
        
        print("Model loaded!")
        
        # Audio recording parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = constants.WHISPER_SAMPLE_RATE  # Always use Whisper's sample rate (16000)
        self.CHUNK = 1024
        
        # Processing parameters 
        self.audio_buffer = []
        self.is_recording = False
        self.last_processed = 0
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.audio_buffer.append(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """Process audio in the background"""
        while self.is_recording:
            # If we have enough new audio data to process (around 2 seconds)
            if len(self.audio_buffer) > self.last_processed + 30:
                # Combine new chunks
                new_chunks = self.audio_buffer[self.last_processed:self.last_processed + 30]
                audio_data = np.hstack(new_chunks).astype(np.float32) / 32768.0  # Convert to float32
                
                try:
                    # Process with WhisperCpp
                    segments = self.model.transcribe(audio_data)
                    
                    # Print transcribed segments
                    for segment in segments:
                        if segment.text.strip():  # Only print if there's actual text
                            print(f"\r{segment.text.strip()}")
                    
                except Exception as e:
                    print(f"Error in transcription: {e}")
                
                # Update the last processed index
                self.last_processed += 30
            
            # Sleep a bit to avoid using too much CPU
            time.sleep(0.5)
    
    def start_streaming(self):
        """Start streaming from microphone and transcribing"""
        self.is_recording = True
        self.audio_buffer = []
        self.last_processed = 0
        
        # Open PyAudio stream
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
        
        print("Listening... (Press 'q' to stop)")
    
    def stop_streaming(self):
        """Stop streaming and clean up"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Stop and close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Process any remaining audio
        if len(self.audio_buffer) > self.last_processed:
            remaining_chunks = self.audio_buffer[self.last_processed:]
            if remaining_chunks:
                audio_data = np.hstack(remaining_chunks).astype(np.float32) / 32768.0
                segments = self.model.transcribe(audio_data)
                for segment in segments:
                    if segment.text.strip():
                        print(f"\r{segment.text.strip()}")
        
        # Wait for the processing thread to finish
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)
    
    def close(self):
        """Clean up resources"""
        self.stop_streaming()
        self.p.terminate()


def main():
    # Available models: tiny.en, base.en, small.en, medium.en
    # For CPU-only usage, tiny.en or base.en are recommended
    transcriber = WhisperCppStreamingTranscriber(model_name="base.en")
    
    def on_press(key):
        try:
            if key.char == 'q':
                print("\nStopping transcription...")
                return False  # Stop listener
        except AttributeError:
            pass
    
    try:
        # Start the transcriber
        transcriber.start_streaming()
        
        # Set up keyboard listener
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()  # Wait for 'q' press
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Clean up
        transcriber.stop_streaming()
        transcriber.close()
        print("Transcription ended.")


if __name__ == "__main__":
    main()