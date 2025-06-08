#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time speech-to-text streaming with pywhispercpp using a local model file
Optimized for real-time LLM integration with efficient silence detection
and final transcript selection
"""
import numpy as np
import pyaudio
import threading
import time
import os
import sys
import signal
import queue
from collections import deque
from pynput import keyboard
from pywhispercpp.model import Model
import pywhispercpp.constants as constants

class WhisperCppStreamingTranscriber:
    def __init__(self, model_path, buffer_duration_seconds=5):
        """
        Initialize the transcriber with a local pywhispercpp model
        
        Args:
            model_path: Full path to your local ggml model file
            buffer_duration_seconds: Time window in seconds to hold audio for processing
        """
        # Verify the model file exists
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print(f"Loading Whisper.cpp model from: {model_path}")
        
        try:
            # Initialize the pywhispercpp model with your local model file
            self.model = Model(
                model=model_path,  # Directly pass the model file path
                single_segment=False,  # Allow multiple segments for better context
                print_progress=False,
                n_threads=4,
                language="en"  # Force English for better accuracy
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Audio recording parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = constants.WHISPER_SAMPLE_RATE  # Always use Whisper's sample rate (16000)
        self.CHUNK = 1024
        
        # Calculate buffer parameters
        self.buffer_duration_seconds = buffer_duration_seconds
        self.buffer_size = int(self.RATE * buffer_duration_seconds)
        self.overlap_ratio = 0.5  # 50% overlap for continuous transcription
        
        # Processing parameters
        self.audio_queue = queue.Queue()
        self.rolling_buffer = np.array([], dtype=np.float32)
        self.is_recording = False
        
        # Transcript handling
        self.latest_full_transcript = ""  # Store the most recent complete transcript
        self.transcription_history = deque(maxlen=5)  # Store last few transcriptions
        
        # Silence detection parameters
        self.silence_threshold = 0.01  # Threshold for silence detection
        self.silence_frames = 0
        self.silence_frames_threshold = 30  # About 1 second of silence
        self.is_speech_active = False  # Track if speech is currently active
        
        # Speech detection flags
        self.has_detected_speech = False  # Flag to track if we've detected meaningful speech
        self.greeting_keywords = ["hello", "hey", "hi", "ok", "okay"]  # Keywords to trigger processing
        
        # Threading and synchronization
        self.process_thread = None
        self.lock = threading.Lock()
        
        # Initialize PyAudio
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            print(f"Error initializing PyAudio: {e}")
            sys.exit(1)
        self.stream = None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream - optimized for low latency"""
        if status:
            print(f"PyAudio status: {status}")
        
        try:
            # Convert audio data to numpy array and normalize to float32
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to processing queue
            self.audio_queue.put(audio_data)
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def _is_silent(self, audio_data, threshold=None):
        """Check if audio segment is silent"""
        if threshold is None:
            threshold = self.silence_threshold
        return np.abs(audio_data).mean() < threshold
    
    def _contains_greeting(self, text):
        """Check if text contains greeting keywords to trigger speech detection"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.greeting_keywords)
    
    def _process_audio(self):
        """Process audio in the background - optimized for real-time performance with silence detection"""
        last_transcription_time = time.time()
        processing_interval = 0.3  # Process every 300ms for low latency
        continuous_speech_buffer = np.array([], dtype=np.float32)  # Buffer for active speech
        
        while self.is_recording:
            try:
                chunk_count = 0
                start_time = time.time()
                
                # Collect audio chunks until processing interval is reached
                while not self.audio_queue.empty() and chunk_count < 20:  # Limit chunks per processing cycle
                    chunk = self.audio_queue.get(block=False)
                    with self.lock:
                        self.rolling_buffer = np.append(self.rolling_buffer, chunk)
                    chunk_count += 1
                    
                    # Check for silence in this chunk
                    if self._is_silent(chunk):
                        self.silence_frames += 1
                    else:
                        self.silence_frames = 0
                        if not self.is_speech_active:
                            print("\n[SPEECH DETECTED]")
                            self.is_speech_active = True
                            # Start fresh with a new buffer when speech begins
                            continuous_speech_buffer = np.array([], dtype=np.float32)
                
                # Maintain buffer size to prevent memory growth
                with self.lock:
                    if len(self.rolling_buffer) > self.buffer_size:
                        # Keep buffer at fixed size, dropping oldest data
                        overlap_samples = int(self.buffer_size * self.overlap_ratio)
                        self.rolling_buffer = self.rolling_buffer[-self.buffer_size:]
                
                # If speech is active, add to continuous speech buffer
                current_time = time.time()
                if self.is_speech_active:
                    with self.lock:
                        # Get the latest audio data
                        latest_audio = np.copy(self.rolling_buffer[-self.CHUNK*chunk_count:]) if chunk_count > 0 else np.array([], dtype=np.float32)
                    
                    if len(latest_audio) > 0:
                        continuous_speech_buffer = np.append(continuous_speech_buffer, latest_audio)
                        
                        # Limit continuous buffer size but allow for longer utterances
                        max_continuous_buffer = self.RATE * 15  # 15 seconds max (increased from 10)
                        if len(continuous_speech_buffer) > max_continuous_buffer:
                            continuous_speech_buffer = continuous_speech_buffer[-max_continuous_buffer:]
                
                # Check if speech has stopped
                if self.is_speech_active and self.silence_frames >= self.silence_frames_threshold:
                    print("\n[SILENCE DETECTED]")
                    self.is_speech_active = False
                    
                    # Process the complete speech segment if we have enough data
                    if len(continuous_speech_buffer) > self.RATE * 0.5:
                        self._process_speech_segment(continuous_speech_buffer, is_final=True)
                        continuous_speech_buffer = np.array([], dtype=np.float32)
                    
                    # Reset silence counter
                    self.silence_frames = 0
                
                # Determine if we should process during ongoing speech
                should_process = (self.is_speech_active and 
                                 (current_time - last_transcription_time >= processing_interval) and 
                                 len(continuous_speech_buffer) > self.RATE * 0.75)  # Need enough speech to process
                
                # Process active speech periodically for visual feedback only
                if should_process:
                    self._process_speech_segment(continuous_speech_buffer, is_final=False)
                    last_transcription_time = time.time()
                
                # Adaptive sleep based on processing load
                elapsed = time.time() - start_time
                sleep_time = max(0.01, processing_interval - elapsed)  # At least 10ms
                time.sleep(sleep_time)
                
            except queue.Empty:
                time.sleep(0.05)  # Nothing in queue, sleep briefly
            except Exception as e:
                print(f"Error in audio processing thread: {e}")
                time.sleep(0.1)  # Wait before trying again
    
    def _process_speech_segment(self, audio_buffer, is_final=False):
        """Process a segment of speech audio"""
        try:
            # Process with WhisperCpp
            segments = self.model.transcribe(audio_buffer)
            
            has_blank_audio = False
            full_transcript = ""
            
            for segment in segments:
                segment_text = segment.text.strip()
                
                # Check for [BLANK_AUDIO] markers
                if "[BLANK_AUDIO]" in segment_text:
                    has_blank_audio = True
                    continue
                
                # Process meaningful speech
                if segment_text:
                    # Check for greeting words to initiate processing if not already started
                    if not self.has_detected_speech and self._contains_greeting(segment_text):
                        print(f"\n[GREETING DETECTED]: {segment_text}")
                        self.has_detected_speech = True
                    
                    # Only add text if we've already detected meaningful speech
                    if self.has_detected_speech:
                        # Avoid adding duplicate content
                        if full_transcript:
                            full_transcript += " " + segment_text
                        else:
                            full_transcript = segment_text
            
            # Process the transcript if we have something
            if full_transcript and self.has_detected_speech:
                # Avoid re-transcribing the same content
                is_new = not any(self._text_similarity(full_transcript, prev) > 0.9 for prev in self.transcription_history)
                
                if is_new or is_final:
                    # Print current transcript for visual feedback
                    print(f"\n[{time.strftime('%H:%M:%S')}] {full_transcript}")
                    
                    # Update the latest full transcript - this will be what we use for LLM
                    self.latest_full_transcript = full_transcript
                    
                    # Store in history
                    if is_new:
                        self.transcription_history.append(full_transcript)
            
            # If this is the final segment (silence detected) and we have a transcript
            if is_final and self.latest_full_transcript:
                print(f"\n[SENTENCE COMPLETE] - Silence detected")
                self._send_to_llm(self.latest_full_transcript)
                self.latest_full_transcript = ""
                
        except Exception as e:
            print(f"Error in transcription: {e}")

    def _text_similarity(self, text1, text2):
        """Simple similarity metric to avoid duplicate transcriptions"""
        # Simple Jaccard similarity of words
        if not text1 or not text2:
            return 0
            
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0
    
    def _send_to_llm(self, text):
        """Send completed text to LLM for processing"""
        # This is where you would integrate with your LLM Q&A system
        print(f"\n[LLM INPUT]: {text.strip()}")
        # Example: response = your_llm_model.generate_answer(text)
        # print(f"[LLM RESPONSE]: {response}")
    
    def start_streaming(self):
        """Start streaming from microphone and transcribing"""
        self.is_recording = True
        self.rolling_buffer = np.array([], dtype=np.float32)
        self.audio_queue = queue.Queue()
        self.transcription_history.clear()
        self.latest_full_transcript = ""
        self.silence_frames = 0
        self.is_speech_active = False
        self.has_detected_speech = False
        
        # Open PyAudio stream
        try:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self._audio_callback
            )
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.is_recording = False
            return False
        
        # Start processing thread
        try:
            self.process_thread = threading.Thread(target=self._process_audio)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            print("Listening... (Press 'q' to stop)")
            print("Say a greeting (hello, hey, hi) to begin processing...")
            return True
        except Exception as e:
            print(f"Error starting processing thread: {e}")
            self.is_recording = False
            if self.stream:
                self.stream.close()
            return False
    
    def stop_streaming(self):
        """Stop streaming and clean up"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Stop and close stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"Error stopping stream: {e}")
        
        # Process any remaining audio
        try:
            with self.lock:
                if len(self.rolling_buffer) > 0 and self.has_detected_speech:
                    segments = self.model.transcribe(self.rolling_buffer)
                    new_text = []
                    for segment in segments:
                        segment_text = segment.text.strip()
                        if segment_text and "[BLANK_AUDIO]" not in segment_text:
                            new_text.append(segment_text)
                    
                    if new_text:
                        final_text = " ".join(new_text)
                        print(f"\nFinal segment: {final_text}")
                        self.latest_full_transcript = final_text
        except Exception as e:
            print(f"Error processing final audio: {e}")
        
        # Wait for the processing thread to finish
        if self.process_thread and self.process_thread.is_alive():
            try:
                self.process_thread.join(timeout=2)
            except Exception as e:
                print(f"Error joining processing thread: {e}")
                
        # Send any remaining text
        if self.latest_full_transcript and len(self.latest_full_transcript) > 5:
            print(f"\n[FINAL SENTENCE]: {self.latest_full_transcript}")
            self._send_to_llm(self.latest_full_transcript)
    
    def close(self):
        """Clean up resources"""
        self.stop_streaming()
        
        try:
            self.p.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")


def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\nInterrupted by signal. Stopping...")
    if 'transcriber' in globals():
        transcriber.stop_streaming()
        transcriber.close()
    print("Transcription ended.")
    sys.exit(0)


def main():
    # Register signal handlers for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # IMPORTANT: Replace this with the actual path to your local ggml model file
    MODEL_PATH = "/home/flayo/Desktop/flayo -zone/whisper.cpp/models/ggml-base.en.bin"
    
    # Ask for model path if not provided as a constant
    if not os.path.isfile(MODEL_PATH):
        MODEL_PATH = input("Enter the full path to your ggml model file: ")
    
    try:
        # Initialize the transcriber
        transcriber = WhisperCppStreamingTranscriber(model_path=MODEL_PATH)
    except Exception as e:
        print(f"Failed to initialize transcriber: {e}")
        sys.exit(1)
    
    def on_press(key):
        try:
            if key.char == 'q':
                print("\nStopping transcription...")
                return False  # Stop listener
        except AttributeError:
            pass
        except Exception as e:
            print(f"Error in keyboard handling: {e}")
    
    try:
        # Start the transcriber
        if not transcriber.start_streaming():
            print("Failed to start streaming")
            transcriber.close()
            sys.exit(1)
        
        # Set up keyboard listener
        try:
            with keyboard.Listener(on_press=on_press) as listener:
                listener.join()  # Wait for 'q' press
        except Exception as e:
            print(f"Error with keyboard listener: {e}")
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up
        print("\nCleaning up resources...")
        try:
            transcriber.stop_streaming()
            transcriber.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        print("Transcription ended.")


if __name__ == "__main__":
    main()