#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time speech-to-text streaming with pywhispercpp using a local model file
Optimized with sentence-based segmentation to prevent re-processing of audio
Fixed: proper sentence end detection (not catching normal sentences as processing indicators)
"""
import numpy as np
import pyaudio
import threading
import time
import os
import sys
import signal
import queue
import re
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
                model=model_path,
                single_segment=False,
                print_progress=False,
                n_threads=4,
                language="en"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Audio recording parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = constants.WHISPER_SAMPLE_RATE  # 16000 Hz
        self.CHUNK = 1024
        
        # Calculate buffer parameters
        self.buffer_duration_seconds = buffer_duration_seconds
        self.buffer_size = int(self.RATE * buffer_duration_seconds)
        self.overlap_ratio = 0.5
        
        # Processing parameters
        self.audio_queue = queue.Queue()
        self.rolling_buffer = np.array([], dtype=np.float32)
        self.is_recording = False
        
        # Sentence-based segmentation parameters
        self.current_sentence_buffer = np.array([], dtype=np.float32)  # Audio for current sentence
        self.current_sentence_text = ""  # Accumulated text for current sentence
        self.completed_sentences = []  # Store completed sentences
        self.sentence_start_time = None  # Track when current sentence started
        self.max_sentence_duration = 30.0  # Force segmentation after 30 seconds
        self.min_sentence_duration = 1.5   # Minimum duration before allowing segmentation
        
        # Fixed sentence detection patterns
        self.sentence_endings = ['.', '?', '!']
        self.pause_endings = [',', ';', ':']  # Shorter pauses, not full sentence breaks
        
        # Silence detection parameters
        self.silence_threshold = 0.01
        self.silence_frames = 0
        self.silence_frames_threshold = 25
        self.is_speech_active = False
        
        # Speech detection flags
        self.has_detected_speech = False
        self.greeting_keywords = ["hello", "hey", "hi", "ok", "okay"]
        
        # Processing state
        self.last_transcription_time = time.time()
        self.processing_interval = 1.0
        
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
        """Callback function for PyAudio stream"""
        if status:
            print(f"PyAudio status: {status}")
        
        try:
            # Convert audio data to numpy array and normalize
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
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
        """Check if text contains greeting keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.greeting_keywords)
    
    def _is_processing_indicator(self, text):
        """
        Check if text contains processing indicators that should NOT end a sentence
        FIXED: Only catch actual processing indicators, not normal sentences
        """
        if not text:
            return False
        
        text = text.strip()
        
        # Only check for explicit ellipsis patterns at the END of text
        # These are the ONLY patterns that indicate processing/thinking
        processing_patterns = [
            r'\.{3,}\s*$',      # Three or more dots at end: "thinking..."
            r'\s+\.{2,}\s*$',   # Spaced dots at end: "well .."
            r'\.{2,}$',         # Two or more dots at end (but not single period)
        ]
        
        # Check each pattern
        for pattern in processing_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for specific incomplete phrase patterns
        incomplete_patterns = [
            r'\b(um|uh|er|ah)\.{2,}\s*$',     # "um..." at end
            r'\b(and|so|but|well)\.{2,}\s*$', # "and..." at end
            r'\b(you know|i mean)\.{2,}\s*$', # "you know..." at end
        ]
        
        text_lower = text.lower()
        for pattern in incomplete_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _detect_sentence_end(self, text):
        """
        Detect if text contains sentence-ending punctuation
        Returns: (is_sentence_end, is_pause_point)
        FIXED: Better logic for detecting real sentence endings
        """
        if not text:
            return False, False
        
        text = text.strip()
        
        # First check if this is a processing indicator - if so, don't end sentence
        if self._is_processing_indicator(text):
            print(f"[DEBUG] Processing indicator detected: '{text[-15:]}' - NOT ending sentence")
            return False, False
        
        # Check for real sentence endings
        has_sentence_end = False
        has_pause = False
        
        # Look for sentence endings in the last few characters
        for ending in self.sentence_endings:
            if text.endswith(ending) or text.endswith(ending + ' '):
                has_sentence_end = True
                print(f"[DEBUG] Sentence end detected with '{ending}': '{text[-15:]}'")
                break
        
        # Check for pause points (commas, etc.)
        for ending in self.pause_endings:
            if text.endswith(ending) or text.endswith(ending + ' '):
                has_pause = True
                break
        
        return has_sentence_end, has_pause
    
    def _should_force_segmentation(self):
        """Check if we should force segmentation due to time limits"""
        if self.sentence_start_time is None:
            return False
        
        duration = time.time() - self.sentence_start_time
        return duration > self.max_sentence_duration
    
    def _can_segment(self):
        """Check if enough time has passed to allow segmentation"""
        if self.sentence_start_time is None:
            return True
        
        duration = time.time() - self.sentence_start_time
        return duration > self.min_sentence_duration
    
    def _finalize_sentence(self, final_text=None):
        """
        Finalize the current sentence and reset buffers
        """
        if final_text:
            self.current_sentence_text = final_text
        
        if self.current_sentence_text and len(self.current_sentence_text.strip()) > 3:
            # Clean up the sentence text (remove processing indicators)
            cleaned_text = self.current_sentence_text.strip()
            
            # Remove trailing ellipsis or incomplete processing indicators
            cleaned_text = re.sub(r'\.{2,}\s*$', '', cleaned_text)  # Remove trailing ellipsis
            cleaned_text = re.sub(r'\s+(um|uh|and|so|but|well)\.{2,}\s*$', '', cleaned_text, flags=re.IGNORECASE)
            
            if len(cleaned_text.strip()) > 3:  # Only process if we still have meaningful content
                # Store completed sentence
                sentence_data = {
                    'text': cleaned_text.strip(),
                    'timestamp': time.strftime('%H:%M:%S'),
                    'duration': time.time() - self.sentence_start_time if self.sentence_start_time else 0
                }
                self.completed_sentences.append(sentence_data)
                
                print(f"\n[SENTENCE COMPLETE] {sentence_data['text']}")
                
                # Send to LLM immediately
                self._send_to_llm(sentence_data['text'])
        
        # Reset for next sentence
        self.current_sentence_buffer = np.array([], dtype=np.float32)
        self.current_sentence_text = ""
        self.sentence_start_time = None
    
    def _process_audio(self):
        """Process audio with sentence-based segmentation"""
        while self.is_recording:
            try:
                chunk_count = 0
                start_time = time.time()
                
                # Collect audio chunks
                while not self.audio_queue.empty() and chunk_count < 15:
                    chunk = self.audio_queue.get(block=False)
                    
                    with self.lock:
                        self.rolling_buffer = np.append(self.rolling_buffer, chunk)
                    
                    chunk_count += 1
                    
                    # Silence detection
                    if self._is_silent(chunk):
                        self.silence_frames += 1
                    else:
                        self.silence_frames = 0
                        if not self.is_speech_active:
                            print("\n[SPEECH DETECTED]")
                            self.is_speech_active = True
                            # Start new sentence timing if not already started
                            if self.sentence_start_time is None:
                                self.sentence_start_time = time.time()
                
                # Maintain rolling buffer size
                with self.lock:
                    if len(self.rolling_buffer) > self.buffer_size:
                        self.rolling_buffer = self.rolling_buffer[-self.buffer_size:]
                
                # Add to current sentence buffer during speech
                current_time = time.time()
                if self.is_speech_active and chunk_count > 0:
                    with self.lock:
                        # Fixed: Ensure we use integer indices for slicing
                        chunk_samples = int(self.CHUNK * chunk_count)
                        if len(self.rolling_buffer) >= chunk_samples:
                            latest_audio = np.copy(self.rolling_buffer[-chunk_samples:])
                        else:
                            latest_audio = np.copy(self.rolling_buffer)
                    
                    if len(latest_audio) > 0:
                        self.current_sentence_buffer = np.append(self.current_sentence_buffer, latest_audio)
                        
                        # Limit sentence buffer size (max 30 seconds)
                        max_sentence_buffer = int(self.RATE * self.max_sentence_duration)
                        if len(self.current_sentence_buffer) > max_sentence_buffer:
                            self.current_sentence_buffer = self.current_sentence_buffer[-max_sentence_buffer:]
                
                # Check for silence (potential sentence end)
                if self.is_speech_active and self.silence_frames >= self.silence_frames_threshold:
                    print("\n[SILENCE DETECTED]")
                    self.is_speech_active = False
                    
                    # Process current sentence buffer if we have content
                    min_buffer_size = int(self.RATE * 0.5)
                    if len(self.current_sentence_buffer) > min_buffer_size:
                        self._process_sentence_segment(self.current_sentence_buffer, is_silence_triggered=True)
                    
                    self.silence_frames = 0
                
                # Process during ongoing speech (periodic transcription)
                min_processing_buffer = int(self.RATE * 0.8)
                should_process = (
                    self.is_speech_active and 
                    (current_time - self.last_transcription_time >= self.processing_interval) and 
                    len(self.current_sentence_buffer) > min_processing_buffer
                )
                
                # Force segmentation if sentence is too long
                should_force = self._should_force_segmentation()
                
                if should_process or should_force:
                    if should_force:
                        print("\n[FORCE SEGMENTATION - Time limit reached]")
                    
                    self._process_sentence_segment(
                        self.current_sentence_buffer, 
                        is_silence_triggered=should_force
                    )
                    self.last_transcription_time = time.time()
                
                # Adaptive sleep
                elapsed = time.time() - start_time
                sleep_time = max(0.01, 0.05 - elapsed)
                time.sleep(sleep_time)
                
            except queue.Empty:
                time.sleep(0.05)
            except Exception as e:
                print(f"Error in audio processing thread: {e}")
                time.sleep(0.1)
    
    def _process_sentence_segment(self, audio_buffer, is_silence_triggered=False):
        """
        Process a sentence segment with proper sentence end detection
        """
        min_segment_size = int(self.RATE * 0.3)
        if len(audio_buffer) < min_segment_size:
            return
        
        try:
            # Transcribe the current sentence buffer
            segments = self.model.transcribe(audio_buffer)
            
            full_transcript = ""
            for segment in segments:
                segment_text = segment.text.strip()
                
                if "[BLANK_AUDIO]" in segment_text:
                    continue
                
                if segment_text:
                    if not self.has_detected_speech and self._contains_greeting(segment_text):
                        print(f"\n[GREETING DETECTED]: {segment_text}")
                        self.has_detected_speech = True
                    
                    if self.has_detected_speech:
                        if full_transcript:
                            full_transcript += " " + segment_text
                        else:
                            full_transcript = segment_text
            
            if not full_transcript or not self.has_detected_speech:
                return
            
            print(f"\n[TRANSCRIPTION] {full_transcript}")
            
            # Update current sentence text
            if self.current_sentence_text:
                # Check for overlap and append only new content
                words_current = self.current_sentence_text.split()
                words_new = full_transcript.split()
                
                # Simple overlap detection
                overlap_found = False
                if len(words_current) >= 2 and len(words_new) >= 2:
                    last_words = ' '.join(words_current[-2:])
                    if last_words in full_transcript:
                        # Find where overlap ends and append new content
                        overlap_pos = full_transcript.find(last_words) + len(last_words)
                        new_content = full_transcript[overlap_pos:].strip()
                        if new_content:
                            self.current_sentence_text += " " + new_content
                            overlap_found = True
                
                if not overlap_found:
                    # No clear overlap, append with space
                    self.current_sentence_text += " " + full_transcript
            else:
                self.current_sentence_text = full_transcript
            
            # Check for sentence ending with improved logic
            is_sentence_end, is_pause = self._detect_sentence_end(self.current_sentence_text)
            
            # Decide whether to finalize sentence
            should_finalize = False
            
            if is_sentence_end and self._can_segment():
                should_finalize = True
                print(f"[SENTENCE END DETECTED] via punctuation")
            elif is_silence_triggered and self._can_segment() and not self._is_processing_indicator(self.current_sentence_text):
                should_finalize = True
                print(f"[SENTENCE END DETECTED] via silence")
            elif self._should_force_segmentation():
                should_finalize = True
                print(f"[SENTENCE END DETECTED] via time limit")
            
            # Finalize sentence if conditions are met
            if should_finalize:
                self._finalize_sentence()
            
        except Exception as e:
            print(f"Error in sentence segment processing: {e}")
    
    def _send_to_llm(self, text):
        """Send completed sentence to LLM for processing"""
        print(f"\n[LLM INPUT]: {text}")
        # Your LLM integration code here
        # Example: response = your_llm_model.generate_answer(text)
        # print(f"[LLM RESPONSE]: {response}")
    
    def start_streaming(self):
        """Start streaming from microphone and transcribing"""
        self.is_recording = True
        self.rolling_buffer = np.array([], dtype=np.float32)
        self.current_sentence_buffer = np.array([], dtype=np.float32)
        self.current_sentence_text = ""
        self.completed_sentences = []
        self.sentence_start_time = None
        self.audio_queue = queue.Queue()
        self.silence_frames = 0
        self.is_speech_active = False
        self.has_detected_speech = False
        self.last_transcription_time = time.time()
        
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
            
            print("Listening with sentence-based segmentation...")
            print("Say a greeting (hello, hey, hi) to begin processing...")
            print("Sentences will be processed individually - no re-processing!")
            print("Only actual ellipsis (...) will prevent sentence ending!")
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
        
        # Process any remaining audio in current sentence buffer
        if len(self.current_sentence_buffer) > 0 and self.has_detected_speech:
            try:
                print("\n[PROCESSING FINAL SEGMENT]")
                self._process_sentence_segment(self.current_sentence_buffer, is_silence_triggered=True)
            except Exception as e:
                print(f"Error processing final segment: {e}")
        
        # Wait for processing thread to finish
        if self.process_thread and self.process_thread.is_alive():
            try:
                self.process_thread.join(timeout=2)
            except Exception as e:
                print(f"Error joining processing thread: {e}")
        
        # Print summary
        if self.completed_sentences:
            print(f"\n[SESSION SUMMARY] Processed {len(self.completed_sentences)} sentences:")
            for i, sentence in enumerate(self.completed_sentences, 1):
                print(f"  {i}. [{sentence['timestamp']}] {sentence['text']}")
    
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
    
    # Model path - update this to your model location
    MODEL_PATH = "/home/erfan/Desktop/Rag-zone/whisperacpp/whisper.cpp/models/ggml-base.en.bin"
    
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
                return False
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
                listener.join()
        except Exception as e:
            print(f"Error with keyboard listener: {e}")
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("\nCleaning up resources...")
        try:
            transcriber.stop_streaming()
            transcriber.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        print("Transcription ended.")


if __name__ == "__main__":
    main()