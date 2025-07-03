#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time speech-to-text streaming with pywhispercpp using a local model file
Optimized with sentence-based segmentation to prevent re-processing of audio
Final Fix: Use last transcription only + proper timer reset logic
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
    def __init__(self, model_path, buffer_duration_seconds=2):
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
        self.completed_sentences = []  # Store completed sentences
        self.sentence_start_time = None  # Track when current sentence started
        self.max_sentence_duration = 60.0  # Max sentence duration
        self.min_sentence_duration = 1.5   # Minimum duration before allowing segmentation
        
        # SIMPLIFIED: Timer and transcription logic
        self.last_transcription = ""           # Always keep the LAST transcription only
        self.waiting_for_continuation = False  # Flag if we're waiting after punctuation
        self.finalization_delay = 5.0         # Wait 5 seconds after punctuation
        self.punctuation_detected_time = None # When we detected punctuation
        self.last_word_count = 0              # Track word count to detect new words
        
        # Fixed sentence detection patterns
        self.sentence_endings = ['.', '?', '!']
        self.pause_endings = [',', ';', ':']  # Shorter pauses, not full sentence breaks
        
        # Silence detection parameters
        self.silence_threshold = 0.01
        self.silence_frames = 0
        self.silence_frames_threshold = 25
        self.is_speech_active = False
        
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
    
    def _starts_with_noise_annotation(self, text):
        """
        Check if text starts with a noise annotation like (wind blowing), (silence), etc.
        Only checks the beginning of the text - noise annotations later in text are allowed.
        """
        if not text:
            return False
        
        text = text.strip()
        
        # Check if text starts with parentheses (likely noise annotation)
        if text.startswith('('):
            # Extract the first parenthetical expression
            match = re.match(r'^\s*\(([^)]+)\)', text)
            if match:
                annotation_content = match.group(1).lower()
                
                # Common noise annotation keywords
                noise_keywords = [
                    'wind', 'blowing', 'silence', 'noise', 'static', 'music', 
                    'breathing', 'rustling', 'door', 'footstep', 'car', 'phone', 
                    'typing', 'clicking', 'mumbl', 'unintellig', 'inaudib', 
                    'cough', 'sneez', 'laugh', 'cry', 'applause', 'sound',
                    'background', 'ambient', 'hum', 'buzz'
                ]
                
                # Check if the annotation contains noise-related keywords
                if any(keyword in annotation_content for keyword in noise_keywords):
                    print(f"[NOISE START DETECTED] Rejecting sentence starting with: '{match.group(0)}'")
                    return True
        
        return False
    
    def _is_noise_only_transcription(self, text):
        """
        Check if transcription contains only noise annotations (no real speech)
        """
        if not text:
            return True
        
        # Remove all parenthetical expressions (noise annotations)
        cleaned = re.sub(r'\([^)]+\)', '', text).strip()
        
        # If nothing left after removing noise annotations, it's noise-only
        return len(cleaned) < 3
    
    def _count_meaningful_words(self, text):
        """
        Count meaningful words in text (excluding noise annotations)
        """
        if not text:
            return 0
        
        # Remove noise annotations
        cleaned = re.sub(r'\([^)]+\)', '', text).strip()
        
        if not cleaned:
            return 0
        
        # Count words
        words = cleaned.split()
        return len(words)
    
    def _has_new_words(self, current_transcription):
        """
        Check if current transcription has new words compared to tracked word count
        """
        current_word_count = self._count_meaningful_words(current_transcription)
        
        if current_word_count > self.last_word_count:
            print(f"[NEW WORDS DETECTED] Word count: {self.last_word_count} → {current_word_count}")
            self.last_word_count = current_word_count
            return True
        
        return False
    
    def _is_processing_indicator(self, text):
        """
        Check if text contains processing indicators that should NOT end a sentence
        """
        if not text:
            return False
        
        text = text.strip()
        
        # Only check for explicit ellipsis patterns at the END of text
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
    
    def _should_finalize_after_delay(self):
        """
        Check if we should finalize after 5-second delay
        Only returns True if:
        1. We're waiting for continuation after punctuation
        2. 5 seconds have passed
        3. No new words were detected
        """
        if not self.waiting_for_continuation or self.punctuation_detected_time is None:
            return False
        
        current_time = time.time()
        delay_elapsed = current_time - self.punctuation_detected_time
        
        # print(f"----------------------------------------------- ")
        # print(f"[DEBUG] Delay elapsed: {delay_elapsed:.2f}s since punctuation detected")
        # print(f"[DEBUG] Waiting for continuation: {self.waiting_for_continuation}, "
        #       f"Punctuation time: {self.punctuation_detected_time}, " 
        #       f"Finalization delay: {self.finalization_delay}s")
        
        
        # Check if delay period has passed
        if delay_elapsed >= self.finalization_delay:
            print(f"[TIMER EXPIRED] {self.finalization_delay}s passed, finalizing sentence")
            return True
        
        return False
    
    def _destroy_timer(self):
        """
        Destroy the current timer and reset waiting state
        """
        if self.waiting_for_continuation:
            print(f"[TIMER DESTROYED] New words detected, cancelling timer")
            self.waiting_for_continuation = False
            self.punctuation_detected_time = None
    
    def _finalize_sentence(self, final_text=None):
        """
        Finalize the current sentence and reset buffers
        Use the last transcription as the final sentence
        """
        sentence_text = final_text if final_text else self.last_transcription
        
        if sentence_text and len(sentence_text.strip()) > 3:
            # Clean up the sentence text (remove processing indicators from end only)
            cleaned_text = sentence_text.strip()
            
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
        self.last_transcription = ""
        self.sentence_start_time = None
        self.waiting_for_continuation = False
        self.punctuation_detected_time = None
        self.last_word_count = 0
    
    def _process_audio(self):
        """Process audio with sentence-based segmentation and proper timer logic"""
        while self.is_recording:
            try:
                chunk_count = 0
                start_time = time.time()
                
                # Check if we should finalize after delay period
                if self._should_finalize_after_delay():
                    self._finalize_sentence()
                
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
                        
                        # Limit sentence buffer size (max 60 seconds)
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
                
                # Force segmentation if sentence is too long (but only if not waiting for continuation)
                should_force = self._should_force_segmentation() and not self.waiting_for_continuation
                
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
        Process a sentence segment with simplified logic:
        - Always replace last_transcription with new transcription
        - Check for new words to destroy timer
        - Check for ending punctuation to start timer
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
                    if full_transcript:
                        full_transcript += " " + segment_text
                    else:
                        full_transcript = segment_text
            
            if not full_transcript:
                return
            
            # Skip if starts with noise annotation and we don't have a sentence yet
            if not self.last_transcription and self._starts_with_noise_annotation(full_transcript):
                print(f"[REJECTED] Sentence starts with noise annotation: '{full_transcript[:50]}...'")
                return
            
            print(f"\n[TRANSCRIPTION] {full_transcript}")
            
            # ALWAYS replace the last transcription with the new one
            self.last_transcription = full_transcript
            
            # Check if we have new words (this will destroy timer if active)
            if self._has_new_words(full_transcript):
                self._destroy_timer()
            
            # Check for sentence ending
            is_sentence_end, is_pause = self._detect_sentence_end(full_transcript)
            
            # Start timer ONLY if:
            # 1. We see sentence ending punctuation
            # 2. We can segment (time requirements met)
            # 3. It's not just noise
            # 4. We're not already waiting
            if (is_sentence_end and 
                self._can_segment() and 
                not self._is_noise_only_transcription(full_transcript) and 
                not self.waiting_for_continuation):
                
                # Start the timer
                self.waiting_for_continuation = True
                self.punctuation_detected_time = time.time()
                # print(f"setting new timer for punctuation detected at {self.punctuation_detected_time}")
                print(f"[PUNCTUATION DETECTED] Starting {self.finalization_delay}s timer...")
            
            # Handle immediate finalization for silence (only if no timer active)
            elif (is_silence_triggered and 
                  self._can_segment() and 
                  not self._is_processing_indicator(full_transcript) and 
                  not self.waiting_for_continuation):
                
                if self._is_noise_only_transcription(full_transcript):
                    # Only noise during silence and no timer - finalize if we have content
                    if self.last_transcription and len(self.last_transcription.strip()) > 3:
                        print(f"[SILENCE FINALIZATION] Finalizing on silence with noise only")
                        self._finalize_sentence()
                else:
                    # Real speech during silence - continue
                    pass
            
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
        self.last_transcription = ""
        self.completed_sentences = []
        self.sentence_start_time = None
        self.waiting_for_continuation = False
        self.punctuation_detected_time = None
        self.last_word_count = 0
        self.audio_queue = queue.Queue()
        self.silence_frames = 0
        self.is_speech_active = False
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
            
            print("Listening with SIMPLIFIED sentence logic...")
            print("- Uses LAST transcription only (no accumulation)")
            print("- Timer starts on punctuation, destroyed on new words")
            print("- 5-second delay before sending to LLM")
            print("- No more duplication issues!")
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
        if len(self.current_sentence_buffer) > 0:
            try:
                print("\n[PROCESSING FINAL SEGMENT]")
                self._process_sentence_segment(self.current_sentence_buffer, is_silence_triggered=True)
            except Exception as e:
                print(f"Error processing final segment: {e}")
        
        # Finalize any pending sentence
        if self.last_transcription and len(self.last_transcription.strip()) > 3:
            print("\n[FINALIZING PENDING SENTENCE]")
            self._finalize_sentence()
        
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