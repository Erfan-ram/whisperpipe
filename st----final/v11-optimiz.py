#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time speech-to-text streaming with OpenAI Whisper using a local model
Optimized Dual Buffer System - Text Only Stable Buffer
Memory efficient: Only keeps stable text, discards processed audio
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
import whisper
import torch

class WhisperStreamingTranscriberWithSpecials:
    def __init__(self, model_name="base.en", buffer_duration_seconds=5):
        """
        Initialize the transcriber with OpenAI Whisper model
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, base.en, small.en)
            buffer_duration_seconds: Time window in seconds to hold audio for processing
        """
        print(f"Loading Whisper model: {model_name}")
        
        try:
            # Initialize the OpenAI Whisper model
            self.model = whisper.load_model(model_name)
            print("Model loaded successfully!")
            
            # Check if CUDA is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Audio recording parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper expects 16kHz
        self.CHUNK = 1024
        
        # Calculate buffer parameters
        self.buffer_duration_seconds = buffer_duration_seconds
        self.buffer_size = int(self.RATE * buffer_duration_seconds)
        
        # Processing parameters
        self.audio_queue = queue.Queue()
        self.rolling_buffer = np.array([], dtype=np.float32)
        self.is_recording = False
        
        # Optimized Buffer System - TEXT ONLY
        self.stable_text_buffer = ""  # Confirmed text that won't change
        self.active_audio_buffer = np.array([], dtype=np.float32)  # Current processing audio ONLY
        
        # Transcription tracking for pattern detection
        self.transcription_history = []  # Store last few transcriptions
        self.temp_timestamps_dict = {}  # Word -> end_time mapping when duplicates found
        self.duplicate_detection_state = "waiting"  # "waiting", "found_duplicate", "confirmed"
        self.confirmed_pattern = ""  # The pattern we've confirmed 3 times
        
        # Basic parameters
        self.completed_sentences = []  # Store completed sentences
        self.sentence_start_time = None  # Track when current sentence started
        self.max_sentence_duration = 60.0  # Max sentence duration
        
        # Current transcription state
        self.last_transcription = ""           # Current transcription
        self.last_word_count = 0              # Track word count to detect new words
        
        # Processing state
        self.last_transcription_time = time.time()
        self.processing_interval = 1.0
        
        # Foreign language and annotation detection parameters
        self.foreign_language_rejection_count = 0
        self.max_foreign_rejections = 3  # Reset after 3 consecutive foreign language detections
        self.last_rejection_time = None
        self.rejection_reset_timeout = 10.0  # Reset rejection count after 10 seconds
        
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
    
    def _find_longest_common_prefix(self, text1, text2):
        """
        Find the longest common prefix between two transcriptions
        Returns the common text portion
        """
        if not text1 or not text2:
            return ""
        
        # Split into words for better comparison
        words1 = text1.split()
        words2 = text2.split()
        
        common_words = []
        for i in range(min(len(words1), len(words2))):
            if words1[i] == words2[i]:
                common_words.append(words1[i])
            else:
                break
        
        return " ".join(common_words)
    
    def _extract_word_timestamps(self, result):
        """
        Extract word-level timestamps from Whisper result
        Returns: dict of {word: end_time}
        """
        word_timestamps = {}
        
        if "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word = word_info.get("word", "").strip()
                        end_time = word_info.get("end", 0)
                        if word:
                            # Handle punctuation attached to words
                            word_clean = word.lower().strip(".,!?;:")
                            word_timestamps[word] = end_time
                            if word_clean != word.lower():
                                word_timestamps[word_clean] = end_time
        
        return word_timestamps
    
    def _find_last_word_end_time(self, text, word_timestamps):
        """
        Find the end time of the last word in the given text
        """
        if not text or not word_timestamps:
            return None
        
        words = text.split()
        if not words:
            return None
        
        # Try to find the last word's timestamp
        last_word = words[-1]
        
        # Try exact match first
        if last_word in word_timestamps:
            return word_timestamps[last_word]
        
        # Try without punctuation
        last_word_clean = last_word.lower().strip(".,!?;:")
        if last_word_clean in word_timestamps:
            return word_timestamps[last_word_clean]
        
        # Try to find any word that contains our word
        for word, timestamp in word_timestamps.items():
            if last_word.lower() in word.lower() or word.lower() in last_word.lower():
                return timestamp
        
        return None
    
    def _commit_to_stable_buffer(self, stable_text, end_time):
        """
        Move confirmed text to stable buffer and trim audio efficiently
        OPTIMIZED: Only keep text, discard processed audio to save memory
        """
        print(f"\n[COMMITTING TO STABLE] Text: '{stable_text}'")
        
        # Add to stable text buffer
        if self.stable_text_buffer:
            self.stable_text_buffer += " " + stable_text
        else:
            self.stable_text_buffer = stable_text
        
        # OPTIMIZED: Trim processed audio from active buffer (don't store it)
        end_samples = int(end_time * self.RATE)
        
        if len(self.active_audio_buffer) > end_samples:
            # Keep only unprocessed audio in active buffer
            self.active_audio_buffer = self.active_audio_buffer[end_samples:]
            print(f"[AUDIO TRIMMED] Removed {end_time}s of processed audio")
        else:
            # If end_time exceeds buffer, clear all processed audio
            self.active_audio_buffer = np.array([], dtype=np.float32)
            print(f"[AUDIO CLEARED] All audio processed")
        
        # Clean debug output as requested
        print(f"\nstable buffer: {self.stable_text_buffer}")
        remaining_text = self.last_transcription[len(stable_text):].strip()
        print(f"active buffer: {remaining_text}")
        
        # Check if we should send to LLM
        self._check_and_send_to_llm()
    
    def _check_and_send_to_llm(self):
        """
        OPTIMIZED: Send to LLM when stable buffer has content and active buffer is minimal
        """
        if not self.stable_text_buffer:
            return
        
        # Check if active buffer is effectively empty (no meaningful content being processed)
        active_content_length = len(self.active_audio_buffer) / self.RATE  # seconds
        
        # If active buffer is very small (less than 0.5 seconds) or empty, send to LLM
        if active_content_length < 0.5:
            print(f"\n[AUTO SEND TO LLM] Stable buffer ready")
            self._send_to_llm(self.stable_text_buffer)
            
            # Clear the stable buffer after sending
            self._clear_stable_buffer()
    
    def _clear_stable_buffer(self):
        """
        Clear the stable buffer after sending to LLM
        OPTIMIZED: Only clear text (no audio to clear)
        """
        print(f"[STABLE BUFFER CLEARED] Content sent to LLM")
        self.stable_text_buffer = ""
    
    def _process_transcription_pattern(self, new_text, word_timestamps):
        """
        Process new transcription for pattern detection and buffer management
        """
        # Add to history
        self.transcription_history.append(new_text)
        
        # Keep only last 3 transcriptions
        if len(self.transcription_history) > 3:
            self.transcription_history.pop(0)
        
        # Need at least 2 transcriptions to compare
        if len(self.transcription_history) < 2:
            return
        
        current_text = self.transcription_history[-1]
        previous_text = self.transcription_history[-2]
        
        # Find common prefix between current and previous
        common_prefix = self._find_longest_common_prefix(previous_text, current_text)
        
        if len(common_prefix) > 10:  # Only consider meaningful common prefixes
            
            if self.duplicate_detection_state == "waiting":
                # First time we see a duplicate
                print(f"\n[DUPLICATE DETECTED] Common text: '{common_prefix}'")
                
                # Save word timestamps for this pattern
                self.temp_timestamps_dict = word_timestamps.copy()
                self.confirmed_pattern = common_prefix
                self.duplicate_detection_state = "found_duplicate"
                
                # Debug output
                print(f"[saved dic double founded : {common_prefix}]")
                
            elif self.duplicate_detection_state == "found_duplicate":
                # Check if we have 3rd confirmation
                if len(self.transcription_history) >= 3:
                    third_text = self.transcription_history[-3]
                    common_with_third = self._find_longest_common_prefix(common_prefix, third_text)
                    
                    if len(common_with_third) > 10 and common_with_third == self.confirmed_pattern:
                        # We have 3-way confirmation!
                        print(f"\n[3-WAY CONFIRMATION] Confirmed pattern: '{self.confirmed_pattern}'")
                        
                        # Find the end time of the last word in confirmed pattern
                        end_time = self._find_last_word_end_time(self.confirmed_pattern, self.temp_timestamps_dict)
                        
                        if end_time is not None:
                            # Commit to stable buffer
                            self._commit_to_stable_buffer(self.confirmed_pattern, end_time)
                            
                            # Reset state for next pattern detection
                            self.duplicate_detection_state = "waiting"
                            self.confirmed_pattern = ""
                            self.temp_timestamps_dict = {}
                            
                            # Clear transcription history to start fresh
                            self.transcription_history = []
                        else:
                            print(f"[WARNING] Could not find timestamp for last word in pattern")
    
    def _detect_foreign_language_or_annotation(self, text):
        """
        Detect if transcription contains foreign language indicators or audio annotations
        Returns: (is_foreign_language, is_audio_annotation, rejection_reason)
        """
        if not text:
            return False, False, None
        
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # Pattern 1: Direct foreign language indicators
        foreign_language_patterns = [
            r'\(.*speaking in.*language.*\)',
            r'\(.*foreign language.*\)',
            r'\(.*speaks in.*\)',
            r'\(.*indistinct.*speaks in.*\)',
            r'\(.*speaking.*french.*\)',
            r'\(.*speaking.*spanish.*\)',
            r'\(.*speaking.*german.*\)',
            r'\(.*non-english.*\)',
            r'\[.*foreign.*language.*\]',
            r'\[.*non-english.*\]'
        ]
        
        for pattern in foreign_language_patterns:
            if re.search(pattern, text_lower):
                return True, False, f"Foreign language pattern: {pattern}"
        
        # Pattern 2: Audio/environmental annotations
        audio_annotation_patterns = [
            r'\(.*wind.*blowing.*\)',
            r'\(.*buzzing.*\)',
            r'\(.*static.*\)',
            r'\(.*noise.*\)',
            r'\(.*music.*\)',
            r'\(.*background.*\)',
            r'\(.*ambient.*\)',
            r'\(.*sound.*\)',
            r'\(.*audio.*\)',
            r'\(.*silence.*\)',
            r'\(.*muffled.*\)',
            r'\(.*distorted.*\)',
            r'\[.*music.*\]',
            r'\[.*noise.*\]',
            r'\[.*silence.*\]'
        ]
        
        for pattern in audio_annotation_patterns:
            if re.search(pattern, text_lower):
                return False, True, f"Audio annotation: {pattern}"
        
        # Pattern 3: Check if transcription is MOSTLY parenthetical content
        cleaned_text = re.sub(r'\([^)]*\)', '', text_clean)
        cleaned_text = re.sub(r'\[[^\]]*\]', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # If less than 20% is actual speech content, consider it annotation-heavy
        if len(cleaned_text) < len(text_clean) * 0.2:
            return False, True, "Mostly parenthetical/bracketed content"
        
        return False, False, None
    
    def _should_reset_due_to_foreign_language(self):
        """
        Check if we should reset the sentence state due to repeated foreign language detection
        """
        current_time = time.time()
        
        # Reset rejection counter if enough time has passed
        if (self.last_rejection_time and 
            current_time - self.last_rejection_time > self.rejection_reset_timeout):
            self.foreign_language_rejection_count = 0
            self.last_rejection_time = None
        
        return self.foreign_language_rejection_count >= self.max_foreign_rejections
    
    def _reset_sentence_state(self, reason="Foreign language detection"):
        """
        Reset the current sentence state and clear buffers
        OPTIMIZED: Only clear necessary data
        """
        print(f"\n[RESET STATE] {reason}")
        
        # Clear sentence-related state
        self.active_audio_buffer = np.array([], dtype=np.float32)
        self.last_transcription = ""
        self.sentence_start_time = None
        self.last_word_count = 0
        
        # Reset pattern detection state
        self.transcription_history = []
        self.temp_timestamps_dict = {}
        self.duplicate_detection_state = "waiting"
        self.confirmed_pattern = ""
        
        # Reset foreign language detection counters
        self.foreign_language_rejection_count = 0
        self.last_rejection_time = None
        
        print(f"[RESET STATE] Ready for fresh detection")
    
    def _count_meaningful_words(self, text):
        """
        Count meaningful words in text (excluding noise annotations and special tokens)
        """
        if not text:
            return 0
        
        # Remove noise annotations
        cleaned = re.sub(r'\([^)]+\)', '', text).strip()
        
        # Remove Whisper special tokens
        cleaned = re.sub(r'\[[^\]]+\]', '', cleaned).strip()
        
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
    
    def _should_force_segmentation(self):
        """Check if we should force segmentation due to time limits"""
        if self.sentence_start_time is None:
            return False
        
        duration = time.time() - self.sentence_start_time
        return duration > self.max_sentence_duration
    
    def _process_audio(self):
        """
        OPTIMIZED: Simplified audio processing focused on efficiency
        """
        while self.is_recording:
            try:
                chunk_count = 0
                start_time = time.time()
                
                # Check if we should reset due to foreign language detection
                if self._should_reset_due_to_foreign_language():
                    self._reset_sentence_state("Too many foreign language detections")
                
                # Collect audio chunks
                while not self.audio_queue.empty() and chunk_count < 15:
                    chunk = self.audio_queue.get(block=False)
                    
                    with self.lock:
                        self.rolling_buffer = np.append(self.rolling_buffer, chunk)
                    
                    chunk_count += 1
                    
                    # Start sentence timing if not already started
                    if self.sentence_start_time is None:
                        self.sentence_start_time = time.time()
                
                # Maintain rolling buffer size
                with self.lock:
                    if len(self.rolling_buffer) > self.buffer_size:
                        self.rolling_buffer = self.rolling_buffer[-self.buffer_size:]
                
                # Add to active audio buffer
                current_time = time.time()
                if chunk_count > 0:
                    with self.lock:
                        chunk_samples = int(self.CHUNK * chunk_count)
                        if len(self.rolling_buffer) >= chunk_samples:
                            latest_audio = np.copy(self.rolling_buffer[-chunk_samples:])
                        else:
                            latest_audio = np.copy(self.rolling_buffer)
                    
                    if len(latest_audio) > 0:
                        self.active_audio_buffer = np.append(self.active_audio_buffer, latest_audio)
                        
                        # Limit active buffer size (max 60 seconds)
                        max_active_buffer = int(self.RATE * self.max_sentence_duration)
                        if len(self.active_audio_buffer) > max_active_buffer:
                            self.active_audio_buffer = self.active_audio_buffer[-max_active_buffer:]
                
                # Process during ongoing audio (periodic transcription)
                min_processing_buffer = int(self.RATE * 0.8)
                should_process = (
                    (current_time - self.last_transcription_time >= self.processing_interval) and 
                    len(self.active_audio_buffer) > min_processing_buffer
                )
                
                # Force segmentation if sentence is too long
                should_force = self._should_force_segmentation()
                
                if should_process or should_force:
                    if should_force:
                        print("\n[FORCE SEGMENTATION - Time limit reached]")
                    
                    self._process_sentence_segment(self.active_audio_buffer)
                    self.last_transcription_time = time.time()
                
                # Optimized sleep
                elapsed = time.time() - start_time
                sleep_time = max(0.01, 0.05 - elapsed)
                time.sleep(sleep_time)
                
            except queue.Empty:
                time.sleep(0.05)
            except Exception as e:
                print(f"Error in audio processing thread: {e}")
                time.sleep(0.1)
    
    def _process_sentence_segment(self, audio_buffer):
        """
        OPTIMIZED: Process sentence segment efficiently
        """
        min_segment_size = int(self.RATE * 0.3)
        if len(audio_buffer) < min_segment_size:
            return
        
        try:
            # Transcribe using OpenAI Whisper with word-level timestamps
            result = self.model.transcribe(
                audio_buffer, 
                fp16=False, 
                language="en",
                word_timestamps=True,  # Enable word-level timestamps!
                suppress_tokens=None  # Don't suppress any tokens, including special ones
            )
            
            new_text = result["text"].strip()
            
            if not new_text:
                return
            
            print(f"\n[TRANSCRIPTION] {new_text}")
            
            # Extract word-level timestamps
            word_timestamps = self._extract_word_timestamps(result)
            
            # Check for foreign language or audio annotations
            is_foreign, is_audio_annotation, rejection_reason = self._detect_foreign_language_or_annotation(new_text)
            
            if is_foreign or is_audio_annotation:
                print(f"[REJECTED] {rejection_reason}")
                
                # Increment rejection counter and update timestamp
                self.foreign_language_rejection_count += 1
                self.last_rejection_time = time.time()
                
                print(f"[REJECTION COUNT] {self.foreign_language_rejection_count}/{self.max_foreign_rejections}")
                
                # If we've had too many rejections, reset immediately
                if self.foreign_language_rejection_count >= self.max_foreign_rejections:
                    self._reset_sentence_state("Maximum foreign language rejections reached")
                
                return  # Don't process this transcription further
            
            # Reset foreign language rejection counter on successful English transcription
            if self.foreign_language_rejection_count > 0:
                print(f"[ENGLISH DETECTED] Resetting foreign language rejection counter")
                self.foreign_language_rejection_count = 0
                self.last_rejection_time = None
            
            # Process pattern detection and buffer management
            self._process_transcription_pattern(new_text, word_timestamps)
            
            # Replace the last transcription with the new one
            self.last_transcription = new_text
            
            # Check if we have new words
            self._has_new_words(new_text)
            
        except Exception as e:
            print(f"Error in sentence segment processing: {e}")
    
    def _send_to_llm(self, text):
        """Send completed sentence to LLM for processing"""
        print(f"\n[LLM INPUT]: {text}")
        # Your LLM integration code here
        # Example: response = your_llm_model.generate_answer(text)
        # print(f"[LLM RESPONSE]: {response}")
        
        # Store completed sentence for summary
        sentence_data = {
            'text': text.strip(),
            'timestamp': time.strftime('%H:%M:%S'),
            'duration': time.time() - self.sentence_start_time if self.sentence_start_time else 0
        }
        self.completed_sentences.append(sentence_data)
    
    def start_streaming(self):
        """Start streaming from microphone and transcribing"""
        self.is_recording = True
        self.rolling_buffer = np.array([], dtype=np.float32)
        
        # Initialize optimized buffer system
        self.stable_text_buffer = ""  # TEXT ONLY - no audio storage
        self.active_audio_buffer = np.array([], dtype=np.float32)
        
        self.last_transcription = ""
        self.completed_sentences = []
        self.sentence_start_time = None
        self.last_word_count = 0
        self.audio_queue = queue.Queue()
        self.last_transcription_time = time.time()
        
        # Reset pattern detection state
        self.transcription_history = []
        self.temp_timestamps_dict = {}
        self.duplicate_detection_state = "waiting"
        self.confirmed_pattern = ""
        
        # Reset foreign language detection state
        self.foreign_language_rejection_count = 0
        self.last_rejection_time = None
        
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
            
            print("Listening with Optimized Memory-Efficient System...")
            print("- Text-only stable buffer (no audio waste)")
            print("- Intelligent pattern detection")
            print("- Memory optimized for long conversations")
            print("- Auto-send to LLM when ready")
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
        
        # Process any remaining audio in active buffer
        if len(self.active_audio_buffer) > 0:
            try:
                print("\n[PROCESSING FINAL SEGMENT]")
                self._process_sentence_segment(self.active_audio_buffer)
            except Exception as e:
                print(f"Error processing final segment: {e}")
        
        # Send any remaining content in stable buffer
        if self.stable_text_buffer:
            print("\n[SENDING FINAL STABLE BUFFER]")
            self._send_to_llm(self.stable_text_buffer)
        
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
    
    # Model selection - you can change this to any Whisper model
    MODEL_NAME = "base.en"  # Options: tiny, base, small, medium, large, tiny.en, base.en, small.en
    
    try:
        # Initialize the transcriber with OpenAI Whisper
        transcriber = WhisperStreamingTranscriberWithSpecials(model_name=MODEL_NAME)
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