#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time speech-to-text streaming with OpenAI Whisper using a local model
Enhanced with Dual Buffer System (Text Only) and Simplified Timer Logic
Prevents exponential reprocessing by committing stable portions
Simplified to use only stable buffer delay timing
Fixed: Preserve stable buffer during noise rejections
"""
import numpy as np
import pyaudio
import threading
import time
import sys
import signal
import queue
import re
from pynput import keyboard
import whisper
import torch

# Try to import sounddevice, but handle gracefully if not available
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError) as e:
    SOUNDDEVICE_AVAILABLE = False
    print(f"[WARNING] sounddevice not available: {e}")
    print("[INFO] Audio device management will use PyAudio fallback")

class pipeStream:
    def __init__(self, model_name="base", language="en", finalization_delay=10.0, processing_interval=1.0, buffer_duration_seconds=5.0, debug_mode=False):
        """
        Initialize the transcriber with OpenAI Whisper model
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, base, small.en)
            language: Language code for transcription (e.g., "en", "es", "fr")
            finalization_delay: Wait time in seconds before finalizing transcription (default 10.0)
            processing_interval: Interval in seconds between processing cycles (default 1.0)
            buffer_duration_seconds: Time window in seconds to hold audio for processing (default 5.0)
            debug_mode: Enable debug mode for detailed logging (default False)
        """
        self._debug_mode_enabled = debug_mode
        self.language = language  # Store language for Whisper transcription
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
        self.max_buffer_size = int(self.RATE * buffer_duration_seconds)
        
        # Processing parameters
        self.audio_queue = queue.Queue()
        self.rolling_buffer = np.array([], dtype=np.float32)
        self.is_recording = False
        
        # Enhanced Dual Buffer System (Text Only)
        self.stable_text_buffer = ""  # Confirmed text that won't change
        self.active_audio_buffer = np.array([], dtype=np.float32)  # Current processing audio
        
        # Transcription tracking for pattern detection
        self.transcription_history = []  # Store last few transcriptions
        # self.temp_timestamps_dict = {}  # Word -> end_time mapping when duplicates found
        self.duplicate_detection_state = "waiting"  # "waiting", "found_duplicate", "confirmed"
        self.confirmed_pattern = ""  # The pattern we've confirmed 3 times
        
        # Sentence-based segmentation parameters
        self.completed_sentences = []  # Store completed sentences
        self.sentence_start_time = None  # Track when current sentence started
        self.max_sentence_duration = 120.0  # Max sentence duration
        self.max_active_buffer_duration = 25.0  # Max active buffer duration
        self.min_sentence_duration = 1.5   # Minimum duration before allowing segmentation
        self.last_language = None  # Track last detected language
        
        # Simplified Timer Logic - Only stable buffer based
        self.last_transcription = ""           # Always keep the LAST transcription only
        self.finalization_delay = finalization_delay        # Wait specified seconds after last stable buffer update
        self.last_stable_buffer_update = None # When stable buffer was last updated
        self.last_word_count = 0              # Track word count to detect new words
        
        # Sentence detection patterns
        self.sentence_endings = ['.', '?', '!']
        self.pause_endings = [',', ';', ':']  # Shorter pauses, not full sentence breaks
        
        # Processing state
        self.last_transcription_time = time.time()
        self.processing_interval = processing_interval
        
        # Foreign language and annotation detection parameters
        self.foreign_language_rejection_count = 0
        self.max_foreign_rejections = 3  # Reset after 3 consecutive foreign language detections
        self.last_rejection_time = None
        self.rejection_reset_timeout = 5.0  # Reset rejection count after 5 seconds
        # empty_transcribe detection parameters
        self.empty_transcribe_rejection_count = 0
        self.max_empty_rejections = 4  # Reset after 4 consecutive times
        # Threading and synchronization
        self.process_thread = None
        self.lock = threading.Lock()
        
        # Callback system for LLM integration
        self._def_callback = None
        
        # Pause/Resume functionality
        self._is_paused = False
        self._pause_lock = threading.Lock()
        
        # Summary tracking
        self._summary_printed = False
        
        # Initialize PyAudio
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            print(f"Error initializing PyAudio: {e}")
            sys.exit(1)
        self.stream = None
        
        # Audio device management
        self._selected_input_device = None
        
        # Setup signal handling for graceful shutdown
        self._setup_signal_handling()
    
    def _setup_signal_handling(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            print(f"\n[SIGNAL] Received signal {sig}. Shutting down gracefully...")
            self.stop_streaming()
            self.close()
            print("Transcription ended.")
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._debug_print("[SIGNAL] Signal handlers registered (SIGINT, SIGTERM)")
    
    def input_devices(self):
        """
        List all available audio input devices with their IDs
        
        Returns:
            list: List of dictionaries containing device info
                 Each dict has keys: 'id', 'name', 'channels', 'default_samplerate'
        """
        devices = []
        
        if SOUNDDEVICE_AVAILABLE:
            try:
                device_list = sd.query_devices()
                
                for i, device in enumerate(device_list):
                    # Only include devices that support input
                    if device['max_input_channels'] > 0:
                        devices.append({
                            'id': i,
                            'name': device['name'],
                            'channels': device['max_input_channels'],
                            'default_samplerate': device['default_samplerate'],
                            'is_default': i == sd.default.device[0] if sd.default.device[0] is not None else False
                        })
                
            except Exception as e:
                print(f"Error querying audio devices with sounddevice: {e}")
                devices = self._get_pyaudio_devices()
        else:
            # Fallback to PyAudio device listing
            devices = self._get_pyaudio_devices()
        
        # Print devices for user convenience
        if devices:
            print("\n[AUDIO DEVICES] Available input devices:")
            for device in devices:
                default_marker = " (DEFAULT)" if device.get('is_default', False) else ""
                print(f"  ID {device['id']}: {device['name']}{default_marker}")
                print(f"    Channels: {device['channels']}, Sample Rate: {device.get('default_samplerate', 'Unknown')}")
        else:
            print("\n[AUDIO DEVICES] No input devices found or error querying devices")
        
        return devices
    
    def _get_pyaudio_devices(self):
        """
        Get audio devices using PyAudio as fallback when sounddevice is not available
        
        Returns:
            list: List of device dictionaries
        """
        devices = []
        
        try:
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                
                # Only include devices that support input
                if device_info['maxInputChannels'] > 0:
                    devices.append({
                        'id': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'default_samplerate': device_info['defaultSampleRate'],
                        'is_default': i == self.p.get_default_input_device_info()['index']
                    })
                    
        except Exception as e:
            print(f"Error querying PyAudio devices: {e}")
        
        return devices
    
    def set_input_device(self, device_id):
        """
        Set a specific audio input device by ID
        
        Args:
            device_id (int): Device ID from input_devices() method
            
        Returns:
            bool: True if device was set successfully, False otherwise
        """
        try:
            # Check if we're currently recording - can't change device while streaming
            if self.is_recording:
                print("[ERROR] Cannot change input device while streaming. Stop streaming first.")
                return False
            
            # Validate device ID using the appropriate method
            if SOUNDDEVICE_AVAILABLE:
                try:
                    devices = sd.query_devices()
                    if device_id < 0 or device_id >= len(devices):
                        print(f"[ERROR] Invalid device ID: {device_id}")
                        return False
                    
                    device = devices[device_id]
                    if device['max_input_channels'] == 0:
                        print(f"[ERROR] Device {device_id} ({device['name']}) does not support audio input")
                        return False
                    
                    device_name = device['name']
                except Exception as e:
                    print(f"Error validating device with sounddevice: {e}")
                    return False
            else:
                # Fallback to PyAudio validation
                try:
                    if device_id < 0 or device_id >= self.p.get_device_count():
                        print(f"[ERROR] Invalid device ID: {device_id}")
                        return False
                    
                    device_info = self.p.get_device_info_by_index(device_id)
                    if device_info['maxInputChannels'] == 0:
                        print(f"[ERROR] Device {device_id} ({device_info['name']}) does not support audio input")
                        return False
                    
                    device_name = device_info['name']
                except Exception as e:
                    print(f"Error validating device with PyAudio: {e}")
                    return False
            
            self._selected_input_device = device_id
            print(f"[AUDIO] Input device set to ID {device_id}: {device_name}")
            
            return True
            
        except Exception as e:
            print(f"Error setting input device: {e}")
            return False
    
    def get_current_input_device(self):
        """
        Get information about the currently selected input device
        
        Returns:
            dict: Device information or None if no device selected
        """
        if self._selected_input_device is None:
            return None
        
        try:
            if SOUNDDEVICE_AVAILABLE:
                devices = sd.query_devices()
                device = devices[self._selected_input_device]
                return {
                    'id': self._selected_input_device,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate']
                }
            else:
                # Fallback to PyAudio
                device_info = self.p.get_device_info_by_index(self._selected_input_device)
                return {
                    'id': self._selected_input_device,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'default_samplerate': device_info['defaultSampleRate']
                }
        except Exception as e:
            print(f"Error getting current device info: {e}")
            return None
    
    def debug_mode(self, enabled=True):
        """
        Enable or disable debug mode
        
        Args:
            enabled: True to enable debug mode, False to disable (default True)
        """
        self._debug_mode_enabled = enabled
        if enabled:
            print("Debug mode enabled")
        else:
            print("Debug mode disabled")
    
    def set_def_callback(self, callback_function) -> str:
        """
        Set a callback function to handle text when it's sent to LLM
        
        Args:
            callback_function: Function that takes a string (text) as parameter
                             Example: def my_llm_handler(text):
                                         print(f"Processing: {text}")
                                         # Your LLM processing logic here
        Returns:
            str: A message indicating the result of the callback processing
        """
        if callback_function is not None and not callable(callback_function):
            raise ValueError("Callback function must be callable or None")
        
        self._def_callback = callback_function
        if callback_function:
            print("LLM callback function registered")
        else:
            print("LLM callback function cleared")
    
    def pause_streaming(self):
        """
        Pause the audio streaming and processing temporarily
        The audio stream continues but processing is paused
        """
        with self._pause_lock:
            if not self.is_recording:
                print("Transcriber is not currently running")
                return False
            
            if self._is_paused:
                print("Transcriber is already paused")
                return False
            
            self._is_paused = True
            print("Audio streaming paused")
            return True
    
    def resume_streaming(self):
        """
        Resume the paused audio streaming and processing
        """
        with self._pause_lock:
            if not self.is_recording:
                print("Transcriber is not currently running")
                return False
            
            if not self._is_paused:
                print("Transcriber is not paused")
                return False
            
            self._is_paused = False
            print("Audio streaming resumed")
            return True
    
    def is_paused(self):
        """
        Check if the transcriber is currently paused
        
        Returns:
            bool: True if paused, False if running or stopped
        """
        with self._pause_lock:
            return self._is_paused
    
    def is_running(self):
        """
        Check if the transcriber is currently running (not stopped)
        
        Returns:
            bool: True if running (may be paused), False if stopped
        """
        return self.is_recording
    
    def _debug_print(self, message):
        """
        Print debug message only if debug mode is enabled
        
        Args:
            message: Debug message to print
        """
        if self._debug_mode_enabled:
            print(message)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream"""
        if status:
            self._debug_print(f"PyAudio status: {status}")
        
        try:
            # Convert audio data to numpy array and normalize
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_queue.put(audio_data)
        except Exception as e:
            self._debug_print(f"Error in audio callback: {e}")
        
        return (in_data, pyaudio.paContinue)

    # def _find_longest_common_prefix(self, text1, text2, label=""):
    #     """
    #     Find the longest common prefix between two transcriptions
    #     Returns the common text portion
    #     """
    #     if not text1 or not text2:
    #         return ""
        
    #     # Split into words for better comparison
    #     words1 = text1.lower().split()
    #     words2 = text2.lower().split()
        
    #     common_words = []
    #     for i in range(min(len(words1), len(words2))):
    #         if words1[i] == words2[i]:
    #             common_words.append(words1[i])
    #         else:
    #             break
    #     res = " ".join(common_words)
    #     print(f"DEBUG: Found common prefix: '{res}'\n and length {len(res)} and caller : {label}")
        
    #     return res

    def _find_longest_common_prefix_with_similarity(self, text1, text2, min_similarity=0.8, return_from="text1"):
        """
        Find the longest common prefix between two transcriptions using similarity scoring
        Returns the common text portion based on similarity threshold
        
        Args:
            text1, text2: Input texts to compare
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            return_from: "text1" or "text2" - which text to return words from
        
        Returns:
            String: The longest prefix that maintains the similarity threshold
        """
        if not text1 or not text2:
            return ""
        
        # Split into words for comparison
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        original_words1 = text1.split()
        original_words2 = text2.split()
        
        if not words1 or not words2:
            return ""
        
        common_words = []
        total_comparisons = 0
        similar_matches = 0
        
        # Compare words from the beginning
        max_length = min(len(words1), len(words2))
        
        for i in range(max_length):
            word1 = words1[i]
            word2 = words2[i]
            total_comparisons += 1
            
            # Calculate similarity for this word pair
            word_similarity = self._calculate_word_similarity(word1, word2)
            
            # If words are similar enough, count as match
            if word_similarity >= min_similarity:
                similar_matches += 1
                # Use words from specified text
                if return_from == "text2":
                    original_word = original_words2[i] if i < len(original_words2) else word2
                else:
                    original_word = original_words1[i] if i < len(original_words1) else word1
                common_words.append(original_word)
            else:
                # Check if overall similarity is still above threshold
                current_similarity = similar_matches / total_comparisons if total_comparisons > 0 else 0
                
                if current_similarity < min_similarity:
                    # If adding this dissimilar word drops us below threshold, stop
                    break
                else:
                    # If overall similarity is still good, include this word but mark as different
                    similar_matches += 0.5  # Partial credit for maintaining flow
                    if return_from == "text2":
                        original_word = original_words2[i] if i < len(original_words2) else word2
                    else:
                        original_word = original_words1[i] if i < len(original_words1) else word1
                    common_words.append(original_word)
        
        # Final similarity check - ensure we meet the minimum threshold
        final_similarity = similar_matches / total_comparisons if total_comparisons > 0 else 0
        
        # If we don't meet the threshold, trim back until we do
        if final_similarity < min_similarity and len(common_words) > 1:
            # Try removing words from the end until we meet threshold
            for trim_count in range(1, len(common_words)):
                trimmed_comparisons = total_comparisons - trim_count
                trimmed_matches = similar_matches - (trim_count * 0.5)  # Assume trimmed words were partial matches
                
                if trimmed_comparisons > 0:
                    trimmed_similarity = trimmed_matches / trimmed_comparisons
                    if trimmed_similarity >= min_similarity:
                        common_words = common_words[:-trim_count]
                        final_similarity = trimmed_similarity
                        break
        
        result = " ".join(common_words)
        final_similarity = final_similarity * 100  # Convert to percentage for logging
        # final_Length = len(result)
        self._debug_print(f"DEBUG: Found common prefix with {final_similarity:.1} similarity: '{result}'")
        # self._debug_print(f"DEBUG: Length {final_Length} characters, returning from {return_from}")

        return result,final_similarity

    def _calculate_word_similarity(self, word1, word2):
        """
        Calculate similarity between two words using edit distance only
        
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if word1 == word2:
            return 1.0
        
        # Handle punctuation
        clean_word1 = self._clean_word_for_comparison(word1)
        clean_word2 = self._clean_word_for_comparison(word2)
        
        if clean_word1 == clean_word2:
            return 0.95  # High similarity for punctuation differences
        
        # Use edit distance for general similarity
        return self._levenshtein_similarity(clean_word1, clean_word2)

    def _clean_word_for_comparison(self, word):
        """
        Clean word by removing punctuation and normalizing
        """
        import string
        # Remove punctuation
        cleaned = word.translate(str.maketrans('', '', string.punctuation))
        return cleaned.lower().strip()

    def _levenshtein_similarity(self, word1, word2):
        """
        Calculate similarity using Levenshtein distance
        
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if not word1 or not word2:
            return 0.0
        
        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(word1, word2)
        max_length = max(len(word1), len(word2))
        
        if max_length == 0:
            return 1.0
        
        # Convert distance to similarity (0 distance = 1.0 similarity)
        similarity = 1.0 - (distance / max_length)
        
        # Only consider it similar if it's above a threshold
        return similarity if similarity >= 0.6 else 0.0

    def _levenshtein_distance(self, s1, s2):
        """
        Calculate the Levenshtein distance between two strings
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _extract_word_timestamps(self, result):
        """
        Extract word-level timestamps from Whisper result
        Returns: dict of {end_time: word}
        """
        word_timestamps = {}
        
        if "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word = word_info.get("word", "").strip()
                        end_time = word_info.get("end", 0)
                        if word:
                            # Use timestamp as key to prevent overwrites when words repeat
                            word_timestamps[end_time] = word
        
        return word_timestamps
    
    def _find_last_word_end_time(self, text, word_timestamps):
        """
        Find the end time of the last word in the given text
        Enhanced: Consider previous word context to choose the correct occurrence
        """
        if not text or not word_timestamps:
            return None
        
        words = text.split()
        if not words:
            return None
        
        # Try to find the last word's timestamp
        last_word = words[-1]
        
        # Check if last word contains processing indicators like "..."
        if self._is_processing_indicator(last_word):
            self._debug_print(f"Not end detected - processing indicator found . word is {last_word}")
            return None
        
        self._debug_print(f"[DEBUG] Finding end time for last word: '{last_word}'")
        self._debug_print(f"[DEBUG] Word timestamps available: {list(word_timestamps.values())}")
        
        # Clean the last word for comparison
        last_word_clean = last_word.lower().strip(".,!?;:")
        
        # Find all timestamps where the word matches our target word
        matching_timestamps = []
        for timestamp, word in word_timestamps.items():
            word_clean = word.lower().strip(".,!?;:")
            if (word == last_word or 
                word_clean == last_word_clean or 
                word.lower() == last_word.lower()):
                matching_timestamps.append(timestamp)
        
        if matching_timestamps:
            self._debug_print(f"[DEBUG] Found matching timestamps: {matching_timestamps}")
            
            # Enhanced logic: Consider previous word context if we have multiple matches
            if len(matching_timestamps) > 1 and len(words) > 1:
                expected_previous_word = words[-2]  # Get the word before the last word
                expected_previous_clean = expected_previous_word.lower().strip(".,!?;:")
                
                self._debug_print(f"[DEBUG] Multiple matches found, checking context with previous word: '{expected_previous_word}'")
                
                # Create sorted list of word timestamps for sequence analysis
                sorted_timestamps = sorted(word_timestamps.items())
                
                # Find matches that have the correct previous word context
                context_matches = []
                for target_timestamp in matching_timestamps:
                    # Find the position of this timestamp in the sequence
                    for i, (timestamp, word) in enumerate(sorted_timestamps):
                        if timestamp == target_timestamp:
                            # Check if there's a previous word and if it matches our expected context
                            if i > 0:
                                prev_timestamp, prev_word = sorted_timestamps[i-1]
                                prev_word_clean = prev_word.lower().strip(".,!?;:")
                                
                                if (prev_word == expected_previous_word or 
                                    prev_word_clean == expected_previous_clean or 
                                    prev_word.lower() == expected_previous_word.lower()):
                                    context_matches.append(target_timestamp)
                                    self._debug_print(f"[DEBUG] Context match found: '{prev_word}' -> '{word}' at {target_timestamp}")
                            break
                
                if context_matches:
                    # If we found context matches, use the latest one among them
                    result = max(context_matches)
                    self._debug_print(f"[DEBUG] Using context-based match: {result}")
                    return result
                else:
                    self._debug_print(f"[DEBUG] No context matches found, falling back to latest timestamp")
            
            # Fallback: Return the highest (latest) timestamp for the word
            result = max(matching_timestamps)
            self._debug_print(f"[DEBUG] Returning latest timestamp: {result}")
            return result
        
        # Try to find any word that contains our word (partial matching)
        for timestamp, word in word_timestamps.items():
            if last_word.lower() in word.lower() or word.lower() in last_word.lower():
                self._debug_print(f"[DEBUG] Partial match found: {word} at {timestamp}")
                return timestamp
        
        return None
    
    def _commit_to_stable_buffer(self, stable_text, end_time):
        """
        Move confirmed text to stable buffer and update audio buffer
        IMPORTANT: Reset stable buffer timer when new content is added
        """
        self._debug_print(f"\n[COMMITTING TO STABLE] Text: '{stable_text}'")
        self._debug_print(f"[COMMITTING TO STABLE] End time: {end_time}s")
        
        # Add to stable text buffer
        if self.stable_text_buffer:
            self.stable_text_buffer += " " + stable_text
        else:
            self.stable_text_buffer = stable_text
            
        # Show time before reset
        time_now = time.time()
        if self.last_stable_buffer_update:
            time_passed_before_commit = time_now - self.last_stable_buffer_update
            self._debug_print(f"\033[92m[DEBUG ---- TIME PASSED BEFORE COMMIT] {time_passed_before_commit:.1f}s since last stable buffer update\033[0m")
        
        # CRITICAL: Reset stable buffer timer when new content is committed
        self.last_stable_buffer_update = time.time()
        self._debug_print(f"\033[93m[TIMER RESET] Stable buffer timer reset due to new content commit\033[0m")
        
        # Calculate audio samples to remove from active buffer with overlap
        # alpha = 0.05  # 5% overlap factor to prevent cutting off audio mid-word
        # alpha = 0.1  # 10% overlap factor to prevent cutting off audio mid-word
        # end_samples = int(end_time * self.RATE * (1 - alpha))

        end_samples_without_overlap = int(end_time * self.RATE)
        end_samples = end_samples_without_overlap
        
        if len(self.active_audio_buffer) > end_samples:
            # Keep remaining audio in active buffer
            self.active_audio_buffer = self.active_audio_buffer[end_samples:]
            
            self._debug_print(f"[BUFFER MOVED] Removed {end_time}s of audio from active buffer")
            self._debug_print(f"[BUFFER STATUS] Active audio remaining: {len(self.active_audio_buffer)/self.RATE:.1f}s")
        
        # Debug output as requested with colors - THESE ARE ESSENTIAL MESSAGES
        print(f"\n\033[92mstable buffer: {self.stable_text_buffer}\033[0m")
        remaining_text = self.last_transcription[len(stable_text):].strip() if self.last_transcription else ""
        print(f"\033[93mactive buffer: {remaining_text}\033[0m")
    
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
        common_prefix , similarity = self._find_longest_common_prefix_with_similarity(previous_text, current_text, 0.5, "text2")
        self._debug_print(f"similarity : {similarity}")
        
        # special condition for similarity = 100
        if similarity == 100.0 and len(common_prefix) > 20:
            self._debug_print(f"\n[SPECIAL CONDITION] Similarity is 100% Confirmed pattern: '{common_prefix}'")
            end_time = self._find_last_word_end_time(common_prefix, word_timestamps)
            self._debug_print(f"[DEBUG] End time for special condition: {end_time}")
            
            if end_time is not None:
                # Commit to stable buffer
                self._commit_to_stable_buffer(common_prefix, end_time)
                # Reset state for next pattern detection
                self.duplicate_detection_state = "waiting"
                self.confirmed_pattern = ""
                
                # Clear transcription history to start fresh
                self.transcription_history = []

        # end special condition for similarity = 100
        elif similarity >= 50.0 and len(common_prefix) > 17:
            
            if self.duplicate_detection_state == "waiting":
                # First time we see a duplicate
                self._debug_print(f"\n[DUPLICATE DETECTED] Common text: '{common_prefix}'")
                # Store the common prefix as the confirmed pattern
                self.confirmed_pattern = common_prefix
                self.duplicate_detection_state = "found_duplicate"
                
                # Debug output
                self._debug_print(f"[saved dic double founded : {common_prefix}]")
                
            elif self.duplicate_detection_state == "found_duplicate":
                # Check if we have 3rd confirmation
                if len(self.transcription_history) >= 3:
                    # third_text = self.transcription_history[-3]
                    common_with_third , similarity = self._find_longest_common_prefix_with_similarity(common_prefix, self.confirmed_pattern, 0.5, "text1")

                    if similarity >= 50 :
                        # We have 3-way confirmation!
                        self._debug_print(f"\n[3-WAY CONFIRMATION] Confirmed pattern: '{common_with_third}'")
                        
                        # Find the end time of the last word in confirmed pattern
                        end_time = self._find_last_word_end_time(common_with_third, word_timestamps)
                        if end_time is not None:
                            # Commit to stable buffer
                            self._commit_to_stable_buffer(common_with_third, end_time)
                            
                            # Reset state for next pattern detection
                            self.duplicate_detection_state = "waiting"
                            self.confirmed_pattern = ""
                            
                            # Clear transcription history to start fresh
                            self.transcription_history = []
                        else:
                            self._debug_print(f"[WARNING] Could not find timestamp for last word in pattern")

                    else:
                        self.confirmed_pattern = common_prefix
                        self._debug_print(f"[DEBUG] ignored 3rd sentence .Updated confirmed pattern to: third :new logic . last vs new always")
                        
                # else:
                #     # Reset state if no meaningful common prefix found
                #     if self.duplicate_detection_state != "waiting":
                #         print(f"[RESET] No meaningful common prefix found, resetting duplicate detection state")
                #         self.duplicate_detection_state = "waiting"
                #         self.confirmed_pattern = ""
                #         self.temp_timestamps_dict = {}

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
            r'\(.*speaking.*in.*\)',         # Catches any "speaking in [anything]" pattern
            r'\(.*speaking in.*language.*\)',
            r'\(.*speaks.*in.*\)',           # Catches any "speaks in [anything]" pattern  
            r'\(.*foreign.*language.*\)',    # Foreign language indicators
            r'\(.*non-english.*\)',          # Non-English indicators
            # r'\[.*speaking.*in.*\]',         # Same patterns in square brackets
            # r'\[.*speaks.*in.*\]',
            # r'\[.*foreign.*language.*\]',
            # r'\[.*non-english.*\]',
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
        # Remove all parenthetical and bracketed content
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
        FIXED: Only clear active components, preserve stable buffer and timer
        """
        self._debug_print(f"\n[RESET STATE] {reason}")
        self._debug_print(f"[RESET STATE] Clearing ONLY active components, preserving stable buffer")
        
        # PRESERVE stable text buffer and timer - DO NOT CLEAR THESE
        # self.stable_text_buffer = ""  # DON'T CLEAR - preserve good content
        # self.last_stable_buffer_update = None  # DON'T CLEAR - preserve timer
        
        # Clear only active/volatile state
        self.active_audio_buffer = np.array([], dtype=np.float32)
        self.last_transcription = ""
        self.last_word_count = 0
        
        # Reset pattern detection state
        self.transcription_history = []
        # self.temp_timestamps_dict = {}
        self.duplicate_detection_state = "waiting"
        self.confirmed_pattern = ""
        
        # Reset foreign language detection counters
        self.foreign_language_rejection_count = 0
        self.last_rejection_time = None
        
        self._debug_print(f"[RESET STATE] Stable buffer preserved: '{self.stable_text_buffer}'")
        if self.last_stable_buffer_update:
            elapsed = time.time() - self.last_stable_buffer_update
            self._debug_print(f"[RESET STATE] Timer continues: {elapsed:.1f}s since last stable update")
        
        if self.sentence_start_time and self.last_stable_buffer_update is None:
            # If we had a sentence start time but no stable buffer update, means it is not a valid sentence
            self.sentence_start_time = None
            self._debug_print(f"[RESET STATE] Sentence start time cleared")
        
        self._debug_print(f"[RESET STATE] Ready for fresh sentence detection")
    
    def _is_noise_only_transcription(self, text):
        """
        Check if transcription contains only noise annotations (no real speech)
        """
        if not text:
            return True
        
        # Remove all parenthetical expressions (noise annotations)
        cleaned = re.sub(r'\([^)]+\)', '', text).strip()
        
        # Remove Whisper special tokens
        cleaned = re.sub(r'\[[^\]]+\]', '', cleaned).strip()
        
        # If nothing left after removing noise annotations, it's noise-only
        return len(cleaned) < 3
    
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
    
    # def _detect_sentence_end(self, text):
    #     """
    #     Detect if text contains sentence-ending punctuation
    #     Returns: (is_sentence_end, is_pause_point)
    #     """
    #     if not text:
    #         return False, False
        
    #     text = text.strip()
        
    #     # First check if this is a processing indicator - if so, don't end sentence
    #     if self._is_processing_indicator(text):
    #         print(f"[DEBUG] Processing indicator detected: '{text[-15:]}' - NOT ending sentence")
    #         return False, False
        
    #     # Check for real sentence endings
    #     has_sentence_end = False
    #     has_pause = False
        
    #     # Look for sentence endings in the last few characters
    #     for ending in self.sentence_endings:
    #         if text.endswith(ending) or text.endswith(ending + ' '):
    #             has_sentence_end = True
    #             print(f"[DEBUG] Sentence end detected with '{ending}': '{text[-15:]}'")
    #             break
        
    #     # Check for pause points (commas, etc.)
    #     for ending in self.pause_endings:
    #         if text.endswith(ending) or text.endswith(ending + ' '):
    #             has_pause = True
    #             break
        
    #     return has_sentence_end, has_pause
    
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
        Simplified: Check if we should finalize based only on stable buffer updates
        Only returns True if 10 seconds have passed since last stable buffer update
        """
        if self.last_stable_buffer_update is None:
            return False
        
        current_time = time.time()
        stable_buffer_delay_elapsed = current_time - self.last_stable_buffer_update
        
        should_finalize = stable_buffer_delay_elapsed >= self.finalization_delay
        
        if should_finalize:
            print(f"[TIMER EXPIRED] Stable buffer delay: {stable_buffer_delay_elapsed:.1f}s - finalizing sentence")
            return True
        # optinal for debug
        # else:
        #     # Show countdown for debugging (less frequently)
        #     remaining = self.finalization_delay - stable_buffer_delay_elapsed
        #     if int(remaining * 10) % 5 == 0:  # Show every 0.5 seconds
        #         print(f"[TIMER] {remaining:.1f}s remaining until finalization")
        
        return False
    
    def _finalize_sentence(self, final_text=None):
        """
        Finalize the current sentence and reset buffers
        Enhanced: Combine stable buffer with current transcription
        """
        # Combine stable buffer with current transcription
        full_text = self.stable_text_buffer
        if self.last_transcription:
            if full_text:
                full_text += " " + self.last_transcription
            else:
                full_text = self.last_transcription
        
        sentence_text = final_text if final_text else full_text
        
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
        self.stable_text_buffer = ""  # Clear stable text buffer
        self.active_audio_buffer = np.array([], dtype=np.float32)
        self.last_transcription = ""
        self.sentence_start_time = None
        self.last_stable_buffer_update = None  # Reset timer
        self.last_word_count = 0
        
        # Reset pattern detection state
        self.transcription_history = []
        # self.temp_timestamps_dict = {}
        self.duplicate_detection_state = "waiting"
        self.confirmed_pattern = ""
    
    def _process_audio(self):
        """Simplified audio processing with enhanced dual buffer management"""
        while self.is_recording:
            try:
                # Check if paused - if so, sleep and continue
                with self._pause_lock:
                    if self._is_paused:
                        time.sleep(0.1)
                        continue
                
                chunk_count = 0
                start_time = time.time()
                
                # Check if we should finalize after delay period (simplified logic)
                if self._should_finalize_after_delay():
                    
                    # ---------------- new logic + skip if we have detected another language----------------
                    # Process only first 1/3 of buffer since user likely didn't speak at the end
                    one_third_buffer = self.active_audio_buffer[:len(self.active_audio_buffer)//3]
                    if len(one_third_buffer) > int(self.RATE * 0.3) and self.last_language == self.language:  # Ensure minimum size
                        self._debug_print(f" Just proccessing first 1/3 of active audio buffer in last transcription")
                        self._process_sentence_segment(one_third_buffer)
                    # ---------------- new logic ----------------
                    self._finalize_sentence()
                
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
                    if len(self.rolling_buffer) > self.max_buffer_size:
                        self.rolling_buffer = self.rolling_buffer[-self.max_buffer_size:]
                
                # Add to active audio buffer
                current_time = time.time()
                if chunk_count > 0:
                    with self.lock:
                        # Ensure we use integer indices for slicing
                        chunk_samples = int(self.CHUNK * chunk_count)
                        if len(self.rolling_buffer) >= chunk_samples:
                            latest_audio = np.copy(self.rolling_buffer[-chunk_samples:])
                        else:
                            latest_audio = np.copy(self.rolling_buffer)
                    
                    if len(latest_audio) > 0:
                        self.active_audio_buffer = np.append(self.active_audio_buffer, latest_audio)

                        # Limit active buffer size (max 30 seconds)
                        max_active_buffer = int(self.RATE * self.max_active_buffer_duration)
                        # print(f"[ACTIVE BUFFER vs MAX] {len(self.active_audio_buffer)} length vs {max_active_buffer} max active buffer size")
                        
                        # TODO: this situation heapend when every time it transcribes different text based on bad audio
                        # so it keeps adding to active buffer and never clears it
                        # ideally this should not happen, but if it does, we should manage it better
                        # if active buffer is too long, trim it to last 24 seconds
                        # this is a temporary fix, ideally we should have better adaptive management
                        # though below code works fine but need more management
                        if len(self.active_audio_buffer) > max_active_buffer:
                            self._debug_print(f"\033[93m[ACTIVE BUFFER TRIMMED] Active buffer full, i must use better adaptive management\033[0m")
                            four_fifth = (max_active_buffer // 5) * 4
                            self.active_audio_buffer = self.active_audio_buffer[-four_fifth:]

                # Process during ongoing audio (periodic transcription)
                min_processing_buffer = int(self.RATE * 0.8)
                should_process = (
                    (current_time - self.last_transcription_time >= self.processing_interval) and 
                    len(self.active_audio_buffer) > min_processing_buffer
                )
                
                # Force segmentation if sentence is too long
                # TODO: Make this more adaptive and pass to llm each 2 minute for special long sentences
                should_force = self._should_force_segmentation()
                
                if should_process or should_force:
                    self._process_sentence_segment(self.active_audio_buffer)
                    self.last_transcription_time = time.time()
                    
                    if should_force:
                        self._debug_print("\033[91m\n[FORCE SEGMENTATION - Time limit reached]\033[0m")
                        self._finalize_sentence()
                
                    if max_active_buffer:
                        self._debug_print(f"[ACTIVE BUFFER status] {len(self.active_audio_buffer)/self.RATE:.1f}s active audio buffer")
                    if self.sentence_start_time:
                        self._debug_print(f"[SENTENCE TIMING] {time.time() - self.sentence_start_time:.1f}s")

                # Adaptive sleep
                elapsed = time.time() - start_time
                sleep_time = max(0.01, 0.05 - elapsed)
                time.sleep(sleep_time)
                
            except queue.Empty:
                time.sleep(0.05)
            except Exception as e:
                self._debug_print(f"Error in audio processing thread: {e}")
                time.sleep(0.1)
    
    def _process_sentence_segment(self, audio_buffer):
        """
        Process a sentence segment with enhanced dual buffer system and simplified timer
        """
        min_segment_size = int(self.RATE * 0.3)
        if len(audio_buffer) < min_segment_size:
            return
        
        # ------------------------------------- should be tested more -------------------------------------
        try:
            # #TODO: maybe we need a better adapdatable advance forein language detector precydure
            # based on procces some segemnts to detern language
            # Detect language using Whisper's built-in language detection (if model supports it)
            if len(audio_buffer) > int(self.RATE * 1.0):  # Only if we have at least 1 second of audio
                try:
                    # Prepare audio for language detection using Whisper's preprocessing
                    audio_padded = whisper.pad_or_trim(audio_buffer)
                    
                    # Make log-Mel spectrogram and move to the same device as the model
                    mel = whisper.log_mel_spectrogram(audio_padded, n_mels=self.model.dims.n_mels).to(self.model.device)
                    
                    # Detect the spoken language
                    _, probs = self.model.detect_language(mel)
                    detected_language = max(probs, key=probs.get)
                    # if detected_language == "nn":
                    #     print(f"[LANGUAGE DETECTION] Detected Norwegian Nynorsk (nn), skipping segment")
                    #     return  # Skip if detected as Norwegian Nynorsk (common false positive) == means speaker is not speaking
                    self._debug_print(f"[LANGUAGE DETECTION] Detected language: {detected_language} (confidence: {probs[detected_language]:.3f})")
                    self.last_language = detected_language

                    # If detected language doesn't match expected language, handle accordingly
                    if detected_language != self.language or detected_language == "nn":
                        self._debug_print(f"[LANGUAGE MISMATCH] Expected: {self.language}, Detected: {detected_language}")
                        
                        # Increment rejection counter for language mismatch
                        self.foreign_language_rejection_count += 1
                        self.last_rejection_time = time.time()
                        
                        self._debug_print(f"[LANGUAGE REJECTION COUNT] {self.foreign_language_rejection_count}/{self.max_foreign_rejections}")
                    
                        # If we've had too many language mismatches, reset
                        if self.foreign_language_rejection_count >= self.max_foreign_rejections:
                            self._reset_sentence_state("Maximum foreign language detections reached or detected silence (nn)")
                    
                        return  # Don't process this transcription further
                    
                    # return  # Don't process this transcription further
                except Exception as e:
                    # Handle models that don't support language detection (e.g., English-only models)
                    self._debug_print(f"[LANGUAGE DETECTION] Model doesn't support language detection: {e}")
                    self._debug_print(f"[LANGUAGE DETECTION] Skipping language validation for this model")
                    
            # ------------------------------------- should be tested more --------------------------------------------------------------
                    
            # Transcribe using OpenAI Whisper with word-level timestamps
            result = self.model.transcribe(
                audio_buffer, 
                fp16=False, 
                language=self.language,
                word_timestamps=True,  # Enable word-level timestamps!
                suppress_tokens=None  # Don't suppress any tokens, including special ones
            )
            # #TODO: idea : there is a methode also focused on silence detection to make the most of it but need a bit more resources
            # Check last quarter of audio buffer for silence before full transcription
            # quarter_samples = len(audio_buffer) // 4
            # if quarter_samples > 0:
            #     last_quarter = audio_buffer[-quarter_samples:]
            #     # Quick silence check on last quarter
            #     test_res = self.model.transcribe(
            #         last_quarter, 
            #         fp16=False, 
            #         language="en",
            #         suppress_tokens=None
            #     )
            #
            #     # If last quarter is silent/empty, transcribe full buffer
            #     if not test_res["text"].strip():
            #         self._debug_print("[SILENCE DETECTED] Last quarter silent, processing full buffer")
            #     else:
            #         self._debug_print(f"[ACTIVITY DETECTED] Last quarter: '{test_res['text'].strip()}'")

            
            new_text = result["text"].strip()
            # self._debug_print(f"\n[TRANSCRIPTION pure RESULT] -{new_text}-")
            
            if not new_text:
                # only empty transcription found mode
                self.empty_transcribe_rejection_count += 1
                self._debug_print(f"[EMPTY TRANSCRIPTION FOUND] Count: {self.empty_transcribe_rejection_count}/{self.max_empty_rejections}")
                if self.empty_transcribe_rejection_count >= self.max_empty_rejections:
                    self.empty_transcribe_rejection_count = 0
                    self._reset_sentence_state("Maximum empty transcriptions reached")
                return
            
            
            print(f"\033[91m\n[TRANSCRIPTION] {new_text}\033[0m")
            
            # Extract word-level timestamps
            word_timestamps = self._extract_word_timestamps(result)
            
            # Check for foreign language or audio annotations
            is_foreign, is_audio_annotation, rejection_reason = self._detect_foreign_language_or_annotation(new_text)
            
            if is_foreign or is_audio_annotation:
                self._debug_print(f"[REJECTED] {rejection_reason}")
                
                # Increment rejection counter and update timestamp
                self.foreign_language_rejection_count += 1
                self.last_rejection_time = time.time()
                
                self._debug_print(f"[REJECTION COUNT] {self.foreign_language_rejection_count}/{self.max_foreign_rejections}")
                
                # If we've had too many rejections, reset immediately
                if self.foreign_language_rejection_count >= self.max_foreign_rejections:
                    self._reset_sentence_state("Maximum foreign language rejections reached")
                
                return  # Don't process this transcription further
            
            # Reset foreign language rejection counter on successful English transcription
            if self.foreign_language_rejection_count > 0:
                self._debug_print(f"[ENGLISH DETECTED] Resetting foreign language rejection counter")
                self.foreign_language_rejection_count = 0
                self.last_rejection_time = None
            
            # Process pattern detection and buffer management
            self._process_transcription_pattern(new_text, word_timestamps)
            
            # ALWAYS replace the last transcription with the new one
            self.last_transcription = new_text
            
            # Check if we have new words
            if self._has_new_words(new_text):
                self._debug_print(f"[NEW WORDS] Detected new words in transcription")
            
            # Detect sentence ending for logging purposes (no timer activation)
            # is_sentence_end, is_pause = self._detect_sentence_end(new_text)
            # if is_sentence_end:
            #     self._debug_print(f"[SENTENCE END DETECTED] But relying only on stable buffer timing")
            
        except Exception as e:
            self._debug_print(f"Error in sentence segment processing: {e}")
    
    def print_session_summary(self):
        """
        Print a summary of the transcription session
        This method can be called independently or automatically on stop
        """
        if self.completed_sentences:
            print(f"\n[SESSION SUMMARY] Processed {len(self.completed_sentences)} sentences:")
            for i, sentence in enumerate(self.completed_sentences, 1):
                print(f"  {i}. [{sentence['timestamp']}] {sentence['text']}")
        else:
            print("\n[SESSION SUMMARY] No completed sentences were transcribed.")
    
    def get_all_transcribed_text(self):
        """
        Get all transcribed text as a single list of strings
        Returns a list of all completed sentence texts
        """
        return [sentence['text'] for sentence in self.completed_sentences]
    
    def get_completed_sentences(self):
        """
        Get all completed sentences with metadata
        Returns the complete list of sentence dictionaries with timestamp and duration
        """
        return self.completed_sentences.copy()
    
    def _send_to_llm(self, text):
        """Send completed sentence to LLM for processing"""
        print(f"\033[94m\n[LLM INPUT]: {text}\033[0m")
        
        # If a callback is registered, use it; otherwise use default behavior
        if self._def_callback:
            try:
                self._def_callback(text)
            except Exception as e:
                print(f"\033[91m[LLM CALLBACK ERROR]: {e}\033[0m")
                # Fall back to default behavior on error
                print(f"[LLM FALLBACK]: Using default behavior due to callback error")
        else:
            # Default behavior - just display the text
            # Users can implement their own LLM integration by setting a callback
            print(f"\033[93m As user didnt set a callback, it will just print the text , you can also get text using get_all_transcribed_text() and get_completed_sentences() methods\033[0m")

    def start_streaming(self):
        """Start streaming from microphone and transcribing"""
        self.is_recording = True
        self._is_paused = False  # Initialize pause state
        self.rolling_buffer = np.array([], dtype=np.float32)
        
        # Initialize enhanced dual buffer system (text only)
        self.stable_text_buffer = ""
        self.active_audio_buffer = np.array([], dtype=np.float32)
        
        self.last_transcription = ""
        self.completed_sentences = []
        self.sentence_start_time = None
        self.last_stable_buffer_update = None
        self.last_word_count = 0
        self.audio_queue = queue.Queue()
        self.last_transcription_time = time.time()
        
        # Reset pattern detection state
        self.transcription_history = []
        # self.temp_timestamps_dict = {}
        self.duplicate_detection_state = "waiting"
        self.confirmed_pattern = ""
        
        # Reset foreign language detection state
        self.foreign_language_rejection_count = 0
        self.last_rejection_time = None
        
        # Reset summary tracking
        self._summary_printed = False
        
        # Open PyAudio stream
        try:
            stream_kwargs = {
                'format': self.FORMAT,
                'channels': self.CHANNELS,
                'rate': self.RATE,
                'input': True,
                'frames_per_buffer': self.CHUNK,
                'stream_callback': self._audio_callback
            }
            
            # Add input device if one was selected
            if self._selected_input_device is not None:
                stream_kwargs['input_device_index'] = self._selected_input_device
                device_info = self.get_current_input_device()
                if device_info:
                    print(f"[AUDIO] Using input device: {device_info['name']}")
            
            self.stream = self.p.open(**stream_kwargs)
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.is_recording = False
            return False
        
        # Start processing thread
        try:
            self.process_thread = threading.Thread(target=self._process_audio)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            print("Listening with Fixed Dual Buffer System...")
            self._debug_print("- Text-only stable buffer (no audio buffer)")
            self._debug_print("- Simplified timer based ONLY on stable buffer updates")
            self._debug_print("- Timer RESETS when new content is committed to stable buffer")
            self._debug_print("- FIXED: Stable buffer preserved during noise rejections")
            self._debug_print("- Intelligent pattern detection with 3-way confirmation")
            self._debug_print("- Prevents exponential reprocessing")
            self._debug_print("- Foreign language detection and reset")
            self._debug_print("- LLM sending only when 10s passed since last stable buffer update")
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
                if self.last_language == self.language:
                    self._process_sentence_segment(self.active_audio_buffer)
                self._debug_print("\n[PROCESSING FINAL SEGMENT on stopping]")
            except Exception as e:
                self._debug_print(f"Error processing final segment: {e}")
        
        # Finalize any pending sentence
        # if self.last_transcription or self.stable_text_buffer:
        #     self._debug_print("\n[FINALIZING PENDING SENTENCE]")
        #     self._finalize_sentence()
        
        # Finalize just stable pending sentence
        if self.stable_text_buffer:
            self._debug_print("\n[FINALIZING STABLE SENTENCE]")
            self._finalize_sentence(final_text=self.stable_text_buffer)
        
        # Wait for processing thread to finish
        if self.process_thread and self.process_thread.is_alive():
            try:
                self.process_thread.join(timeout=2)
            except Exception as e:
                print(f"Error joining processing thread: {e}")
        
        # Print summary using the dedicated method
        self.print_session_summary()
        self._summary_printed = True  # Mark that summary has been printed
    
    def close(self):
        """Clean up resources and print final summary"""
        self.stop_streaming()
        
        # Print summary if it hasn't been printed yet (safety measure)
        if self.completed_sentences and hasattr(self, '_summary_printed') and not self._summary_printed:
            self.print_session_summary()
        
        try:
            self.p.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup happens even if close() isn't called"""
        try:
            if hasattr(self, 'completed_sentences') and self.completed_sentences:
                if not hasattr(self, '_summary_printed') or not self._summary_printed:
                    self.print_session_summary()
        except:
            pass  # Ignore errors in destructor


