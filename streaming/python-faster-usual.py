import numpy as np
import pyaudio
import threading
import time
import sys
import os
import signal
import queue
from collections import deque
from pynput import keyboard
from faster_whisper import WhisperModel

class WhisperStreamingTranscriber:
    def __init__(self, model_name="base.en", buffer_duration_seconds=5):
        """
        Initialize the transcriber with a specific Whisper model using faster_whisper.
        
        Args:
            model_name: Whisper model name (tiny.en, base.en, small.en, medium.en, large-v2)
                        The smaller models are faster but less accurate.
            buffer_duration_seconds: Time window in seconds to hold audio for processing
        """
        print(f"Loading Whisper model: {model_name}...")
        try:
            # Initialize the WhisperModel from faster_whisper with INT8 compute type
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
            print("Model loaded!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Audio recording parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        
        # Calculate buffer parameters
        self.buffer_duration_seconds = buffer_duration_seconds
        self.buffer_size = int(self.RATE * buffer_duration_seconds)
        self.overlap_ratio = 0.5  # 50% overlap for continuous transcription
        
        # Processing parameters
        self.audio_queue = queue.Queue()
        self.rolling_buffer = np.array([], dtype=np.float32)
        self.is_recording = False
        self.transcription_history = deque(maxlen=10)  # Store last 10 transcriptions
        self.last_context = ""  # Store context from previous segments
        self.silence_threshold = 0.01
        self.silence_counter = 0
        self.accumulated_text = ""  # Accumulated text for the LLM
        
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
    
    def _is_silent(self, audio_data, threshold=0.01):
        """Check if audio segment is silent"""
        return np.abs(audio_data).mean() < threshold
    
    def _detect_sentence_end(self, text):
        """Check if text appears to end a complete thought/sentence"""
        end_markers = ['.', '?', '!', '\n']
        return any(text.rstrip().endswith(marker) for marker in end_markers)
    
    def _process_audio(self):
        """Process audio in the background - optimized for real-time performance"""
        last_transcription_time = time.time()
        last_sentence_time = time.time()
        processing_interval = 0.3  # Process every 300ms for low latency
        sentence_timeout = 2.0  # Force update after 2 seconds of continuous speech
        
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
                
                # Maintain buffer size to prevent memory growth
                with self.lock:
                    if len(self.rolling_buffer) > self.buffer_size:
                        # Keep buffer at fixed size, dropping oldest data
                        overlap_samples = int(self.buffer_size * self.overlap_ratio)
                        self.rolling_buffer = self.rolling_buffer[-self.buffer_size:]
                
                # Determine if we should process (based on time or queue size)
                current_time = time.time()
                should_process = (current_time - last_transcription_time >= processing_interval) and len(self.rolling_buffer) > self.RATE * 0.5
                
                # Check if audio has enough speech content to process
                if should_process and len(self.rolling_buffer) > 0:
                    with self.lock:
                        buffer_copy = np.copy(self.rolling_buffer)
                    
                    # Check if audio contains speech
                    if self._is_silent(buffer_copy, self.silence_threshold):
                        self.silence_counter += 1
                        if self.silence_counter >= 10:  # After sustained silence, send accumulated text to LLM
                            if self.accumulated_text and len(self.accumulated_text) > 5:  # If we have meaningful text
                                self._send_to_llm(self.accumulated_text)
                                self.accumulated_text = ""
                            self.silence_counter = 0
                            
                        time.sleep(0.05)  # Sleep slightly to reduce CPU
                        continue
                    else:
                        self.silence_counter = 0
                        
                    # Process with faster_whisper
                    try:
                        segments, info = self.model.transcribe(
                            buffer_copy,
                            beam_size=3,  # Lower beam size for speed
                            language="en",
                            vad_filter=True,  # Filter out non-speech
                            vad_parameters=dict(min_silence_duration_ms=300),
                            condition_on_previous_text=True,  # Use context from previous segments
                            initial_prompt=self.last_context if self.last_context else None
                        )
                        
                        # Extract text from segments
                        new_text = []
                        for segment in segments:
                            if segment.text.strip():
                                new_text.append(segment.text.strip())
                        
                        if new_text:
                            full_text = " ".join(new_text)
                            
                            # Check if this is new text, not just a duplicate
                            is_new = not any(self._text_similarity(full_text, prev) > 0.8 for prev in self.transcription_history)
                            
                            if is_new:
                                print(f"\n[{time.strftime('%H:%M:%S')}] {full_text}")
                                self.transcription_history.append(full_text)
                                
                                # Update context with current text for the next iteration
                                self.last_context = full_text
                                
                                # Add to accumulated text for LLM
                                self.accumulated_text += " " + full_text
                                
                                # Check if we should send to the LLM
                                if self._detect_sentence_end(full_text) or (current_time - last_sentence_time > sentence_timeout):
                                    # Only send meaningful accumulated text
                                    if len(self.accumulated_text.strip()) > 10:
                                        self._send_to_llm(self.accumulated_text)
                                        self.accumulated_text = ""
                                    last_sentence_time = current_time
                    
                    except Exception as e:
                        print(f"Error in transcription: {e}")
                    
                    # Update timing for next processing
                    last_transcription_time = time.time()
                
                # Adaptive sleep based on processing load
                elapsed = time.time() - start_time
                sleep_time = max(0.01, processing_interval - elapsed)  # At least 10ms, but try to maintain interval
                time.sleep(sleep_time)
                
            except queue.Empty:
                time.sleep(0.05)  # Nothing in queue, sleep briefly
            except Exception as e:
                print(f"Error in audio processing thread: {e}")
                time.sleep(0.1)  # Wait before trying again

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
        self.last_context = ""
        self.accumulated_text = ""
        
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
        
        # Wait for the processing thread to finish
        if self.process_thread and self.process_thread.is_alive():
            try:
                self.process_thread.join(timeout=2)
            except Exception as e:
                print(f"Error joining processing thread: {e}")
                
        # Send any remaining accumulated text
        if self.accumulated_text and len(self.accumulated_text) > 5:
            self._send_to_llm(self.accumulated_text)
    
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


if __name__ == "__main__":
    # Register signal handlers for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize the transcriber with the "base.en" model
    try:
        transcriber = WhisperStreamingTranscriber(model_name="base.en")
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