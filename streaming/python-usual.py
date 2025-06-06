import whisper
import pyaudio
import numpy as np
import threading
import time
from pynput import keyboard

class WhisperStreamingTranscriber:
    def __init__(self, model_name="base.en"):
        """
        Initialize the transcriber with a specific Whisper model.
        
        Args:
            model_name: Whisper model name (tiny.en, base.en, small.en, medium.en, large)
                        The smaller models are faster but less accurate.
        """
        print(f"Loading Whisper model: {model_name}...")
        self.model = whisper.load_model(model_name)
        print("Model loaded!")
        
        # Audio recording parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        
        # Processing parameters 
        self.audio_buffer = []
        self.is_recording = False
        self.transcription = ""
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
            # If we have new audio data to process
            if len(self.audio_buffer) > self.last_processed:
                # Combine all new chunks
                new_chunks = self.audio_buffer[self.last_processed:]
                audio_data = np.hstack(new_chunks).astype(np.float32) / 32768.0  # Convert to float32
                
                # Process with Whisper
                try:
                    result = self.model.transcribe(audio_data, fp16=False, language="en")
                    new_text = result["text"].strip()
                    
                    if new_text:
                        print(new_text)
                
                except Exception as e:
                    print(f"Error in transcription: {e}")
                
                # Update the last processed index
                self.last_processed = len(self.audio_buffer)
            
            # Sleep a bit to avoid using too much CPU
            time.sleep(1)  # Process every second
    
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
        
        # Wait for the processing thread to finish
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=2)
    
    def close(self):
        """Clean up resources"""
        self.stop_streaming()
        self.p.terminate()


if __name__ == "__main__":
    # Initialize the transcriber with the "base.en" model
    # You can change this to "tiny.en" for faster performance or "medium.en" for better accuracy
    transcriber = WhisperStreamingTranscriber(model_name="base.en")
    
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