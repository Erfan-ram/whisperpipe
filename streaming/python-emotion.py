import whisper
import numpy as np
import pyaudio
import threading
import time
from pynput import keyboard

class WhisperStreamingTranscriberWithSpecials:
    def __init__(self, model_name="base.en"):
        """Initialize Whisper with special tokens enabled"""
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
        """Process audio in the background with special tokens enabled"""
        while self.is_recording:
            if len(self.audio_buffer) > self.last_processed:
                new_chunks = self.audio_buffer[self.last_processed:]
                audio_data = np.hstack(new_chunks).astype(np.float32) / 32768.0
                
                try:
                    # KEY DIFFERENCE: Enable special tokens to detect laughter and silence
                    result = self.model.transcribe(
                        audio_data, 
                        fp16=False, 
                        language="en",
                        # This makes Whisper include special tokens like [laughter] and [silence]
                        suppress_tokens=None  # Don't suppress any tokens, including special ones
                    )
                    
                    new_text = result["text"].strip()
                    
                    if new_text:
                        print(new_text)
                
                except Exception as e:
                    print(f"Error in transcription: {e}")
                
                self.last_processed = len(self.audio_buffer)
            
            time.sleep(1)
    
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
    
    # ... rest of the code remains the same


# Main execution
if __name__ == "__main__":
    transcriber = WhisperStreamingTranscriberWithSpecials(model_name="base.en")
    
    def on_press(key):
        try:
            if key.char == 'q':
                print("\nStopping transcription...")
                return False
        except AttributeError:
            pass
    
    try:
        transcriber.start_streaming()
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        transcriber.stop_streaming()
        transcriber.close()
        print("Transcription ended.")