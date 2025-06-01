import sounddevice as sd
import numpy as np
import wave
from pynput import keyboard
import whisper

# Settings
FILENAME = "recorded.wav"
SAMPLERATE = 16000  # Whisper works best at 16kHz
CHANNELS = 1
DTYPE = 'int16'

is_recording = [False]
audio_data = []

def record_callback(indata, frames, time, status):
    if is_recording[0]:
        audio_data.append(indata.copy())

def start_recording():
    print("Hold SPACE to record...")
    with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype=DTYPE, callback=record_callback):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

def on_press(key):
    if key == keyboard.Key.space and not is_recording[0]:
        print("Recording...")
        is_recording[0] = True

def on_release(key):
    if key == keyboard.Key.space and is_recording[0]:
        print("Recording stopped.")
        is_recording[0] = False
        return False  # Stop listener

def save_audio(filename):
    all_data = np.concatenate(audio_data, axis=0)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLERATE)
        wf.writeframes(all_data.tobytes())

def transcribe_with_whisper(filename):
    model = whisper.load_model("tiny.en")  # Load the Whisper model
    print("Transcribing audio...")
    result = model.transcribe(filename)
    print("\n--- Transcription ---")
    print(result["text"])

if __name__ == "__main__":
    start_recording()
    save_audio(FILENAME)
    transcribe_with_whisper(FILENAME)
