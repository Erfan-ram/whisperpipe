import sounddevice as sd
import numpy as np
import wave
from pynput import keyboard
import whisper

# Configuration
FILENAME = "recorded.wav"
SAMPLERATE = 16000
CHANNELS = 1
DTYPE = 'int16'

# State
is_recording = [False]
audio_data = []
stream = None
pressed_keys = set()

def list_microphones():
    devices = sd.query_devices()
    inputs = [dev for dev in devices if dev['max_input_channels'] > 0]
    print("Available input devices:\n")
    for idx, dev in enumerate(inputs):
        print(f"{idx}: {dev['name']}")
    index = int(input("\nSelect microphone index: "))
    return inputs[index]['name']

def record_callback(indata, frames, time, status):
    if is_recording[0]:
        audio_data.append(indata.copy())

def start_stream(device_name):
    global stream
    stream = sd.InputStream(
        device=device_name,
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=record_callback
    )
    stream.start()
    print(f"\nRecording with: {device_name}")
    print("Press Ctrl + B to start recording")
    print("Press F to stop recording")

def stop_stream():
    if stream:
        stream.stop()
        stream.close()

def save_audio(filename):
    all_data = np.concatenate(audio_data, axis=0)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLERATE)
        wf.writeframes(all_data.tobytes())

def transcribe_with_whisper(filename):
    print("\nTranscribing with Whisper...")
    model = whisper.load_model("tiny.en")
    # model = whisper.load_model("base.en")
    result = model.transcribe(filename)
    print("\n--- Transcription ---")
    print(result["text"])

def on_press(key):
    try:
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            pressed_keys.add('ctrl')
        elif key.char and key.char.lower() == 'b':
            pressed_keys.add('b')
        elif key.char and key.char.lower() == 'f':
            if is_recording[0]:
                print("Recording stopped.")
                is_recording[0] = False
                return False  # Stop listener

        if 'ctrl' in pressed_keys and 'b' in pressed_keys and not is_recording[0]:
            print("Recording started.")
            is_recording[0] = True

    except AttributeError:
        pass  # Ignore special keys without .char

def on_release(key):
    try:
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            pressed_keys.discard('ctrl')
        elif key.char and key.char.lower() == 'b':
            pressed_keys.discard('b')
    except AttributeError:
        pass

if __name__ == "__main__":
    selected_device = list_microphones()
    start_stream(selected_device)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    stop_stream()
    save_audio(FILENAME)
    transcribe_with_whisper(FILENAME)
