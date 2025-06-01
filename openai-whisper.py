import sounddevice as sd
import numpy as np
import wave
from pynput import keyboard
import whisper
import tempfile
import os

class VoiceRecorder:
    def __init__(self, samplerate=16000, channels=1, dtype='int16'):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.device = None
        self.stream = None
        self.audio_data = []
        self.is_recording = False
        self.pressed_keys = set()
        # self.model = whisper.load_model("tiny.en")
        self.model = whisper.load_model("tiny.en" ,device="cpu")
        # self.model = whisper.load_model("base.en")
        # self.model = whisper.load_model("base.fa")

    def select_microphone(self):
        devices = sd.query_devices()
        inputs = [dev for dev in devices if dev['max_input_channels'] > 0]
        print("Available input devices:\n")
        for idx, dev in enumerate(inputs):
            print(f"{idx}: {dev['name']}")
        index = int(input("\nSelect microphone index: "))
        self.device = inputs[index]['name']

    def _record_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def start_stream(self):
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._record_callback
        )
        self.stream.start()
        print(f"\nRecording with: {self.device}")
        print("Press Ctrl + B to start, press F to finish.")

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def save_audio(self, filename):
        all_data = np.concatenate(self.audio_data, axis=0)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.samplerate)
            wf.writeframes(all_data.tobytes())
        print(f"Audio saved to {filename}")
        return filename

    def transcribe_audio(self, save=False):
        all_data = np.concatenate(self.audio_data, axis=0)
        with tempfile.NamedTemporaryFile(delete=not save, suffix=".wav") as temp:
            with wave.open(temp.name, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.samplerate)
                wf.writeframes(all_data.tobytes())
            print("Transcribing...")
            result = self.model.transcribe(temp.name)
            print("\n--- Transcription ---\n")
            print(result["text"])
            return result["text"]

    def handle_keypress(self, key):
        try:
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.pressed_keys.add('ctrl')
            elif key.char and key.char.lower() == 'b':
                self.pressed_keys.add('b')
            elif key.char and key.char.lower() == 'f':
                if self.is_recording:
                    print("Recording stopped.")
                    self.is_recording = False
                    return False  # Stop listener

            if 'ctrl' in self.pressed_keys and 'b' in self.pressed_keys and not self.is_recording:
                print("Recording started.")
                self.is_recording = True

        except AttributeError:
            pass

    def handle_keyrelease(self, key):
        try:
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.pressed_keys.discard('ctrl')
            elif key.char and key.char.lower() == 'b':
                self.pressed_keys.discard('b')
        except AttributeError:
            pass

    def run(self):
        self.select_microphone()
        self.start_stream()

        with keyboard.Listener(on_press=self.handle_keypress, on_release=self.handle_keyrelease) as listener:
            listener.join()

        self.stop_stream()

        # Ask whether to save audio
        # should_save = input("Do you want to save the audio file? (y/N): ").strip().lower() == 'y'
        # if should_save:
        #     filename = input("Enter filename (e.g., 'output.wav'): ").strip()
        #     self.save_audio(filename)
        # self.transcribe_audio(save=should_save)
        self.transcribe_audio()

if __name__ == "__main__":
    vr = VoiceRecorder()
    vr.run()
