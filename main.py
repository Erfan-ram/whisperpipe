import os
import json
import pyaudio
from vosk import Model, KaldiRecognizer

def main():
    #Model setting
    model_path = "/path/to/vosk-model" #Update the model path
    if not os.path.exists(model_path):
        print("Model not found. Please download the model and make the path.")
        return

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    #mic
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()

    print("Please speak...")

    while True:
        data = stream.read(4000)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            text = json.loads(result).get('text', '')
            if text:
                print("you said: " + text)
                #Save
                with open("output.txt", "a") as f:
                    f.write(text + "\n")

if __name__ == "__main__":
    main()
