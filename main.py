import os
import json
from vosk import Model, KaldiRecognizer

def main():
    # Model setting
    # model_path = "models/vosk-model-en-us-0.22-lgraph"
    model_path = "models/vosk-model-en-us-0.22-lgraph"
    if not os.path.exists(model_path):
        print("Model not found. Please download the model and make the path.")
        return

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    # Read audio file
    audio_file_path = "mine.wav"  # Update the audio file path
    if not os.path.exists(audio_file_path):
        print("Audio file not found. Please check the path.")
        return

    with open(audio_file_path, "rb") as audio_file:
        audio_data = audio_file.read()

    if recognizer.AcceptWaveform(audio_data):
        result = recognizer.Result()
        text = json.loads(result).get('text', '')
        if text:
            print("Transcription: " + text)
            # Save
            with open("output.txt", "a") as f:
                f.write(text + "\n")

if __name__ == "__main__":
    main()
