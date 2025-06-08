from faster_whisper import WhisperModel

# model = WhisperModel("base.en")
# Run on GPU with FP16
# model = WhisperModel("base.en", device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel("base.en", device="cpu", compute_type="int8")

segments, info = model.transcribe("/home/flayo/Desktop/flayo -zone/whisper.cpp/samples/jfk.mp3")
# segments, info = model.transcribe("/home/flayo/Desktop/flayo -zone/whisper.cpp/samples/jfk.wav")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))