# Example: evaluate_with_real_audio.py
import soundfile as sf
import numpy as np
from evaluation.whisper_baseline import WhisperBaseline
from evaluation.metrics import calculate_metrics_summary
import threading
import time
import queue
from whisperpipe.core import pipeStream

# Load audio file
audio, sr = sf.read('evaluation/sample.wav')
if audio.ndim == 2:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)


# Resample to 16kHz if needed
if sr != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# Split into 1-second chunks
chunk_size = 16000  # 1 second at 16kHz
chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

# Ground truth transcription
reference = "the stale smell of old beer lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle taste fine with ham tacos al pastor are my favorite a zestful food is the hot cross bun"

# --- Test whisperpipe ---
print("\n--- Testing whisperpipe ---")

# Initialize pipeStream
pipe = pipeStream(model_name="base", language="en", debug_mode=False)

# Manually initialize pipeStream state without starting microphone
# This is a workaround to test pipeStream with a file instead of a microphone
pipe.is_recording = True
pipe._is_paused = False
pipe.rolling_buffer = np.array([], dtype=np.float32)
pipe.stable_text_buffer = ""
pipe.active_audio_buffer = np.array([], dtype=np.float32)
pipe.last_transcription = ""
pipe.completed_sentences = []
pipe.sentence_start_time = None
pipe.last_stable_buffer_update = None
pipe.last_word_count = 0
pipe.audio_queue = queue.Queue()
pipe.last_transcription_time = time.time()
pipe.transcription_history = []
pipe.duplicate_detection_state = "waiting"
pipe.confirmed_pattern = ""
pipe.foreign_language_rejection_count = 0
pipe.last_rejection_time = None
pipe._summary_printed = False
pipe.intermediate_outputs = []

# Start processing thread
pipe.process_thread = threading.Thread(target=pipe._process_audio)
pipe.process_thread.daemon = True
pipe.process_thread.start()

# Re-chunk audio for a more realistic real-time simulation
realtime_chunk_size = 1024
micro_chunks = [audio[i:i+realtime_chunk_size] for i in range(0, len(audio), realtime_chunk_size)]
chunk_duration = realtime_chunk_size / 16000.0

# Feed audio micro-chunks to simulate a real-time stream
print("Feeding audio micro-chunks to whisperpipe...")
for micro_chunk in micro_chunks:
    pipe.audio_queue.put(micro_chunk)
    time.sleep(chunk_duration)

print("All audio chunks fed. Waiting for finalization...")
# Wait for finalization delay after last chunk is fed
time.sleep(pipe.finalization_delay + 2) # a bit more than the delay

# Stop the stream
pipe.stop_streaming()

# Get final transcription
pipe_final_text = " ".join(pipe.get_all_transcribed_text())

# Get intermediate results for metrics
pipe_intermediates_data = pipe.get_intermediate_outputs()
pipe_intermediates = [o['text'] for o in pipe_intermediates_data]
pipe_times = [o['processing_time'] for o in pipe_intermediates_data]

# Calculate metrics for whisperpipe
pipe_metrics = calculate_metrics_summary(reference, pipe_final_text, pipe_intermediates, pipe_times)
print("\n--- whisperpipe Results ---")
print(f"Final Transcription: {pipe_final_text}")
print(f"WER: {pipe_metrics['wer']:.2f}%")
print(f"SI: {pipe_metrics['stability_index']:.2f}%")
print(f"Latency: {pipe_metrics['avg_latency_ms']:.2f} ms")

pipe.close()
