# debug_logger.py
from whisperpipe import pipeStream
import time

transcriber = pipeStream(
    model_name="base",
    language="en",
    enable_evaluation=True,
    finalization_delay=5.0,
    debug_mode=True  # Enable to see what's happening
)

print("Speak for 10 seconds...")
transcriber.start_streaming()
time.sleep(10)
transcriber.stop_streaming()

if transcriber.logger:
    history = transcriber.logger.get_transcription_history()
    
    print(f"\n{'='*60}")
    print(f"Total transcriptions logged: {len(history)}")
    print(f"{'='*60}")
    
    for i, text in enumerate(history, 1):
        print(f"{i:3d}. '{text}'")
    
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")
    
    # Check for duplicates
    unique = list(set(history))
    print(f"Unique transcriptions: {len(unique)}")
    print(f"Duplicate rate: {(len(history) - len(unique)) / len(history) * 100:.1f}%")

transcriber.close()