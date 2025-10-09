# real_audio_test.py
from whisperpipe import pipeStream
from evaluation.metrics import StreamingMetrics
import time

# Create transcriber with evaluation enabled
transcriber = pipeStream(
    model_name="base",
    language="en",
    enable_evaluation=True,
    finalization_delay=5.0,  # Shorter for testing
    debug_mode=False
)

print("Speak into your microphone for 10 seconds...")
print("Say something like: 'The quick brown fox jumps over the lazy dog'")

transcriber.start_streaming()
time.sleep(10)  # Record for 10 seconds
transcriber.stop_streaming()

# Get results
if transcriber.logger:
    history = transcriber.logger.get_transcription_history()
    commits = transcriber.logger.get_stable_commits()
    
    print("\n" + "="*60)
    print("TRANSCRIPTION HISTORY:")
    for i, text in enumerate(history):
        print(f"  {i+1}. {text}")
    
    print("\n" + "="*60)
    print("STABLE COMMITS:")
    for i, commit in enumerate(commits):
        print(f"  {i+1}. [{commit['commit_latency']*1000:.2f}ms] {commit['text']}")
    
    # Calculate metrics
    if history:
        final_output = history[-1]
        reference = "the quick brown fox jumps over the lazy dog"  # What you said
        
        report = StreamingMetrics.generate_report(
            reference=reference,
            hypothesis=final_output,
            transcription_history=history,
            stable_commits=commits
        )
        
        StreamingMetrics.print_report(report)

transcriber.close()