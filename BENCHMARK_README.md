# Benchmark Instructions

## Overview

`Benchmark.py` compares the enhanced whisperpipe implementation with a baseline Whisper streaming approach. It measures:
- **WER (Word Error Rate)**: Transcription accuracy
- **SI (Stability Index)**: Output consistency across intermediate results
- **Latency**: Processing time per transcription
- **Total Processing Time**: Overall time excluding finalization delay

## Key Differences in Testing Methodology

### whisperpipe (Enhanced Streaming)
- Feeds audio in real-time micro-chunks (1024 samples)
- Uses dual-buffer architecture with stability mechanisms
- Processing time measured **excluding** the finalization delay
- Finalization delay is user-configurable waiting time, not actual processing

### Baseline Whisper (Simulated Streaming)
- Transcribes progressively larger chunks: 0-1s, 0-2s, 0-3s, ..., 0-end
- Simulates real streaming where the model reprocesses from the beginning
- Each transcription is independent (no sliding window)
- More realistic comparison to show the benefit of dual-buffer architecture

## Prerequisites

```bash
# Install required packages
pip install soundfile librosa numpy openai-whisper torch

# Optional: For better audio handling
pip install scipy
```

## Prepare Test Audio

### Option 1: Use Real Speech Audio (Recommended)
Place your audio file at `evaluation/sample.wav` with the corresponding reference transcription.

### Option 2: Generate Synthetic Audio (For Testing)
```bash
python generate_sample_audio.py
```

**Note:** Synthetic audio won't produce meaningful transcriptions. Use real speech audio for actual benchmarking.

## Run Benchmark

```bash
python Benchmark.py
```

## Expected Output

```
================================================================================
TEST 1: whisperpipe (Enhanced Streaming)
================================================================================
Loading Whisper model: base
Using device: cuda
Feeding audio to whisperpipe...
Audio feeding complete. Processing time: 30.00s
Waiting for finalization (10s)...

--- whisperpipe Results ---
Final Transcription: [transcribed text]
WER: 12.34%
SI: 78.45%
Avg Latency: 132.45 ms
Total Processing Time (excl. finalization): 30.00s
Number of intermediate outputs: 15

================================================================================
TEST 2: Whisper Baseline (Simulated Streaming)
================================================================================
Loading Whisper baseline model: base
Using device: cuda
Simulating streaming by transcribing progressively larger chunks...
Transcribing 0-30.0s...
Baseline processing complete. Total time: 45.00s

--- Baseline Results ---
Final Transcription: [transcribed text]
WER: 15.67%
SI: 60.12%
Avg Latency: 150.23 ms
Total Processing Time: 45.00s
Number of intermediate outputs: 30

================================================================================
COMPARISON SUMMARY
================================================================================

Metric                         whisperpipe          Baseline             Improvement         
------------------------------------------------------------------------------------------
WER                               12.34%               15.67%                21.2%
Stability Index (SI)              78.45%               60.12%                30.5%
Avg Latency (ms)                 132.45 ms            150.23 ms             17.78 ms
Total Processing Time (s)         30.00 s              45.00 s               33.3%

================================================================================
BENCHMARK COMPLETE
================================================================================
```

## Understanding the Metrics

### WER (Word Error Rate)
- Lower is better
- Measures transcription accuracy
- Formula: `(Substitutions + Deletions + Insertions) / Total Words × 100%`

### SI (Stability Index)
- Higher is better (0-100%)
- Novel metric measuring output consistency
- Formula: `(1 - avg_edit_distance / avg_length) × 100%`
- 100% = perfectly stable (no revisions)
- 0% = completely unstable (constant changes)

### Latency
- Lower is better
- Average time to process each transcription chunk
- Measured in milliseconds

### Total Processing Time
- Time to process all audio
- **whisperpipe**: Excludes finalization delay (actual processing only)
- **Baseline**: Total time for all progressive transcriptions

## Key Implementation Details

### Processing Time Measurement

**whisperpipe:**
```python
# Start timing when audio feeding begins
pipe_start_time = time.time()

# Feed audio chunks
for chunk in micro_chunks:
    pipe.audio_queue.put(chunk)
    time.sleep(chunk_duration)

# End timing when audio feeding completes (before finalization)
pipe_processing_end_time = time.time()
pipe_processing_time = pipe_processing_end_time - pipe_start_time

# Then wait for finalization (not counted in processing time)
time.sleep(pipe.finalization_delay + 2)
```

**Baseline:**
```python
baseline_start_time = time.time()

# Transcribe progressively: 0-1s, 0-2s, 0-3s, ..., 0-end
for i in range(1, int(audio_duration) + 2):
    end_sample = min(i * increment_samples, len(audio))
    audio_chunk = audio[:end_sample]
    transcription, proc_time = baseline.transcribe_progressive_chunk(audio_chunk)

baseline_processing_time = time.time() - baseline_start_time
```

## Customization

### Change Model Size
Edit `Benchmark.py`:
```python
# Line 34 and 150
pipe = pipeStream(model_name="base", ...)  # Change to "tiny", "small", "medium", "large"
baseline = WhisperBaseline(model_name="base", ...)
```

### Change Finalization Delay
Edit `Benchmark.py`:
```python
# Line 34
pipe = pipeStream(model_name="base", language="en", finalization_delay=5.0, ...)
```

### Change Progressive Increment
Edit `Benchmark.py`:
```python
# Line 166
increment_seconds = 1.0  # Change to 0.5, 2.0, etc.
```

## Troubleshooting

### Issue: "No module named 'soundfile'"
```bash
pip install soundfile
```

### Issue: "No such file: evaluation/sample.wav"
Either:
1. Place your own audio file there, or
2. Run `python generate_sample_audio.py` to create a test file

### Issue: CUDA out of memory
Use a smaller model:
```python
model_name="tiny"  # or "base" instead of "medium"/"large"
```

Or force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
python Benchmark.py
```

### Issue: Transcription is empty with synthetic audio
This is expected. Whisper is trained on speech and won't transcribe synthetic tones. Use real speech audio for meaningful benchmarks.

## Real-World Testing

For publication-quality results, use standard datasets:

1. **LibriSpeech** (clean speech)
   - Download: http://www.openslr.org/12/
   - Use test-clean subset

2. **CommonVoice** (diverse speakers)
   - Download: https://commonvoice.mozilla.org/

3. **TED-LIUM** (conversational)
   - Download: https://www.openslr.org/51/

Update the `reference` variable in `Benchmark.py` with the ground truth transcription for your audio file.

## Citation

If you use this benchmark in your research:

```bibtex
@misc{whisperpipe-benchmark,
  title={Enhanced Whisper Streaming with Dual-Buffer Architecture},
  author={Your Name},
  year={2024}
}
```
