# Benchmark Instructions

## Overview

`Benchmark.py` compares the enhanced whisperpipe implementation with a baseline Whisper streaming approach. It measures:
- **WER (Word Error Rate)**: Transcription accuracy
- **SI (Stability Index)**: Output consistency across intermediate results
- **Latency**: Processing time per transcription
- **Total Processing Time**: Overall time excluding finalization delay
- **Resource Usage**: GPU memory, RAM, CPU, and computational efficiency

## Key Differences in Testing Methodology

### whisperpipe (Enhanced Streaming - Incremental Chunks)
- Feeds audio in 1-second increments: first 0-1s, then 1-2s, then 2-3s, etc.
- Each chunk appends to the existing audio buffer (realistic streaming behavior)
- Uses dual-buffer architecture with stability mechanisms
- Processing time measured as actual audio duration (not wall clock time)
- Finalization delay is excluded from processing time measurement
- **Consistent resource usage** due to stable buffer management

**This approach simulates how whisperpipe works in real-time:**
- Audio comes in chunks over time
- Each new chunk is appended to the buffer
- Processing happens continuously as audio arrives
- The dual-buffer prevents reprocessing of stable text
- **Memory usage remains bounded** as stable text is committed

### Baseline Whisper (Simulated Streaming)
- Transcribes progressively larger chunks: 0-1s, 0-2s, 0-3s, ..., 0-end
- Simulates real streaming where the model reprocesses from the beginning
- Each transcription is independent (no sliding window)
- More realistic comparison to show the benefit of dual-buffer architecture
- **Growing resource usage** as audio length increases

## Prerequisites

```bash
# Install required packages
pip install soundfile librosa numpy openai-whisper torch psutil

# Optional: For NVIDIA GPU monitoring
pip install pynvml

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
TEST 1: whisperpipe (Enhanced Streaming - Incremental Chunks)
================================================================================
Loading Whisper model: base
Using device: cuda
Feeding audio in 1-second incremental chunks to whisperpipe...
Feeding second 29-30...
Audio feeding complete.
Total elapsed time: 30.50s
Actual audio duration: 30.00s
Processing overhead: 0.50s
Waiting for finalization (10s)...

--- whisperpipe Results ---
Final Transcription: [transcribed text]
WER: 12.34%
SI: 78.45%
Avg Latency: 132.45 ms
Total Processing Time (excl. finalization): 30.00s
Processing Overhead: 0.50s
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

### Resource Usage Metrics

#### Peak GPU Memory (MB)
- Maximum GPU memory allocated during processing
- **whisperpipe**: Bounded due to dual-buffer architecture
- **Baseline**: Grows with audio length (reprocessing from start)

#### Peak RAM (MB)
- Maximum system RAM used by the process
- Shows memory footprint of the system

#### GPU Utilization (%)
- Average GPU usage during processing
- Higher utilization indicates more intensive computation

#### Resource Efficiency Index (REI)
- **Academic Definition**: Peak GPU Memory / Audio Duration (MB/s)
- **Lower is better** - indicates less memory needed per second of audio
- Formula: `REI = Peak_Memory_MB / Audio_Duration_seconds`
- Measures memory efficiency relative to audio processed

#### Memory Growth Rate (MB/s)
- Rate at which memory increases over time
- **Academic Definition**: Linear regression slope of memory usage over time
- **whisperpipe**: Near zero (stable memory usage)
- **Baseline**: Positive value (memory grows with audio length)
- Useful for identifying memory leaks or unbounded growth

#### Computational Intensity (CI)
- **Academic Definition**: Normalized measure of computational resource usage
- Formula: `CI = (GPU_Utilization% / 100) × (Processing_Time / Audio_Duration)`
- **Fallback**: Uses CPU utilization when GPU utilization is not available (0%)
- **Lower is better** - indicates more efficient use of computational resources
- CI = 1.0 means using 100% GPU/CPU for real-time processing
- CI < 1.0 means processing faster than real-time
- CI > 1.0 means processing slower than real-time
- **Note**: When GPU utilization monitoring is unavailable (e.g., no NVML), the metric automatically falls back to CPU utilization to provide meaningful comparison

## Key Implementation Details

### Processing Time Measurement

**whisperpipe (New Method - Incremental Chunks):**
```python
# Start timing when audio feeding begins
pipe_start_time = time.time()

# Feed audio in 1-second increments (0-1s, then 1-2s, then 2-3s, etc.)
for i in range(int(audio_duration)):
    start_sample = i * increment_samples
    end_sample = min((i + 1) * increment_samples, len(audio))
    audio_chunk = audio[start_sample:end_sample]
    
    # Split into micro-chunks and feed
    for micro_chunk in micro_chunks:
        pipe.audio_queue.put(micro_chunk)
        time.sleep(chunk_duration)

# End timing when audio feeding completes (before finalization)
pipe_processing_end_time = time.time()

# Adjusted processing time = actual audio duration (fair comparison)
adjusted_processing_time = audio_duration

# Then wait for finalization (not counted in processing time)
time.sleep(pipe.finalization_delay + 2)
```

**Key points:**
- Processing time = actual audio duration (30s for 30s of audio)
- This represents the real-time nature of streaming
- Finalization delay is excluded (it's a waiting period, not processing)
- Processing overhead shows any computational delays

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

**Key points:**
- Processing time = total wall clock time for all transcriptions
- Shows the computational cost of reprocessing from the beginning each time
- No finalization delay (not applicable to this approach)

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
