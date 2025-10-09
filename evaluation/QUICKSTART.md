# WhisperPipe Evaluation - Quick Start Guide

This guide shows you how to evaluate WhisperPipe and reproduce the paper results.

## Table of Contents

1. [Installation](#installation)
2. [Quick Test (5 minutes)](#quick-test-5-minutes)
3. [Full Evaluation (Live Microphone)](#full-evaluation-live-microphone)
4. [Automated Evaluation (Pre-recorded Audio)](#automated-evaluation-pre-recorded-audio)
5. [Understanding the Results](#understanding-the-results)
6. [Reproducing Paper Statistics](#reproducing-paper-statistics)
7. [Troubleshooting](#troubleshooting)

## Installation

### 1. Install WhisperPipe

```bash
# From the repository root
cd /path/to/Audio2Text
pip install -e .
```

Or:

```bash
pip install whisperpipe
```

### 2. Verify Installation

```bash
python -c "from whisperpipe import pipeStream; print('✓ WhisperPipe installed')"
python -c "import whisper; print('✓ Whisper installed')"
```

### 3. Install Optional Dependencies

For audio file evaluation:
```bash
pip install scipy  # For audio file generation
```

## Quick Test (5 minutes)

Test the basic functionality without full evaluation:

```bash
cd evaluation

# Test naive baseline (30 seconds)
python -c "
from naive_whisper import NaiveWhisperStream
import time

t = NaiveWhisperStream(model_name='tiny', debug_mode=False)
t.start_streaming()
print('Speak for 10 seconds...')
time.sleep(10)
t.stop_streaming()
"
```

```bash
# Test WhisperPipe (30 seconds)
python -c "
from whisperpipe import pipeStream
import time

t = pipeStream(model_name='tiny', debug_mode=False)
t.start_streaming()
print('Speak for 10 seconds...')
time.sleep(10)
t.stop_streaming()
"
```

## Full Evaluation (Live Microphone)

This is the main evaluation method used in the paper.

### Step 1: Run Comparison Script

```bash
cd evaluation
python compare.py --duration 60 --model base
```

**What happens:**
1. Loads the Whisper model (may take a minute first time)
2. Runs **Naive Baseline** for 60 seconds (you speak)
3. 5 second break
4. Runs **WhisperPipe** for 60 seconds (you speak again)
5. Prints comparison metrics

### Step 2: What to Say

For best results, speak similar content in both tests:

**Option 1: Read a passage**
```
The quick brown fox jumps over the lazy dog. 
This sentence contains every letter of the alphabet.
We use it for testing speech recognition systems.
Whisper is a state-of-the-art automatic speech recognition model
developed by OpenAI for transcribing spoken language into text.
```

**Option 2: Describe something**
- Describe your day
- Explain a technical concept
- Tell a story

**Option 3: Read code or technical content**
- Python function definitions
- Algorithm explanations
- Technical documentation

### Step 3: Review Results

The script will output:

```
COMPARISON RESULTS
================================================================================

📊 EDIT OVERHEAD COMPARISON
  Naive Baseline:    2.1×
  WhisperPipe:       0.45×
  Improvement:       78.6% reduction (4.7× better)

✅ STABILITY COMPARISON
  Naive Baseline:    32.0%
  WhisperPipe:       85.0%
  Improvement:       +53.0 percentage points

⏱️  COMMIT LATENCY
  WhisperPipe:       280ms mean commit latency
```

### Different Models

```bash
# Tiny (fastest, less accurate)
python compare.py --duration 30 --model tiny

# Small (balanced)
python compare.py --duration 60 --model small

# Large (slowest, most accurate)
python compare.py --duration 120 --model large
```

### Different Languages

```bash
# Spanish
python compare.py --duration 60 --language es

# French
python compare.py --duration 60 --language fr

# German
python compare.py --duration 60 --language de
```

## Automated Evaluation (Pre-recorded Audio)

For reproducible results without live speaking:

### Step 1: Generate Test Audio

```bash
cd evaluation
python evaluate_audio.py --generate-sample
```

This creates `test_audio.wav` (30 seconds).

### Step 2: Run Evaluation

```bash
python evaluate_audio.py --audio test_audio.wav --model base
```

### Step 3: Use Your Own Audio

```bash
# Record your own audio file (16kHz WAV recommended)
# Or use any audio file (MP3, WAV, FLAC, etc.)
python evaluate_audio.py --audio myrecording.wav --model base
```

## Understanding the Results

### Edit Overhead (Lower is Better)

**What it measures:** How many times the transcription changes relative to the final length.

**Paper claim:** 0.45× (WhisperPipe) vs 2.1× (Naive)

**Interpretation:**
- **0.45×** = Text is very stable, only minor refinements
- **2.1×** = Text changes more than 2× its final length (very unstable)

**Example:**
- Final text: "hello world" (2 words)
- Naive might go: "hell" → "hello there" → "hello the world" → "hello world" 
  - Many changes = high overhead
- WhisperPipe might go: "hello" → "hello world"
  - Few changes = low overhead

### Stability/Consistency (Higher is Better)

**What it measures:** Percentage of time the transcription stays the same between updates.

**Paper claim:** 85% (WhisperPipe) vs 32% (Naive)

**Interpretation:**
- **85%** = Text stays stable 85% of the time
- **32%** = Text changes 68% of the time (very unstable)

### Commit Latency (Lower is Better)

**What it measures:** Time from speech to stable buffer commitment.

**Paper claim:** 280ms mean latency

**Interpretation:**
- **280ms** = Less than 1/3 second from speaking to stable text
- Important for real-time applications (live captions, voice assistants)

### Processing Time

**What it measures:** How long each transcription cycle takes.

**Key difference:**
- **Naive:** Grows linearly with buffer size (more audio = longer processing)
- **WhisperPipe:** Near-constant (always processes ~same amount)

## Reproducing Paper Statistics

To reproduce the exact statistics from the paper:

### Setup

1. **Model:** Base (or specify in paper)
2. **Duration:** 60-120 seconds per test
3. **Content:** Clear speech, minimal background noise
4. **Multiple runs:** Average 3-5 runs for robustness

### Run Evaluation

```bash
cd evaluation

# Run 1
python compare.py --duration 120 --model base > run1.txt

# Run 2  
python compare.py --duration 120 --model base > run2.txt

# Run 3
python compare.py --duration 120 --model base > run3.txt
```

### Extract Statistics

Each run outputs a "PAPER STATISTICS" section:

```
💡 PAPER STATISTICS
================================================================================

Results demonstrate that WhisperPipe achieves 0.45× edit overhead 
(79% reduction compared to naive re-transcription at 2.1×), 
with 280ms mean commit latency from speech onset to stable buffer.

Our stability analysis shows 85% transcription consistency, representing 
a 53 percentage point improvement over naive streaming approaches (32% stability).
```

Average the numbers across runs for the final paper statistics.

## Troubleshooting

### Issue: Model Download Fails

```bash
# Pre-download the model
python -c "import whisper; whisper.load_model('base')"
```

### Issue: No Microphone Input

```bash
# List audio devices
python -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f'{i}: {info[\"name\"]} (in:{info[\"maxInputChannels\"]})')
"
```

### Issue: Out of Memory

Use a smaller model:
```bash
python compare.py --duration 30 --model tiny
```

### Issue: Poor Transcription Quality

1. **Check microphone:** Test with system audio recorder
2. **Reduce background noise:** Find quieter environment
3. **Speak clearly:** Enunciate, moderate pace
4. **Try better model:** Use `--model small` or `--model medium`

### Issue: Import Errors

```bash
# Reinstall dependencies
pip install --upgrade openai-whisper pyaudio numpy pynput

# Check installation
python -c "import whisper, pyaudio, numpy, pynput; print('All OK')"
```

### Issue: Permission Denied (Microphone)

**macOS:**
```
System Preferences → Security & Privacy → Microphone → Allow Terminal/Python
```

**Linux:**
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Logout and login again
```

**Windows:**
```
Settings → Privacy → Microphone → Allow apps to access microphone
```

## Advanced Usage

### Custom Metrics Collection

```python
from evaluation.metrics import MetricsTracker
from whisperpipe import pipeStream

# Create transcriber
pipe = pipeStream(model_name="base")
tracker = MetricsTracker()

# Wrap commit method
original_commit = pipe._commit_to_stable_buffer
def tracked_commit(text, time):
    tracker.record_stable_buffer_update(text)
    tracker.record_commit_event(text, time)
    return original_commit(text, time)

pipe._commit_to_stable_buffer = tracked_commit

# Run
tracker.start_session()
pipe.start_streaming()
# ... speak ...
pipe.stop_streaming()
tracker.end_session()

# Get metrics
metrics = tracker.get_comprehensive_metrics()
print(f"Edit overhead: {metrics['edit_overhead']:.2f}×")
print(f"Stability: {metrics['stability_percentage']:.1f}%")
```

### Batch Testing

```bash
# Create test script
cat > batch_test.sh << 'EOF'
#!/bin/bash
for model in tiny base small; do
    echo "Testing $model..."
    python compare.py --duration 60 --model $model > results_${model}.txt
done
EOF

chmod +x batch_test.sh
./batch_test.sh
```

## Next Steps

1. **Run basic comparison** (30 seconds with tiny model)
2. **Run full comparison** (60 seconds with base model)
3. **Review metrics** against paper claims
4. **Try different scenarios** (different speech, languages, models)
5. **Generate paper statistics** (multiple runs, averaged)

## Getting Help

- **GitHub Issues:** https://github.com/Erfan-ram/whisperpipe/issues
- **Documentation:** See `evaluation/README.md` for detailed metrics explanation
- **Example Output:** Check `evaluation/README.md` for expected output format

## Summary of Commands

```bash
# Quick test
cd evaluation
python compare.py --duration 30 --model tiny

# Full evaluation
python compare.py --duration 60 --model base

# With audio file
python evaluate_audio.py --generate-sample
python evaluate_audio.py --audio test_audio.wav

# Different configurations
python compare.py --duration 60 --model small --language es
```
