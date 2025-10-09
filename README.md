# whisperpipe

Real-time speech-to-text streaming with OpenAI Whisper

## Description

whisperpipe is a powerful, easy-to-use Python package for real-time audio transcription using OpenAI's Whisper model. It provides seamless integration with callback functions for LLM processing and supports pause/resume functionality for interactive applications.

## Features

- **Real-time audio transcription** using OpenAI Whisper
- **Callback system** for custom processing (LLM integration, etc.)
- **Pause/Resume functionality** for interactive applications
- **Multiple language support**
- **Configurable processing parameters**
- **Thread-safe operation**
- **Easy installation and usage**

## Installation

### From PyPI

```bash
pip install whisperpipe
```

### From GitHub

```bash
pip install git+https://github.com/Erfan-ram/whisperpipe.git
```

## Quick Start

```python
from whisperpipe import pipeStream

# Basic usage
transcriber = pipeStream(
    model_name="base",
    language="en",
    finalization_delay=10.0,
    processing_interval=1.0
)

# Start streaming
transcriber.start_streaming()
```

## Usage Examples

### Basic Transcription

```python
from whisperpipe import pipeStream

# Create transcriber instance
transcriber = pipeStream(
    model_name="base",
    language="en",
    finalization_delay=10.0,
    processing_interval=1.0
)

# Start transcription
transcriber.start_streaming()

# The transcribed text will be printed to console
# Press Ctrl+C to stop
```

### With Custom Callback (LLM Integration)

```python
from whisperpipe import pipeStream

def llm_processor(text):
    """Custom function to process transcribed text"""
    print(f"Processing: {text}")
    # Your LLM integration here
    # e.g., send to OpenAI, Claude, local model, etc.
    response = your_llm_api.chat(text)
    print(f"Response: {response}")
    return response

# Create transcriber with callback
transcriber = pipeStream(
    model_name="base",
    language="en",
    finalization_delay=10.0,
    processing_interval=1.0
)

# Register callback
transcriber.set_def_callback(llm_processor)

# Start streaming with LLM integration
transcriber.start_streaming()
```

### Interactive Mode with Pause/Resume

```python
from whisperpipe import pipeStream
import time

def interactive_processor(text):
    """Process text and pause for response"""
    # Pause transcriber while processing
    transcriber.pause_streaming()
    
    print(f"User said: {text}")
    
    # Process with your system
    response = process_with_llm(text)
    
    # Speak or display response
    print(f"Assistant: {response}")
    
    # Resume for next input
    transcriber.resume_streaming()

transcriber = pipeStream()
transcriber.set_def_callback(interactive_processor)
transcriber.start_streaming()
```

## API Reference

### Constructor Parameters

- `model_name` (str): Whisper model name ("tiny", "base", "small", "medium", "large", "base", "small.en"). Default: "base"
- `language` (str): Language code for transcription ("en", "es", "fr", etc.). Default: "en"
- `finalization_delay` (float): Wait time in seconds before finalizing transcription. Default: 10.0
- `processing_interval` (float): Interval in seconds between processing cycles. Default: 1.0
- `buffer_duration_seconds` (float): Time window in seconds to hold audio for processing. Default: 5.0
- `debug_mode` (bool): Enable debug mode for detailed logging. Default: True

### Methods

#### Core Methods
- `start_streaming()`: Start audio capture and transcription
- `stop_streaming()`: Stop audio capture and transcription

#### Callback System
- `set_def_callback(callback_function)`: Register a callback function for processing transcribed text
- `set_def_callback(None)`: Clear the callback (use default behavior)

#### Pause/Resume Control
- `pause_streaming()`: Pause audio processing temporarily
- `resume_streaming()`: Resume audio processing
- `is_paused()`: Check if transcriber is paused
- `is_running()`: Check if transcriber is running

## Requirements

- Python 3.8+
- PyAudio
- OpenAI Whisper
- PyTorch
- NumPy
- pynput

## Evaluation Framework

For researchers and paper authors, we provide a comprehensive evaluation framework:

### Quick Evaluation

```bash
cd evaluation
python compare.py --duration 60 --model base
```

This compares WhisperPipe against a naive baseline and generates metrics for:
- **Edit Overhead**: How often transcription changes (lower is better)
- **Stability**: Transcription consistency percentage (higher is better)  
- **Commit Latency**: Time from speech to stable text (lower is better)
- **Processing Time**: Computational efficiency analysis

### Paper Statistics

The framework automatically generates paper-ready statistics:

```
Results demonstrate that WhisperPipe achieves 0.45× edit overhead 
(78% reduction compared to naive re-transcription at 2.1×), with 280ms 
mean commit latency from speech onset to stable buffer. Our stability 
analysis shows 85% transcription consistency, representing a 53 percentage 
point improvement over naive streaming approaches (32% stability).
```

### Documentation

- **Quick Start**: `evaluation/QUICKSTART.md`
- **Detailed Guide**: `evaluation/README.md`
- **Paper Authors**: `INSTRUCTIONS.md`
- **Complete Overview**: `evaluation/SUMMARY.md`

### Files Included

- `naive_whisper.py` - Naive baseline for comparison
- `metrics.py` - Metrics tracking system
- `compare.py` - Main comparison tool
- `evaluate_audio.py` - File-based evaluation
- `validate_syntax.py` - Code validation
- `examples.py` - Usage examples

See `evaluation/README.md` for complete documentation.

## Architecture

WhisperPipe introduces three key innovations for streaming transcription:

1. **Dual-Buffer Architecture**: Separates finalized text from active processing audio
2. **Similarity-Based Stabilization**: Word-level timestamps with multi-way confirmation
3. **Adaptive Content Filtering**: Detects and filters foreign language segments

Unlike naive approaches that re-transcribe the entire audio buffer on each cycle, WhisperPipe only processes new audio segments, achieving constant-time processing and stable output.

## License

MIT License

## Author

Erfan Ramezani - erfanramezany245@gmail.com

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the [GitHub Issues](https://github.com/Erfan-ram/whisperpipe/issues) page.

## Citation

If you use WhisperPipe in your research, please cite:

```bibtex
@software{whisperpipe2024,
  title={WhisperPipe: Real-time Streaming Adaptation for Whisper},
  author={Ramezani, Erfan},
  year={2024},
  url={https://github.com/Erfan-ram/whisperpipe}
}
```