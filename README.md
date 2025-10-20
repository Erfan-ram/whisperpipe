# whisperpipe

Real-time speech-to-text streaming with OpenAI Whisper

## Description

whisperpipe is a powerful, easy-to-use Python package for real-time, offline audio transcription using OpenAI's Whisper model. It runs locally, making it a free and private solution for continuous speech-to-text applications. It provides seamless integration with callback functions for LLM processing and supports pause/resume functionality for interactive applications.

## Why whisperpipe?

In a world where most ASR (Automatic Speech Recognition) services are cloud-based, whisperpipe offers a refreshing alternative by harnessing the power of OpenAI's Whisper model to run directly on your local machine. This approach provides several key advantages:

- **Complete Privacy**: Since all transcription is done locally, your voice data never leaves your computer. This is crucial for applications that handle sensitive or private conversations.
- **Zero Cost**: Say goodbye to recurring subscription fees and per-minute charges. whisperpipe is free to use, making it an economical choice for both hobbyists and commercial projects.
- **No Internet Required**: Whether you're on a plane, in a remote location, or simply have an unstable internet connection, whisperpipe works flawlessly offline.
- **Real-time Performance**: Designed for continuous, real-time transcription, whisperpipe is ideal for live applications such as voice-controlled assistants, dictation software, and more.
- **Unleash the Power of Whisper**: By running the Whisper model locally, you have full control over the transcription process, from model selection to performance tuning.

whisperpipe empowers you to build powerful, private, and cost-effective voice applications with ease.

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

- `model_name` (str): Whisper model name ("tiny", "base", "small", "medium", "large"). Default: "base"
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

## License

MIT License

## Author

Erfan Ramezani - erfanramezany245@gmail.com

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the [GitHub Issues](https://github.com/Erfan-ram/whisperpipe/issues) page.