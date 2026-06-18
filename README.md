# whisperpipe

> Real-time, offline speech-to-text streaming powered by OpenAI Whisper

[![PyPI version](https://img.shields.io/pypi/v/whisperpipe)](https://pypi.org/project/whisperpipe/)
[![Python](https://img.shields.io/pypi/pyversions/whisperpipe)](https://pypi.org/project/whisperpipe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2604.25611-b31b1b.svg)](https://arxiv.org/abs/2604.25611)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19646625.svg)](https://doi.org/10.5281/zenodo.19646625)

**whisperpipe** lets you stream microphone audio directly into OpenAI's Whisper model — locally, privately, and for free. No API keys, no internet, no subscriptions.

> 📖 **Academic Backing:** This repository contains the official implementation of our paper, [**WhisperPipe: A Resource-Efficient Streaming Architecture for Real-Time Automatic Speech Recognition**](https://arxiv.org/abs/2604.25611). If you use **whisperpipe** in your research, please cite our paper as described in the [Citation](#citation) section.

---

## Why whisperpipe?

| | Cloud ASR | whisperpipe |
|---|---|---|
| Privacy | Data sent to servers | 100% local |
| Cost | Pay-per-use | Free |
| Offline | No | Yes |
| Latency | Network dependent | Local only |

---

## Features

- **Real-time transcription** — continuous audio capture and live text output
- **Callback system** — hook any function to receive transcribed text (LLM, logging, UI, etc.)
- **Pause / Resume** — stop listening while your assistant responds, resume on demand
- **Multi-language** — any language supported by Whisper
- **Device selection** — choose which microphone to use
- **Thread-safe** — designed for concurrent use
- **CUDA support** — automatically uses GPU if available

---

## Installation

```bash
pip install whisperpipe
```

Or for the latest version directly from GitHub:

```bash
pip install git+https://github.com/Erfan-ram/whisperpipe.git
```

> **System dependencies:** PyAudio requires PortAudio. On Linux: `sudo apt install portaudio19-dev`, on macOS: `brew install portaudio`.

---

## Quick Start

```python
from whisperpipe import pipeStream
import time

transcriber = pipeStream(model="base", language="en")
transcriber.start_streaming()

print("Listening... Press Ctrl+C to stop")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    transcriber.stop_streaming()
```

---

## Usage Examples

### With a Callback (LLM Integration)

Register any function to be called each time a sentence is finalized:

```python
from whisperpipe import pipeStream
import time

def my_callback(text):
    print(f"Transcribed: {text}")
    # Send to your LLM, log it, update UI, etc.

transcriber = pipeStream(model="base", language="en")
transcriber.set_def_callback(my_callback)
transcriber.start_streaming()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    transcriber.stop_streaming()
```

### Turn-based Conversation (Pause / Resume)

Pause listening while your assistant is speaking, then resume:

```python
from whisperpipe import pipeStream
import time

transcriber = pipeStream(model="base", language="en")

def on_speech(text):
    transcriber.pause_streaming()          # Stop listening
    print(f"User: {text}")

    response = f"You said: {text}"         # Replace with your LLM call
    print(f"Assistant: {response}")

    transcriber.resume_streaming()         # Start listening again

transcriber.set_def_callback(on_speech)
transcriber.start_streaming()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    transcriber.stop_streaming()
```

> For more complete examples including manual control and status checking, see [example_usage.py](example_usage.py).

---

## Model Selection

| Model | Size | Speed | Accuracy | Recommended for |
|-------|------|-------|----------|-----------------|
| `tiny` | 75 MB | Fastest | Low | Testing, prototyping |
| `base` | 145 MB | Fast | Good | General use |
| `small` | 466 MB | Medium | Better | Balanced performance |
| `medium` | 1.5 GB | Slow | High | High accuracy needed |
| `large` | 3 GB | Slowest | Best | Maximum accuracy |

---

## API Reference

### `pipeStream()`

```python
pipeStream(
    model="base",
    language="en",
    finalization_delay=10.0,
    processing_interval=1.0,
    buffer_duration_seconds=5.0,
    debug_mode=False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"base"` | Whisper model to load |
| `language` | str | `"en"` | Language code (`"en"`, `"fa"`, `"es"`, ...) |
| `finalization_delay` | float | `10.0` | Seconds of silence before finalizing a sentence |
| `processing_interval` | float | `1.0` | How often (seconds) to process buffered audio |
| `buffer_duration_seconds` | float | `5.0` | Audio buffer size in seconds |
| `debug_mode` | bool | `False` | Print internal debug logs |

### Methods

| Method | Description |
|--------|-------------|
| `start_streaming()` | Start microphone capture and transcription |
| `stop_streaming()` | Stop transcription |
| `set_def_callback(fn)` | Register a callback — called with `(text: str)` on each finalized sentence. Pass `None` to clear. |
| `pause_streaming()` | Temporarily pause audio processing |
| `resume_streaming()` | Resume after a pause |
| `is_running()` | Returns `True` if actively running |
| `is_paused()` | Returns `True` if currently paused |
| `input_devices()` | List available microphone devices with their IDs |

---

## Requirements

- Python 3.9 – 3.12
- `openai-whisper`
- `pyaudio`
- `pynput`
- `sounddevice`
- NumPy and PyTorch (installed automatically with Whisper)

---

## License

MIT — see [LICENSE](LICENSE)

## Authors

**Erfan Ramezani** · erfanramezany245@gmail.com  
**Mohammad Mahdi Giahi**

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## Citation

We can share academic knowledge this way :

```bibtex
@misc{ramezani2026whisperpipe,
  title         = {WhisperPipe: A Resource-Efficient Streaming Architecture for Real-Time Automatic Speech Recognition},
  author        = {Erfan Ramezani and Mohammad Mahdi Giahi and Mohammad Erfan Zarabadipour and Amir Reza Yosefian and Hamid Ghadiri},
  year          = {2026},
  month         = apr,
  publisher     = {arXiv},
  eprint        = {2604.25611},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2604.25611},
  url           = {https://doi.org/10.48550/arXiv.2604.25611}
}
```

And the software repository:

```bibtex
@software{whisperpipe_code_2026,
  author       = {Erfan Ramezani and Mohammad Mahdi Giahi},
  title        = {WhisperPipe: Source Code and Implementation},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19646625},
  url          = {https://doi.org/10.5281/zenodo.19646625}
}
```
