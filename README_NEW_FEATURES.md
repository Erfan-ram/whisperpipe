# Audio2Text - Enhanced with Callback and Pause/Resume Functionality

This enhanced version of Audio2Text provides improved adaptability for various use cases, including real-time streaming and LLM integration scenarios.

## New Features

### 1. LLM Callback Integration

The `_send_to_llm()` method now supports custom callback functions, allowing users to connect their own LLM processing logic or any other processing logic.

#### Usage:

```python
from main_stream import WhisperStreamingTranscriberWithSpecials

# Create transcriber
transcriber = WhisperStreamingTranscriberWithSpecials()

# Define your LLM handler
def my_llm_handler(text):
    print(f"Processing: {text}")
    # Your LLM integration here (OpenAI, Claude, local model, etc.)
    response = your_llm_api.chat(text)
    print(f"Response: {response}")
    return response

# Register the callback
transcriber.set_def_callback(my_llm_handler)

# Start streaming
transcriber.start_streaming()
```

#### Methods:

- `set_def_callback(callback_function)`: Register a callback function
- `set_def_callback(None)`: Clear the callback (use default behavior)

### 2. Pause/Resume Functionality

Control the transcriber state dynamically to support turn-based conversations and LLM input scenarios.

#### Usage:

```python
# Start transcriber
transcriber.start_streaming()

# Pause processing (audio stream continues, but processing stops)
transcriber.pause_streaming()

# Resume processing
transcriber.resume_streaming()

# Check status
if transcriber.is_running() and not transcriber.is_paused():
    print("Transcriber is actively processing")
```

#### Methods:

- `pause_streaming()`: Pause audio processing temporarily
- `resume_streaming()`: Resume audio processing
- `is_paused()`: Check if transcriber is paused
- `is_running()`: Check if transcriber is running

## Usage Scenarios

### Scenario 1: Real-time Streaming with LLM

```python
def llm_processor(text):
    # Send to your LLM
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text}]
    )
    print(f"LLM: {response.choices[0].message.content}")

transcriber = WhisperStreamingTranscriberWithSpecials()
transcriber.set_def_callback(llm_processor)
transcriber.start_streaming()

# Continuous processing - speech automatically sent to LLM
```

### Scenario 2: LLM Input Mode with Pause/Resume

```python
def interactive_llm_processor(text):
    # Pause transcriber while LLM processes
    transcriber.pause_streaming()
    
    # Process with LLM
    response = process_with_llm(text)
    
    # Speak or display response
    speak_response(response)
    
    # Resume transcriber for next input
    transcriber.resume_streaming()

transcriber = WhisperStreamingTranscriberWithSpecials()
transcriber.set_def_callback(interactive_llm_processor)
transcriber.start_streaming()

# Turn-based conversation flow
```

### Scenario 3: Manual Control

```python
transcriber = WhisperStreamingTranscriberWithSpecials()
transcriber.start_streaming()

# Manual control based on application logic
if user_wants_to_pause:
    transcriber.pause_streaming()

if ready_for_input:
    transcriber.resume_streaming()
```

## Error Handling

The callback system includes robust error handling:

```python
def potentially_failing_callback(text):
    if some_error_condition:
        raise Exception("LLM API error")
    return process_text(text)

transcriber.set_def_callback(potentially_failing_callback)
# If callback fails, transcriber falls back to default behavior
```

## Thread Safety

All pause/resume operations are thread-safe and can be called from any thread:

```python
import threading

def control_thread():
    time.sleep(5)
    transcriber.pause_streaming()  # Safe to call from any thread
    time.sleep(2)
    transcriber.resume_streaming()

threading.Thread(target=control_thread).start()
transcriber.start_streaming()
```

## Requirements

```bash
pip install openai-whisper torch pyaudio numpy pynput
```

## Testing

Run the provided test scripts to verify functionality:

```bash
# Test new methods exist and work
python test_minimal.py

# Test with mock implementation  
python test_integration.py

# See usage examples
python example_usage.py
```

## Migration from Original Version

The changes are fully backward compatible. Existing code will work unchanged:

```python
# This still works exactly as before
transcriber = WhisperStreamingTranscriberWithSpecials()
transcriber.start_streaming()
# Text will be printed to console (default behavior)
```

To use new features, simply add:

```python
# Add callback for custom processing
transcriber.set_def_callback(your_function)

# Add pause/resume control as needed
transcriber.pause_streaming()
transcriber.resume_streaming()
```

## Implementation Details

- **Callback System**: Uses a simple function pointer with error handling and fallback to default behavior
- **Pause/Resume**: Implements thread-safe state management using locks
- **Audio Buffer**: Audio stream continues during pause, only processing is paused
- **State Management**: Proper initialization and cleanup of new state variables
- **Compatibility**: All existing functionality preserved without changes