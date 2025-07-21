# Debug Mode Usage Guide

## Overview

The `WhisperStreamingTranscriberWithSpecials` class now includes a debug mode feature to control the verbosity of output messages. This makes the code more user-friendly while preserving detailed debugging information for developers.

## Essential Messages (Always Displayed)

These messages are always shown regardless of debug mode setting:

- `[TRANSCRIPTION]` - The actual transcribed text
- `[NEW WORDS DETECTED] Word count: X → Y` - When new words are detected
- `stable buffer: <content>` - Current stable buffer content  
- `active buffer: <content>` - Current active buffer content
- `[TIMER EXPIRED]` - When the finalization timer expires
- `[SENTENCE COMPLETE]` - When a sentence is completed
- `[LLM INPUT]: <text>` - Text being sent to LLM

## Debug Messages (Conditional)

These messages only appear when debug mode is enabled:

- Pattern detection and similarity calculations
- Word timestamp analysis
- Buffer management details
- Foreign language detection
- Audio processing details
- State reset notifications
- Session summaries

## Usage Examples

### Basic Usage with Default Debug Mode

```python
# Debug mode is enabled by default (debug_mode=True)
transcriber = WhisperStreamingTranscriberWithSpecials(model_name="base.en")

# Start streaming - will show all debug messages
transcriber.start_streaming()
```

### Disable Debug Mode for Clean Output

```python
# Initialize with debug mode disabled
transcriber = WhisperStreamingTranscriberWithSpecials(model_name="base.en", debug_mode=False)

# Or disable it after initialization
transcriber = WhisperStreamingTranscriberWithSpecials(model_name="base.en")
transcriber.debug_mode(False)

# Start streaming - will only show essential messages
transcriber.start_streaming()
```

### Toggle Debug Mode During Runtime

```python
transcriber = WhisperStreamingTranscriberWithSpecials(model_name="base.en")

# Start with debug enabled
transcriber.start_streaming()

# Later, disable debug mode for cleaner output
transcriber.debug_mode(False)

# Re-enable for troubleshooting
transcriber.debug_mode(True)
```

## Methods

### Constructor Parameter

```python
def __init__(self, model_name="base.en", buffer_duration_seconds=5.0, debug_mode=True):
```

- `debug_mode`: Boolean (default True) - Enable debug mode for detailed logging

### Debug Mode Control

```python
def debug_mode(self, enabled=True):
```

- `enabled`: Boolean (default True) - True to enable debug mode, False to disable

## Benefits

**For End Users:**
- Clean, focused output showing only essential transcription information
- Easier to read and understand the transcription process
- Reduced visual clutter

**For Developers:**
- Full debugging information available when needed  
- Can toggle debug mode without code changes
- All original debugging functionality preserved
- Default debug mode enabled for development

## Example Output

### With Debug Mode Enabled:
```
[TRANSCRIPTION] Hello world
[DEBUG] Found common prefix with 85.2% similarity: 'Hello'
[DEBUG] Finding end time for last word: 'world'
[NEW WORDS DETECTED] Word count: 0 → 2
stable buffer: Hello
active buffer: world
[DUPLICATE DETECTED] Common text: 'Hello'
[TIMER EXPIRED] Stable buffer delay: 10.1s - finalizing sentence
[SENTENCE COMPLETE] Hello world
[LLM INPUT]: Hello world
```

### With Debug Mode Disabled:
```
[TRANSCRIPTION] Hello world
[NEW WORDS DETECTED] Word count: 0 → 2
stable buffer: Hello
active buffer: world
[TIMER EXPIRED] Stable buffer delay: 10.1s - finalizing sentence
[SENTENCE COMPLETE] Hello world
[LLM INPUT]: Hello world
```