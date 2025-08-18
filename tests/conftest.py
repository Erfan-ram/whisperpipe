"""
pytest configuration and fixtures for whisperpipe tests
"""
import pytest
import sys
import os

# Add the project root to the Python path so we can import whisperpipe
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing without dependencies"""
    class MockWhisperModel:
        def transcribe(self, audio, **kwargs):
            return {
                "text": "This is a test transcription", 
                "segments": [
                    {"text": "This is a test transcription", "start": 0.0, "end": 2.0}
                ]
            }
    return MockWhisperModel()


@pytest.fixture
def mock_pyaudio():
    """Mock PyAudio for testing without audio dependencies"""
    class MockPyAudio:
        def __init__(self):
            pass
        
        def terminate(self):
            pass
            
        def open(self, **kwargs):
            class MockStream:
                def read(self, frames):
                    return b'0' * (frames * 2)  # Mock audio data
                def stop_stream(self):
                    pass
                def close(self):
                    pass
            return MockStream()
    
    return MockPyAudio()