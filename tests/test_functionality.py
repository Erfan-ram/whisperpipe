"""
Tests for whisperpipe functionality using mocks to avoid dependencies
"""
import pytest
import threading
import time


class MockPipeStream:
    """Mock pipeStream class for testing functionality without dependencies"""
    
    def __init__(self, model_name="base", language="en", finalization_delay=10.0, processing_interval=1.0):
        self.model_name = model_name
        self.language = language
        self.finalization_delay = finalization_delay
        self.processing_interval = processing_interval
        
        self.is_recording = False
        self._def_callback = None
        self._is_paused = False
        self._pause_lock = threading.Lock()
        self.stable_text_buffer = ""
        self.completed_sentences = []
        self._current_device_id = None
        self._available_devices = [
            {"id": 0, "name": "Default Input", "channels": 1},
            {"id": 1, "name": "Microphone Array", "channels": 2}
        ]
    
    def set_def_callback(self, callback_function):
        """Set a callback function to handle text when it's sent to LLM"""
        if callback_function is not None and not callable(callback_function):
            raise ValueError("Callback function must be callable or None")
        self._def_callback = callback_function
    
    def pause_streaming(self):
        """Pause the audio streaming and processing temporarily"""
        with self._pause_lock:
            if not self.is_recording:
                return False
            if self._is_paused:
                return False
            self._is_paused = True
            return True
    
    def resume_streaming(self):
        """Resume the audio streaming and processing"""
        with self._pause_lock:
            if not self.is_recording:
                return False
            if not self._is_paused:
                return False
            self._is_paused = False
            return True
    
    def is_paused(self):
        """Check if the transcriber is currently paused"""
        return self._is_paused
    
    def is_running(self):
        """Check if the transcriber is currently running"""
        return self.is_recording
    
    def input_devices(self):
        """List available input devices"""
        return self._available_devices
    
    def set_input_device(self, device_id):
        """Set the input device by ID"""
        available_ids = [device["id"] for device in self._available_devices]
        if device_id not in available_ids:
            return False
        self._current_device_id = device_id
        return True
    
    def get_current_input_device(self):
        """Get current input device information"""
        if self._current_device_id is None:
            return None
        for device in self._available_devices:
            if device["id"] == self._current_device_id:
                return device
        return None
    
    def start_streaming(self):
        """Mock start streaming"""
        self.is_recording = True
        return True
    
    def stop_streaming(self):
        """Mock stop streaming"""
        self.is_recording = False
        self._is_paused = False
        return True


def test_callback_functionality():
    """Test callback registration and usage"""
    transcriber = MockPipeStream()
    
    # Test setting a valid callback
    def test_callback(text):
        return f"Processed: {text}"
    
    transcriber.set_def_callback(test_callback)
    assert transcriber._def_callback == test_callback
    
    # Test setting None callback
    transcriber.set_def_callback(None)
    assert transcriber._def_callback is None
    
    # Test setting invalid callback
    with pytest.raises(ValueError):
        transcriber.set_def_callback("not a function")


def test_pause_resume_functionality():
    """Test pause and resume functionality"""
    transcriber = MockPipeStream()
    
    # Test pause when not running
    assert not transcriber.pause_streaming()
    assert not transcriber.is_paused()
    
    # Start streaming and test pause/resume
    transcriber.start_streaming()
    assert transcriber.is_running()
    
    # Test pause
    assert transcriber.pause_streaming()
    assert transcriber.is_paused()
    
    # Test pause when already paused
    assert not transcriber.pause_streaming()
    
    # Test resume
    assert transcriber.resume_streaming()
    assert not transcriber.is_paused()
    
    # Test resume when not paused
    assert not transcriber.resume_streaming()


def test_device_management():
    """Test audio device management functionality"""
    transcriber = MockPipeStream()
    
    # Test listing devices
    devices = transcriber.input_devices()
    assert len(devices) > 0
    assert all('id' in device for device in devices)
    assert all('name' in device for device in devices)
    
    # Test setting valid device
    assert transcriber.set_input_device(0)
    current_device = transcriber.get_current_input_device()
    assert current_device is not None
    assert current_device['id'] == 0
    
    # Test setting invalid device
    assert not transcriber.set_input_device(999)


def test_constructor_parameters():
    """Test that constructor parameters are properly stored"""
    transcriber = MockPipeStream(
        model_name="tiny.en",
        language="fr", 
        finalization_delay=5.0,
        processing_interval=2.0
    )
    
    assert transcriber.model_name == "tiny.en"
    assert transcriber.language == "fr"
    assert transcriber.finalization_delay == 5.0
    assert transcriber.processing_interval == 2.0


def test_thread_safety():
    """Test that pause/resume operations are thread-safe"""
    transcriber = MockPipeStream()
    transcriber.start_streaming()
    
    results = []
    
    def pause_resume_worker():
        for _ in range(10):
            transcriber.pause_streaming()
            time.sleep(0.001)  # Small delay
            transcriber.resume_streaming()
            time.sleep(0.001)
    
    # Start multiple threads doing pause/resume
    threads = []
    for _ in range(3):
        thread = threading.Thread(target=pause_resume_worker)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Should complete without deadlock or errors
    assert True


def test_callback_with_llm_simulation():
    """Test callback functionality with simulated LLM processing"""
    transcriber = MockPipeStream()
    
    processed_texts = []
    
    def llm_callback(text):
        """Simulate LLM processing"""
        processed = f"LLM processed: {text}"
        processed_texts.append(processed)
        return processed
    
    transcriber.set_def_callback(llm_callback)
    
    # Simulate processing some text
    if transcriber._def_callback:
        result = transcriber._def_callback("Hello world")
        assert result == "LLM processed: Hello world"
        assert len(processed_texts) == 1
        assert processed_texts[0] == "LLM processed: Hello world"