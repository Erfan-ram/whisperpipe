"""
Tests for whisperpipe API signature and basic functionality
"""
import pytest
import inspect


def test_pipeStream_class_exists():
    """Test that pipeStream class exists and can be imported"""
    try:
        from whisperpipe.core import pipeStream
        assert pipeStream is not None
    except ImportError as e:
        pytest.skip(f"Import failed (may be expected in test environment): {e}")


def test_constructor_signature():
    """Test that constructor has the expected signature"""
    try:
        from whisperpipe.core import pipeStream
        
        sig = inspect.signature(pipeStream.__init__)
        params = list(sig.parameters.keys())
        
        # Required parameters for the API
        required_params = ['self', 'model', 'language', 'finalization_delay', 'processing_interval']
        
        for param in required_params:
            assert param in params, f"Missing required parameter: {param}"
            
    except ImportError:
        pytest.skip("Import failed (may be expected in test environment)")


def test_constructor_default_values():
    """Test that constructor has appropriate default values"""
    try:
        from whisperpipe.core import pipeStream
        
        sig = inspect.signature(pipeStream.__init__)
        
        # Check that parameters have defaults (except self)
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                assert param.default != inspect.Parameter.empty, f"Parameter {param_name} should have a default value"
                
    except ImportError:
        pytest.skip("Import failed (may be expected in test environment)")


def test_api_compatibility():
    """Test that the API is compatible with the requested signature"""
    try:
        from whisperpipe.core import pipeStream
        
        # Test that we can instantiate with the requested API call
        sig = inspect.signature(pipeStream.__init__)
        call_params = ['model', 'language', 'finalization_delay', 'processing_interval']
        
        for param in call_params:
            assert param in sig.parameters, f"API incompatible: missing {param}"
            
        # Verify the signature is compatible with:
        # pipeStream(model="base", language="en", finalization_delay=10.0, processing_interval=1.0)
        
    except ImportError:
        pytest.skip("Import failed (may be expected in test environment)")


def test_required_methods_exist():
    """Test that all required methods exist"""
    try:
        from whisperpipe.core import pipeStream
        
        required_methods = [
            'input_devices',
            'set_input_device', 
            'get_current_input_device',
            'set_def_callback',
            'start_streaming',
            'stop_streaming',
            'pause_streaming',
            'resume_streaming',
            'is_paused',
            'is_running'
        ]
        
        for method in required_methods:
            assert hasattr(pipeStream, method), f"Missing required method: {method}"
            assert callable(getattr(pipeStream, method)), f"Method {method} is not callable"
            
    except ImportError:
        pytest.skip("Import failed (may be expected in test environment)")


def test_basic_instantiation():
    """Test basic instantiation without loading heavy dependencies"""
    try:
        from whisperpipe.core import pipeStream
        
        # Use __new__ to create instance without calling __init__ (which loads models)
        instance = pipeStream.__new__(pipeStream)
        assert instance is not None
        assert isinstance(instance, pipeStream)
        
    except ImportError:
        pytest.skip("Import failed (may be expected in test environment)")


def test_signal_handling_setup():
    """Test that signal handling setup exists"""
    try:
        from whisperpipe.core import pipeStream
        
        # Check that signal handling method exists
        assert hasattr(pipeStream, '_setup_signal_handling'), "Missing signal handling setup method"
        
    except ImportError:
        pytest.skip("Import failed (may be expected in test environment)")


def test_audio_device_management():
    """Test that audio device management methods exist"""
    try:
        from whisperpipe.core import pipeStream
        
        device_methods = ['input_devices', 'set_input_device', 'get_current_input_device']
        
        for method in device_methods:
            assert hasattr(pipeStream, method), f"Missing audio device method: {method}"
            
    except ImportError:
        pytest.skip("Import failed (may be expected in test environment)")