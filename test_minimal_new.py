#!/usr/bin/env python3
"""
Test script to validate the API structure and new features
without requiring full audio dependencies.
"""

import sys
import os

# Add the whisperpipe directory to Python path
sys.path.insert(0, '/home/runner/work/Audio2Text/Audio2Text')

def test_api_structure():
    """Test that the whisperpipe API has the expected structure"""
    print("🧪 Testing whisperpipe API structure...")
    
    try:
        # Test that we can import the package structure
        import whisperpipe
        print("✅ whisperpipe package imports successfully")
        
        # Check that whisperpipe class is accessible
        assert hasattr(whisperpipe, 'whisperpipe'), "whisperpipe class not found in package"
        print("✅ whisperpipe class is accessible")
        
        # Test the class has the expected constructor signature
        import inspect
        constructor_signature = inspect.signature(whisperpipe.whisperpipe.__init__)
        expected_params = ['self', 'model_name', 'language', 'finalization_delay', 'processing_interval']
        actual_params = list(constructor_signature.parameters.keys())
        
        for param in expected_params:
            if param not in actual_params:
                print(f"❌ Missing expected parameter: {param}")
                return False
        
        print("✅ Constructor signature matches expected API")
        
        # Test the class has the expected methods
        expected_methods = [
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
        
        for method in expected_methods:
            if not hasattr(whisperpipe.whisperpipe, method):
                print(f"❌ Missing expected method: {method}")
                return False
        
        print("✅ All expected methods are present")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_handling():
    """Test that import failures are handled gracefully"""
    print("\n🧪 Testing import error handling...")
    
    try:
        # Test that the core module handles missing dependencies gracefully
        from whisperpipe.core import SOUNDDEVICE_AVAILABLE
        print(f"✅ sounddevice availability detected: {SOUNDDEVICE_AVAILABLE}")
        
        return True
    except Exception as e:
        print(f"❌ Import handling test failed: {e}")
        return False

def test_signal_handling_setup():
    """Test that signal handling is set up in the class"""
    print("\n🧪 Testing signal handling setup...")
    
    try:
        # We can't fully test signal handling without dependencies, 
        # but we can check the method exists
        from whisperpipe.core import whisperpipe
        
        # Check that _setup_signal_handling method exists
        assert hasattr(whisperpipe, '_setup_signal_handling'), "Signal handling setup method not found"
        print("✅ Signal handling setup method exists")
        
        return True
    except Exception as e:
        print(f"❌ Signal handling test failed: {e}")
        return False

def test_code_structure():
    """Test that the code structure changes were made correctly"""
    print("\n🧪 Testing code structure changes...")
    
    try:
        # Read the core.py file and check that main() and signal_handler functions were removed
        with open('/home/erfan/Desktop/Rag-zone/Audio2Text/whisperpipe/core.py', 'r') as f:
            content = f.read()
        
        # Check that the old top-level functions are not present
        if 'def main():' in content:
            print("❌ top-level main() function still present - should be removed")
            return False
        
        # Check for the old standalone signal_handler (not the one inside the class)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() == 'def signal_handler(sig, frame):':
                # Check if this is at the top level (not indented inside a class/method)
                if not line.startswith('    ') and not line.startswith('\t'):
                    print("❌ top-level signal_handler() function still present - should be removed")
                    return False
        
        if 'if __name__ == "__main__":' in content:
            print("❌ if __name__ == '__main__' block still present - should be removed")
            return False
        
        print("✅ Old main() and top-level signal_handler functions removed correctly")
        
        # Check that new methods were added
        required_new_methods = [
            'def input_devices(self):',
            'def set_input_device(self, device_id):',
            'def get_current_input_device(self):',
            'def _setup_signal_handling(self):'
        ]
        
        for method in required_new_methods:
            if method not in content:
                print(f"❌ New method not found: {method}")
                return False
        
        print("✅ New audio device management methods added")
        print("✅ Signal handling setup method added")
        
        # Check that signal handling is now done within the class
        if 'def signal_handler(sig, frame):' in content and '_setup_signal_handling' in content:
            print("✅ Signal handling moved inside class method")
        
        return True
    except Exception as e:
        print(f"❌ Code structure test failed: {e}")
        return False

def test_pyproject_changes():
    """Test that pyproject.toml was updated correctly"""
    print("\n🧪 Testing pyproject.toml changes...")
    
    try:
        with open('pyproject.toml', 'r') as f:
            content = f.read()
        
        if 'sounddevice = "^0.4.6"' not in content:
            print("❌ sounddevice dependency not added to pyproject.toml")
            return False
        
        print("✅ sounddevice dependency added to pyproject.toml")
        return True
    except Exception as e:
        print(f"❌ pyproject.toml test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 Testing whisperpipe implementation changes")
    print("=" * 50)
    
    tests = [
        test_pyproject_changes,
        test_code_structure,
        test_import_handling,
        test_signal_handling_setup,
        test_api_structure,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! Implementation looks good.")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    main()