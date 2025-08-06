#!/usr/bin/env python3
"""
Minimal test that validates the implementation without requiring full dependencies
"""

import sys
import os

def test_basic_structure():
    """Test the basic package structure"""
    print("🧪 Testing basic package structure...")
    
    try:
        # Test that the file structure is correct
        assert os.path.exists('/home/runner/work/Audio2Text/Audio2Text/whisperpipe/__init__.py'), "Missing __init__.py"
        assert os.path.exists('/home/runner/work/Audio2Text/Audio2Text/whisperpipe/core.py'), "Missing core.py"
        assert os.path.exists('/home/runner/work/Audio2Text/Audio2Text/pyproject.toml'), "Missing pyproject.toml"
        assert os.path.exists('/home/runner/work/Audio2Text/Audio2Text/simple_usage.py'), "Missing simple_usage.py"
        
        print("✅ All required files present")
        return True
        
    except Exception as e:
        print(f"❌ Basic structure test failed: {e}")
        return False

def test_pyproject_dependencies():
    """Test that pyproject.toml has the right dependencies"""
    print("\n🧪 Testing pyproject.toml dependencies...")
    
    try:
        with open('/home/runner/work/Audio2Text/Audio2Text/pyproject.toml', 'r') as f:
            content = f.read()
        
        required_deps = [
            'sounddevice = "^0.4.6"',
            'openai-whisper = "^20231117"',
            'torch = "^2.0.0"',
            'pyaudio = "^0.2.11"',
            'numpy = "^1.21.0"',
            'pynput = "^1.7.6"'
        ]
        
        for dep in required_deps:
            if dep not in content:
                print(f"❌ Missing dependency: {dep}")
                return False
        
        print("✅ All required dependencies present in pyproject.toml")
        return True
        
    except Exception as e:
        print(f"❌ Dependencies test failed: {e}")
        return False

def test_code_cleanup():
    """Test that the old code was properly removed"""
    print("\n🧪 Testing code cleanup...")
    
    try:
        with open('/home/runner/work/Audio2Text/Audio2Text/whisperpipe/core.py', 'r') as f:
            content = f.read()
        
        # Test that old main execution code is removed
        if 'if __name__ == "__main__":' in content:
            print("❌ if __name__ == '__main__' block still present")
            return False
        
        # Test that standalone main() function is removed
        lines = content.split('\n')
        for line in lines:
            if line.strip() == 'def main():':
                if not line.startswith('    ') and not line.startswith('\t'):
                    print("❌ top-level main() function still present")
                    return False
        
        print("✅ Old main execution code properly removed")
        return True
        
    except Exception as e:
        print(f"❌ Code cleanup test failed: {e}")
        return False

def test_new_methods_added():
    """Test that new methods are present in the code"""
    print("\n🧪 Testing new methods presence...")
    
    try:
        with open('/home/runner/work/Audio2Text/Audio2Text/whisperpipe/core.py', 'r') as f:
            content = f.read()
        
        required_methods = [
            'def input_devices(self):',
            'def set_input_device(self, device_id):',
            'def get_current_input_device(self):',
            'def _setup_signal_handling(self):',
            'def _get_pyaudio_devices(self):'
        ]
        
        for method in required_methods:
            if method not in content:
                print(f"❌ Missing method: {method}")
                return False
        
        print("✅ All new methods added to the code")
        
        # Check for proper signal handling setup
        if 'signal.signal(signal.SIGINT, signal_handler)' in content:
            print("✅ Signal handling setup code present")
        else:
            print("❌ Signal handling setup code missing")
            return False
        
        # Check for sounddevice import handling
        if 'SOUNDDEVICE_AVAILABLE' in content:
            print("✅ Graceful sounddevice import handling present")
        else:
            print("❌ Sounddevice import handling missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ New methods test failed: {e}")
        return False

def test_simple_usage_example():
    """Test that the simple usage example was created"""
    print("\n🧪 Testing simple usage example...")
    
    try:
        with open('/home/runner/work/Audio2Text/Audio2Text/simple_usage.py', 'r') as f:
            content = f.read()
        
        # Check that it uses the new API
        if 'transcriber = whisperpipe(' in content:
            print("✅ Simple usage example uses new API")
        else:
            print("❌ Simple usage example doesn't use new API")
            return False
        
        # Check that it demonstrates device management
        if 'transcriber.input_devices()' in content:
            print("✅ Simple usage example shows device management")
        else:
            print("❌ Simple usage example missing device management")
            return False
        
        # Check that it mentions signal handling is built-in
        if 'signal handling is built-in' in content.lower() or 'signal handling is automatic' in content.lower():
            print("✅ Simple usage example mentions built-in signal handling")
        else:
            print("❌ Simple usage example doesn't mention built-in signal handling")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Simple usage example test failed: {e}")
        return False

def main():
    """Run all minimal tests"""
    print("🎯 Running minimal implementation tests")
    print("=" * 50)
    
    tests = [
        test_basic_structure,
        test_pyproject_dependencies,
        test_code_cleanup,
        test_new_methods_added,
        test_simple_usage_example,
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
    success = main()
    sys.exit(0 if success else 1)