#!/usr/bin/env python3
"""
Source code test to verify the whisperpipe package API without loading dependencies.
"""

import re
import os

def test_api_in_source():
    """Test the API signature by parsing the source code"""
    
    print("🧪 Testing whisperpipe API signature from source...")
    
    # Read the core.py file
    core_file = "whisperpipe/core.py"
    if not os.path.exists(core_file):
        print(f"❌ Core file not found: {core_file}")
        return False
    
    with open(core_file, 'r') as f:
        content = f.read()
    
    # Test 1: Check class name
    print("1. Testing class name...")
    if 'class whisperpipe:' in content:
        print("✅ Found class: whisperpipe")
    else:
        print("❌ Class 'whisperpipe' not found")
        return False
    
    # Test 2: Check constructor signature
    print("2. Testing constructor signature...")
    
    # Find the __init__ method
    init_pattern = r'def __init__\(self,([^)]+)\):'
    match = re.search(init_pattern, content)
    
    if not match:
        print("❌ Constructor not found")
        return False
    
    params_str = match.group(1)
    print(f"   Found parameters: {params_str.strip()}")
    
    # Check for required parameters
    required_params = ['model_name', 'language', 'finalization_delay', 'processing_interval']
    found_params = []
    
    for param in required_params:
        if param in params_str:
            found_params.append(param)
            print(f"✅ Found parameter: {param}")
        else:
            print(f"❌ Missing parameter: {param}")
    
    if len(found_params) == len(required_params):
        print("✅ All required parameters found")
        api_ok = True
    else:
        print(f"❌ Missing {len(required_params) - len(found_params)} parameters")
        api_ok = False
    
    # Test 3: Check default values
    print("3. Testing default values...")
    
    # Check specific defaults
    defaults_check = {
        'model_name="base.en"': 'model_name default',
        'language="en"': 'language default', 
        'finalization_delay=10.0': 'finalization_delay default',
        'processing_interval=1.0': 'processing_interval default'
    }
    
    defaults_ok = True
    for default_pattern, desc in defaults_check.items():
        if default_pattern in params_str:
            print(f"✅ Found {desc}")
        else:
            print(f"⚠️  Default for {desc} may not match expected")
            # Don't fail for this, just warn
    
    # Test 4: Check that language parameter is used in transcription
    print("4. Testing language parameter usage...")
    if 'language=self.language' in content:
        print("✅ Language parameter is used in transcription")
    else:
        print("❌ Language parameter not used in transcription")
        api_ok = False
    
    # Test 5: Check finalization_delay and processing_interval usage
    print("5. Testing parameter assignments...")
    if 'self.finalization_delay = finalization_delay' in content:
        print("✅ finalization_delay parameter assigned")
    else:
        print("❌ finalization_delay parameter not assigned")
        api_ok = False
        
    if 'self.processing_interval = processing_interval' in content:
        print("✅ processing_interval parameter assigned")
    else:
        print("❌ processing_interval parameter not assigned")
        api_ok = False
    
    return api_ok


def test_package_files():
    """Test that all required package files exist"""
    
    print("\n🏗️  Testing package file structure...")
    
    required_files = [
        'whisperpipe/__init__.py',
        'whisperpipe/core.py',
        'pyproject.toml',
        'README.md',
        'LICENSE'
    ]
    
    files_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            files_ok = False
    
    # Test pyproject.toml content
    if os.path.exists('pyproject.toml'):
        print("6. Testing pyproject.toml content...")
        with open('pyproject.toml', 'r') as f:
            toml_content = f.read()
        
        toml_checks = [
            ('name = "whisperpipe"', 'Package name'),
            ('version = "1.0.0"', 'Version'),
            ('Erfan Ramezani', 'Author'),
            ('openai-whisper', 'Whisper dependency'),
            ('pyaudio', 'PyAudio dependency'),
            ('numpy', 'NumPy dependency')
        ]
        
        for check, desc in toml_checks:
            if check in toml_content:
                print(f"✅ Found {desc} in pyproject.toml")
            else:
                print(f"⚠️  {desc} may be missing from pyproject.toml")
    
    return files_ok


def test_dist_files():
    """Test that package builds exist"""
    
    print("\n📦 Testing built packages...")
    
    dist_files = [
        'dist/whisperpipe-1.0.0-py3-none-any.whl',
        'dist/whisperpipe-1.0.0.tar.gz'
    ]
    
    dist_ok = True
    for file_path in dist_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"⚠️  Not found: {file_path} (run 'poetry build' to create)")
            dist_ok = False
    
    return dist_ok


if __name__ == "__main__":
    print("🔬 whisperpipe Package Source Code Test")
    print("=" * 50)
    
    api_ok = test_api_in_source()
    files_ok = test_package_files()
    dist_ok = test_dist_files()
    
    print("\n📊 Test Summary:")
    print(f"   API signature: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"   Package files: {'✅ PASS' if files_ok else '❌ FAIL'}")
    print(f"   Built packages: {'✅ PASS' if dist_ok else '⚠️  MISSING'}")
    
    if api_ok and files_ok:
        print("\n🎉 whisperpipe package API is ready!")
        print("\n📖 Installation instructions:")
        print("   # From PyPI (when published):")
        print("   pip install whisperpipe")
        print("\n   # From GitHub:")
        print("   pip install git+https://github.com/Erfan-ram/whisperpipe.git")
        print("\n📖 Usage example:")
        print("   from whisperpipe import whisperpipe")
        print("   transcriber = whisperpipe(")
        print("       model_name='base.en',")
        print("       language='en',")
        print("       finalization_delay=10.0,")
        print("       processing_interval=1.0")
        print("   )")
        print("   transcriber.start_streaming()")
    else:
        print("\n❌ Package needs fixes before it's ready")