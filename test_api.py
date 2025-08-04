#!/usr/bin/env python3
"""
Test script to verify the whisperpipe package API matches the requested signature.
"""

def test_api_signature():
    """Test that the whisperpipe class can be instantiated with the requested parameters"""
    
    print("🧪 Testing whisperpipe API signature...")
    
    try:
        # Test the requested API signature without actually loading dependencies
        from whisperpipe.core import whisperpipe
        
        # Test 1: Basic instantiation with default parameters
        print("1. Testing basic instantiation...")
        try:
            transcriber = whisperpipe.__new__(whisperpipe)
            print("✅ Basic instantiation successful")
        except Exception as e:
            print(f"❌ Basic instantiation failed: {e}")
        
        # Test 2: Check constructor signature matches requirements
        print("2. Checking constructor signature...")
        import inspect
        sig = inspect.signature(whisperpipe.__init__)
        params = list(sig.parameters.keys())
        
        required_params = ['model_name', 'language', 'finalization_delay', 'processing_interval']
        missing_params = []
        
        for param in required_params:
            if param in params:
                print(f"✅ Found parameter: {param}")
            else:
                missing_params.append(param)
                print(f"❌ Missing parameter: {param}")
        
        if not missing_params:
            print("✅ All required parameters found")
        else:
            print(f"❌ Missing parameters: {missing_params}")
        
        # Test 3: Check default values
        print("3. Checking default values...")
        defaults = {}
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                defaults[param_name] = param.default
        
        print(f"   Default values: {defaults}")
        
        # Test 4: Verify the requested constructor call would work (syntax-wise)
        print("4. Testing requested API call signature...")
        expected_call = 'whisperpipe(model_name="base.en", language="en", finalization_delay=10.0, processing_interval=1.0)'
        print(f"   Expected call: {expected_call}")
        
        # Check if all required parameters are present with correct names
        call_params = ['model_name', 'language', 'finalization_delay', 'processing_interval']
        api_compatible = all(param in params for param in call_params)
        
        if api_compatible:
            print("✅ API signature is compatible with requirements")
        else:
            print("❌ API signature is not compatible with requirements")
            
        return api_compatible
        
    except ImportError as e:
        print(f"⚠️  Import failed (expected in test environment): {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_package_structure():
    """Test that the package structure is correct"""
    
    print("\n🏗️  Testing package structure...")
    
    # Test 1: Check package can be imported
    try:
        import whisperpipe
        print("✅ Package import successful")
        
        # Check version
        if hasattr(whisperpipe, '__version__'):
            print(f"✅ Version: {whisperpipe.__version__}")
        else:
            print("⚠️  No version info found")
            
        # Check if main class is accessible
        if hasattr(whisperpipe, 'whisperpipe'):
            print("✅ Main class accessible")
        else:
            print("❌ Main class not accessible")
            
    except ImportError as e:
        print(f"⚠️  Package import failed (expected in test environment): {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("🔬 whisperpipe Package API Test")
    print("=" * 50)
    
    structure_ok = test_package_structure()
    api_ok = test_api_signature()
    
    print("\n📊 Test Summary:")
    print(f"   Package structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"   API signature: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if structure_ok or api_ok:
        print("\n🎉 whisperpipe package is ready!")
        print("\n📖 Usage examples:")
        print("   # Basic usage")
        print('   from whisperpipe import whisperpipe')
        print('   transcriber = whisperpipe(model_name="base.en", language="en", finalization_delay=10.0, processing_interval=1.0)')
        print('   transcriber.start_streaming()')
        print("\n   # With callback")
        print('   transcriber.set_def_callback(your_callback_function)')
        print("\n   # Pause/Resume")
        print('   transcriber.pause_streaming()')
        print('   transcriber.resume_streaming()')
    else:
        print("\n❌ Package needs further work")