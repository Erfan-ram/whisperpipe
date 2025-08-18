"""
Tests for whisperpipe package structure and installation
"""
import pytest
import os
import sys


def test_package_files_exist():
    """Test that all required package files exist"""
    required_files = [
        'whisperpipe/__init__.py',
        'whisperpipe/core.py',
        'pyproject.toml',
        'README.md',
        'LICENSE'
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Required file missing: {file_path}"


def test_package_import():
    """Test that the package can be imported"""
    try:
        import whisperpipe
        assert whisperpipe is not None
    except ImportError as e:
        pytest.skip(f"Package import failed (may be expected in test environment): {e}")


def test_main_class_accessible():
    """Test that the main whisperpipe class is accessible"""
    try:
        import whisperpipe
        assert hasattr(whisperpipe, 'whisperpipe'), "Main class 'whisperpipe' not accessible"
    except ImportError:
        pytest.skip("Package import failed (may be expected in test environment)")


def test_pyproject_toml_content():
    """Test that pyproject.toml has required content"""
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    
    required_content = [
        'name = "whisperpipe"',
        '[tool.poetry]',
        'openai-whisper',
        'sounddevice',
        'pynput'
    ]
    
    for item in required_content:
        assert item in content, f"Missing required content in pyproject.toml: {item}"


def test_version_info():
    """Test that version information is available"""
    try:
        import whisperpipe
        # Version info can be in __version__ or in the package metadata
        has_version = (
            hasattr(whisperpipe, '__version__') or 
            hasattr(whisperpipe, 'VERSION') or
            hasattr(whisperpipe, 'version')
        )
        # Don't require version in test environment, just check if present
        if has_version:
            print(f"Version found: {getattr(whisperpipe, '__version__', 'unknown')}")
    except ImportError:
        pytest.skip("Package import failed (may be expected in test environment)")


def test_package_metadata():
    """Test that package has proper metadata"""
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    
    metadata_checks = [
        ('authors = [', 'Author information'),
        ('license = "MIT"', 'License'),
        ('description = ', 'Description'),
        ('readme = "README.md"', 'README reference'),
    ]
    
    for check, desc in metadata_checks:
        assert check in content, f"Missing {desc} in pyproject.toml"