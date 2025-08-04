# Publishing Guide for whisperpipe

This guide explains how to publish the whisperpipe package to PyPI and make it installable from GitHub.

## Prerequisites

1. Make sure you have Poetry installed
2. Have a PyPI account (create one at https://pypi.org/account/register/)
3. Have a TestPyPI account for testing (create one at https://test.pypi.org/account/register/)

## Package Structure

The package is now properly structured with:
- `whisperpipe/` - Main package directory
- `whisperpipe/__init__.py` - Package initialization and exports
- `whisperpipe/core.py` - Main implementation (renamed from main_stream.py)
- `pyproject.toml` - Poetry configuration and dependencies
- `README.md` - Package documentation
- `LICENSE` - MIT license
- `MANIFEST.in` - Additional files to include in distribution

## Testing the Package

Before publishing, run the tests to ensure everything works:

```bash
# Test the API and package structure
python test_package.py

# Test functionality (these use mock implementations)
python test_minimal.py
python test_integration.py
```

## Building the Package

```bash
# Build the package distributions
poetry build
```

This creates:
- `dist/whisperpipe-1.0.0-py3-none-any.whl` (wheel)
- `dist/whisperpipe-1.0.0.tar.gz` (source distribution)

## Publishing to PyPI

### Step 1: Test on TestPyPI first

```bash
# Configure TestPyPI repository
poetry config repositories.testpypi https://test.pypi.org/legacy/

# Get your TestPyPI API token from https://test.pypi.org/manage/account/
# Then configure it:
poetry config pypi-token.testpypi pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Publish to TestPyPI
poetry publish -r testpypi
```

### Step 2: Test installation from TestPyPI

```bash
# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ whisperpipe

# Test the installation
python -c "from whisperpipe import whisperpipe; print('Success!')"
```

### Step 3: Publish to production PyPI

```bash
# Get your PyPI API token from https://pypi.org/manage/account/
poetry config pypi-token.pypi pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Publish to PyPI
poetry publish
```

## GitHub Installation

After pushing to GitHub, users can install directly from the repository:

```bash
pip install git+https://github.com/Erfan-ram/whisperpipe.git
```

## Repository Setup for GitHub Installation

1. Make sure your repository is public or accessible to users
2. The repository name should be changed to `whisperpipe` for consistency
3. Add installation instructions to the main README.md

## Usage After Installation

Once published, users can install and use the package as requested:

```python
# Install the package
# pip install whisperpipe

# Use the package
from whisperpipe import whisperpipe

# Create transcriber with the requested API
transcriber = whisperpipe(
    model_name="base.en",
    language="en",
    finalization_delay=10.0,
    processing_interval=1.0
)

# Set up callback for LLM integration (optional)
def my_llm_callback(text):
    print(f"Processing: {text}")
    # Your LLM integration here
    return "processed"

transcriber.set_def_callback(my_llm_callback)

# Start streaming
transcriber.start_streaming()

# Control streaming (optional)
# transcriber.pause_streaming()
# transcriber.resume_streaming()
# transcriber.stop_streaming()
```

## Version Updates

To release new versions:

1. Update the version in `pyproject.toml`
2. Update the version in `whisperpipe/__init__.py`
3. Build and publish:

```bash
poetry version patch  # or minor, major
poetry build
poetry publish
```

## Notes

- The package includes all necessary dependencies in `pyproject.toml`
- All tests pass and the API matches the requested signature
- The package is backward compatible with the original functionality
- Users can install from either PyPI or GitHub as requested