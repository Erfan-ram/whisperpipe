#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Syntax Validation Script

This script validates that all Python files in the evaluation framework
have correct syntax without requiring external dependencies.
"""

import ast
import sys
import os
from pathlib import Path


def check_file_syntax(filepath):
    """
    Check if a Python file has valid syntax
    
    Args:
        filepath: Path to Python file
        
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Try to parse as AST
        ast.parse(source, filename=str(filepath))
        return True, None
        
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def validate_evaluation_framework():
    """Validate all Python files in the evaluation framework"""
    
    # Find evaluation directory
    script_dir = Path(__file__).parent
    
    # Files to check
    files_to_check = [
        script_dir / 'naive_whisper.py',
        script_dir / 'metrics.py',
        script_dir / 'compare.py',
        script_dir / 'evaluate_audio.py',
        script_dir / 'validate_syntax.py',  # This file
        script_dir / '__init__.py',
    ]
    
    print("="*60)
    print("SYNTAX VALIDATION FOR EVALUATION FRAMEWORK")
    print("="*60)
    
    all_valid = True
    results = []
    
    for filepath in files_to_check:
        if not filepath.exists():
            results.append((filepath.name, False, "File not found"))
            all_valid = False
            continue
        
        success, error = check_file_syntax(filepath)
        results.append((filepath.name, success, error))
        
        if not success:
            all_valid = False
    
    # Print results
    print("\nResults:")
    print("-"*60)
    
    for filename, success, error in results:
        status = "✓" if success else "✗"
        print(f"{status} {filename}")
        if error:
            print(f"  Error: {error}")
    
    print("-"*60)
    
    if all_valid:
        print("\n✓ All files have valid syntax!")
        return 0
    else:
        print("\n✗ Some files have syntax errors")
        return 1


def validate_imports():
    """Check what imports are available"""
    
    print("\n" + "="*60)
    print("CHECKING AVAILABLE IMPORTS")
    print("="*60)
    
    required = [
        'numpy',
        'whisper',
        'torch',
        'pyaudio',
        'pynput'
    ]
    
    optional = [
        'scipy',
        'sounddevice'
    ]
    
    print("\nRequired packages:")
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (REQUIRED - install with: pip install {package})")
    
    print("\nOptional packages:")
    for package in optional:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  - {package} (optional - needed for some features)")


def check_code_structure():
    """Perform basic code structure checks"""
    
    print("\n" + "="*60)
    print("CODE STRUCTURE CHECKS")
    print("="*60)
    
    script_dir = Path(__file__).parent
    
    # Check naive_whisper.py
    naive_file = script_dir / 'naive_whisper.py'
    if naive_file.exists():
        with open(naive_file, 'r') as f:
            content = f.read()
        
        checks = [
            ('NaiveWhisperStream class', 'class NaiveWhisperStream' in content),
            ('start_streaming method', 'def start_streaming' in content),
            ('stop_streaming method', 'def stop_streaming' in content),
            ('get_metrics method', 'def get_metrics' in content),
            ('_process_audio method', 'def _process_audio' in content),
        ]
        
        print("\nnaive_whisper.py:")
        for check_name, result in checks:
            status = "✓" if result else "✗"
            print(f"  {status} {check_name}")
    
    # Check metrics.py
    metrics_file = script_dir / 'metrics.py'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            content = f.read()
        
        checks = [
            ('MetricsTracker class', 'class MetricsTracker' in content),
            ('calculate_edit_overhead', 'def calculate_edit_overhead' in content),
            ('calculate_stability', 'def calculate_stability' in content),
            ('calculate_mean_commit_latency', 'def calculate_mean_commit_latency' in content),
            ('get_comprehensive_metrics', 'def get_comprehensive_metrics' in content),
        ]
        
        print("\nmetrics.py:")
        for check_name, result in checks:
            status = "✓" if result else "✗"
            print(f"  {status} {check_name}")
    
    # Check compare.py
    compare_file = script_dir / 'compare.py'
    if compare_file.exists():
        with open(compare_file, 'r') as f:
            content = f.read()
        
        checks = [
            ('compare_implementations function', 'def compare_implementations' in content),
            ('print_comparison function', 'def print_comparison' in content),
            ('main function', 'def main' in content),
            ('Imports NaiveWhisperStream', 'from evaluation.naive_whisper import NaiveWhisperStream' in content),
            ('Imports pipeStream', 'from whisperpipe import pipeStream' in content),
        ]
        
        print("\ncompare.py:")
        for check_name, result in checks:
            status = "✓" if result else "✗"
            print(f"  {status} {check_name}")


def main():
    """Main validation function"""
    
    # Syntax validation
    exit_code = validate_evaluation_framework()
    
    # Import checks (informational)
    validate_imports()
    
    # Structure checks
    check_code_structure()
    
    print("\n" + "="*60)
    
    if exit_code == 0:
        print("✓ VALIDATION PASSED")
        print("\nNext steps:")
        print("1. Install required dependencies: pip install -e .")
        print("2. Run comparison: python compare.py --duration 30 --model tiny")
        print("3. Check QUICKSTART.md for detailed instructions")
    else:
        print("✗ VALIDATION FAILED")
        print("\nPlease fix syntax errors before proceeding")
    
    print("="*60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
