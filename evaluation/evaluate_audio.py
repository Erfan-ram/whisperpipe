#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated Evaluation with Pre-recorded Audio

This script allows testing both implementations using pre-recorded audio files
for reproducible results without requiring live microphone input.

Usage:
    python evaluate_audio.py --audio sample.wav --model base
    python evaluate_audio.py --generate-sample  # Creates test audio
"""

import argparse
import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import whisper
except ImportError:
    print("Error: whisper not installed. Run: pip install openai-whisper")
    sys.exit(1)


def generate_sample_audio(output_path="test_audio.wav", duration=30):
    """
    Generate a sample audio file for testing
    
    Args:
        output_path: Where to save the WAV file
        duration: Duration in seconds
    """
    try:
        from scipy.io import wavfile
    except ImportError:
        print("Error: scipy required for audio generation")
        print("Install with: pip install scipy")
        return False
    
    # Generate simple sine wave test signal
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple test signal (mixture of frequencies)
    audio = np.zeros_like(t)
    for freq in [440, 880, 1320]:  # A4, A5, E6
        audio += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    
    wavfile.write(output_path, sample_rate, audio)
    print(f"Generated test audio: {output_path}")
    print(f"Duration: {duration}s")
    print(f"Sample rate: {sample_rate}Hz")
    
    return True


def load_audio_file(audio_path):
    """
    Load audio file using whisper's audio loading
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        np.array: Audio data at 16kHz
    """
    try:
        audio = whisper.load_audio(audio_path)
        return audio
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None


def evaluate_naive_on_audio(audio_data, model_name="base", language="en"):
    """
    Evaluate naive implementation on pre-recorded audio
    
    Args:
        audio_data: Audio numpy array (16kHz)
        model_name: Whisper model name
        language: Language code
        
    Returns:
        dict: Metrics from evaluation
    """
    print("\n" + "="*60)
    print("EVALUATING NAIVE BASELINE")
    print("="*60)
    
    # Load model
    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name)
    
    # Simulate streaming by processing in chunks
    chunk_duration = 1.0  # Process 1 second at a time
    sample_rate = 16000
    chunk_size = int(sample_rate * chunk_duration)
    
    buffer = np.array([], dtype=np.float32)
    transcription_history = []
    edit_count = 0
    last_text = ""
    processing_times = []
    
    total_chunks = len(audio_data) // chunk_size
    print(f"Processing {total_chunks} chunks...")
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        buffer = np.append(buffer, chunk)
        
        # Limit buffer size (30 seconds)
        max_buffer = sample_rate * 30
        if len(buffer) > max_buffer:
            buffer = buffer[-max_buffer:]
        
        # Process if enough audio
        if len(buffer) >= sample_rate:
            start_time = time.time()
            
            # NAIVE: Re-transcribe entire buffer
            result = model.transcribe(buffer, language=language, fp16=False)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            new_text = result['text'].strip()
            
            # Track changes
            if new_text and new_text != last_text:
                old_words = set(last_text.split())
                new_words = set(new_text.split())
                edit_count += len(new_words - old_words) + len(old_words - new_words)
                
                transcription_history.append(new_text)
                last_text = new_text
                
                # Progress indicator
                progress = (i / len(audio_data)) * 100
                print(f"  {progress:.0f}% - Buffer: {len(buffer)/sample_rate:.1f}s - "
                      f"Time: {processing_time:.2f}s - Text: {new_text[:50]}...")
    
    # Calculate metrics
    final_word_count = len(last_text.split())
    edit_overhead = edit_count / final_word_count if final_word_count > 0 else 0
    
    # Stability
    stability = 0.0
    if len(transcription_history) > 1:
        changes = sum(1 for i in range(1, len(transcription_history)) 
                     if transcription_history[i] != transcription_history[i-1])
        stability = (1 - changes / (len(transcription_history) - 1)) * 100
    
    metrics = {
        'edit_overhead': edit_overhead,
        'edit_count': edit_count,
        'final_word_count': final_word_count,
        'stability': stability,
        'transcription_count': len(transcription_history),
        'avg_processing_time': np.mean(processing_times),
        'max_processing_time': np.max(processing_times),
        'final_text': last_text
    }
    
    print("\nNaive Baseline Results:")
    print(f"  Edit Overhead: {edit_overhead:.2f}×")
    print(f"  Stability: {stability:.1f}%")
    print(f"  Avg Processing Time: {np.mean(processing_times):.3f}s")
    print(f"  Final Text: {last_text}")
    
    return metrics


def evaluate_whisperpipe_simulation(audio_data, model_name="base", language="en"):
    """
    Simulate WhisperPipe behavior on pre-recorded audio
    
    This is a simplified simulation that demonstrates the dual-buffer concept
    without full WhisperPipe implementation.
    
    Args:
        audio_data: Audio numpy array (16kHz)
        model_name: Whisper model name
        language: Language code
        
    Returns:
        dict: Metrics from evaluation
    """
    print("\n" + "="*60)
    print("SIMULATING WHISPERPIPE")
    print("="*60)
    
    # Load model
    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name)
    
    # WhisperPipe approach: Dual buffer
    chunk_duration = 1.0
    sample_rate = 16000
    chunk_size = int(sample_rate * chunk_duration)
    
    stable_text = ""
    active_buffer = np.array([], dtype=np.float32)
    
    transcription_history = []
    commit_count = 0
    processing_times = []
    last_active_text = ""
    stability_samples = []
    
    total_chunks = len(audio_data) // chunk_size
    print(f"Processing {total_chunks} chunks...")
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        active_buffer = np.append(active_buffer, chunk)
        
        # Process active buffer
        if len(active_buffer) >= sample_rate:
            start_time = time.time()
            
            # Only transcribe active buffer (not entire history)
            result = model.transcribe(active_buffer, language=language, fp16=False)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            new_text = result['text'].strip()
            transcription_history.append(new_text)
            
            # Stability tracking
            if last_active_text and new_text == last_active_text:
                stability_samples.append(1)
                
                # Simulate commit after 3 confirmations
                if len(stability_samples) >= 3 and sum(stability_samples[-3:]) == 3:
                    # Commit to stable buffer
                    if stable_text:
                        stable_text += " " + new_text
                    else:
                        stable_text = new_text
                    
                    commit_count += 1
                    
                    # Clear active buffer (simulate audio removal)
                    active_buffer = np.array([], dtype=np.float32)
                    
                    print(f"  COMMIT #{commit_count}: {new_text[:50]}...")
            else:
                stability_samples.append(0)
            
            last_active_text = new_text
            
            # Progress
            progress = (i / len(audio_data)) * 100
            print(f"  {progress:.0f}% - Active: {len(active_buffer)/sample_rate:.1f}s - "
                  f"Time: {processing_time:.2f}s")
            
            # Limit active buffer
            max_active = sample_rate * 25
            if len(active_buffer) > max_active:
                active_buffer = active_buffer[-max_active:]
    
    # Calculate metrics
    final_text = stable_text if stable_text else last_active_text
    final_word_count = len(final_text.split())
    
    # Edit overhead based on commits
    edit_overhead = commit_count / final_word_count if final_word_count > 0 else 0
    
    # Stability
    stability = (sum(stability_samples) / len(stability_samples) * 100) if stability_samples else 0
    
    metrics = {
        'edit_overhead': edit_overhead,
        'commit_count': commit_count,
        'final_word_count': final_word_count,
        'stability': stability,
        'transcription_count': len(transcription_history),
        'avg_processing_time': np.mean(processing_times),
        'max_processing_time': np.max(processing_times),
        'final_text': final_text
    }
    
    print("\nWhisperPipe Simulation Results:")
    print(f"  Commits: {commit_count}")
    print(f"  Stability: {stability:.1f}%")
    print(f"  Avg Processing Time: {np.mean(processing_times):.3f}s")
    print(f"  Final Text: {final_text}")
    
    return metrics


def print_comparison(naive_metrics, pipe_metrics):
    """Print comparison between implementations"""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print("\n📊 EDIT OVERHEAD")
    print(f"  Naive:      {naive_metrics['edit_overhead']:.2f}×")
    print(f"  WhisperPipe: {pipe_metrics['edit_overhead']:.2f}×")
    
    if naive_metrics['edit_overhead'] > 0:
        reduction = ((naive_metrics['edit_overhead'] - pipe_metrics['edit_overhead']) / 
                    naive_metrics['edit_overhead']) * 100
        print(f"  Reduction:  {reduction:.1f}%")
    
    print("\n✅ STABILITY")
    print(f"  Naive:      {naive_metrics['stability']:.1f}%")
    print(f"  WhisperPipe: {pipe_metrics['stability']:.1f}%")
    print(f"  Improvement: +{pipe_metrics['stability'] - naive_metrics['stability']:.1f} pp")
    
    print("\n⚡ PROCESSING TIME")
    print(f"  Naive Avg:  {naive_metrics['avg_processing_time']:.3f}s")
    print(f"  WhisperPipe Avg: {pipe_metrics['avg_processing_time']:.3f}s")
    
    if naive_metrics['avg_processing_time'] > 0:
        speedup = naive_metrics['avg_processing_time'] / pipe_metrics['avg_processing_time']
        print(f"  Speedup:    {speedup:.2f}×")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate implementations on audio files'
    )
    parser.add_argument(
        '--audio',
        type=str,
        help='Path to audio file (WAV, MP3, etc.)'
    )
    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Generate a sample audio file for testing'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model (default: base)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language code (default: en)'
    )
    
    args = parser.parse_args()
    
    if args.generate_sample:
        generate_sample_audio("test_audio.wav", duration=30)
        print("\nNow run: python evaluate_audio.py --audio test_audio.wav")
        return
    
    if not args.audio:
        print("Error: --audio required (or use --generate-sample)")
        print("\nUsage:")
        print("  python evaluate_audio.py --audio yourfile.wav")
        print("  python evaluate_audio.py --generate-sample")
        return
    
    # Load audio
    print(f"Loading audio: {args.audio}")
    audio = load_audio_file(args.audio)
    
    if audio is None:
        return
    
    duration = len(audio) / 16000
    print(f"Duration: {duration:.1f}s")
    
    # Evaluate both
    naive_metrics = evaluate_naive_on_audio(audio, args.model, args.language)
    pipe_metrics = evaluate_whisperpipe_simulation(audio, args.model, args.language)
    
    # Compare
    print_comparison(naive_metrics, pipe_metrics)


if __name__ == "__main__":
    main()
