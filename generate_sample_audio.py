#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a synthetic audio file for testing
This creates a simple audio file with beeps at different frequencies
"""

import numpy as np
import soundfile as sf

# Parameters
sample_rate = 16000
duration = 30  # 30 seconds

# Generate synthetic audio with multiple frequency components
t = np.linspace(0, duration, int(sample_rate * duration))
audio = np.zeros_like(t, dtype=np.float32)

# Add multiple frequency components to simulate speech-like audio
frequencies = [200, 400, 800, 1200, 1600]  # Hz
for i, freq in enumerate(frequencies):
    amplitude = 0.1 / len(frequencies)
    audio += amplitude * np.sin(2 * np.pi * freq * t)

# Add some variation to simulate speech patterns
envelope = np.abs(np.sin(2 * np.pi * 0.5 * t))  # 0.5 Hz modulation
audio = audio * envelope

# Normalize
audio = audio / np.max(np.abs(audio)) * 0.8

# Save to file
output_path = 'evaluation/sample.wav'
sf.write(output_path, audio, sample_rate)

print(f"Generated synthetic audio file: {output_path}")
print(f"Duration: {duration} seconds")
print(f"Sample rate: {sample_rate} Hz")
print("\nNote: This is synthetic audio. For real benchmarking, use actual speech audio files.")
