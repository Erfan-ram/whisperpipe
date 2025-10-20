"""
whisperpipe - Real-time speech-to-text streaming with OpenAI Whisper

A powerful, easy-to-use package for real-time audio transcription using OpenAI's Whisper model.
Features include callback support for LLM integration and pause/resume functionality.
"""

from .core import pipeStream

__version__ = "1.0.0"
__author__ = "Erfan Ramezani"
__email__ = "erfanramezany245@gmail.com"

__all__ = ["pipeStream"]
