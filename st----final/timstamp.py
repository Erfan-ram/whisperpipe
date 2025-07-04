import whisper
import torch
import numpy as np
from typing import List, Dict, Any
import json

def transcribe_with_word_timestamps(audio_path: str, model_name: str = "base") -> List[Dict[str, Any]]:
    """
    Transcribe audio file with word-level timestamps using Whisper.
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Whisper model to use ("tiny", "base", "small", "medium", "large")
    
    Returns:
        List[Dict]: List of words with timestamps
    """
    
    # Load the Whisper model
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    # Transcribe with word timestamps
    print("Transcribing audio...")
    result = model.transcribe(
        audio_path,
        word_timestamps=True,  # Enable word-level timestamps
        verbose=True
    )
    
    # Extract word-level information
    words_with_timestamps = []
    
    for segment in result["segments"]:
        if "words" in segment:
            for word_info in segment["words"]:
                word_data = {
                    "word": word_info["word"].strip(),
                    "start": round(word_info["start"], 3),
                    "end": round(word_info["end"], 3),
                    "confidence": word_info.get("probability", 0.0)
                }
                words_with_timestamps.append(word_data)
    
    return words_with_timestamps

def print_words_with_timestamps(words: List[Dict[str, Any]]) -> None:
    """
    Print each word with its timestamp information.
    
    Args:
        words (List[Dict]): List of words with timestamp data
    """
    print("\n" + "="*60)
    print("WORD-LEVEL TIMESTAMPS")
    print("="*60)
    print(f"{'Word':<20} {'Start (s)':<12} {'End (s)':<12} {'Duration (s)':<12} {'Confidence':<12}")
    print("-" * 68)
    
    for word_info in words:
        word = word_info["word"]
        start = word_info["start"]
        end = word_info["end"]
        duration = round(end - start, 3)
        confidence = round(word_info["confidence"], 3)
        
        print(f"{word:<20} {start:<12} {end:<12} {duration:<12} {confidence:<12}")

def print_real_time_format(words: List[Dict[str, Any]]) -> None:
    """
    Print words in a real-time format showing when each word appears.
    
    Args:
        words (List[Dict]): List of words with timestamp data
    """
    print("\n" + "="*60)
    print("REAL-TIME WORD APPEARANCE")
    print("="*60)
    
    for word_info in words:
        word = word_info["word"]
        start = word_info["start"]
        
        # Format: [MM:SS.mmm] word
        minutes = int(start // 60)
        seconds = start % 60
        print(f"[{minutes:02d}:{seconds:06.3f}] {word}")

def save_timestamps_to_json(words: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save word timestamps to a JSON file.
    
    Args:
        words (List[Dict]): List of words with timestamp data
        output_path (str): Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(words, f, indent=2, ensure_ascii=False)
    print(f"\nTimestamps saved to: {output_path}")

def create_srt_subtitle(words: List[Dict[str, Any]], output_path: str, words_per_subtitle: int = 5) -> None:
    """
    Create SRT subtitle file from word timestamps.
    
    Args:
        words (List[Dict]): List of words with timestamp data
        output_path (str): Path to save the SRT file
        words_per_subtitle (int): Number of words per subtitle entry
    """
    
    def format_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        subtitle_index = 1
        
        for i in range(0, len(words), words_per_subtitle):
            chunk = words[i:i + words_per_subtitle]
            
            start_time = chunk[0]["start"]
            end_time = chunk[-1]["end"]
            text = " ".join([word["word"] for word in chunk])
            
            f.write(f"{subtitle_index}\n")
            f.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            f.write(f"{text}\n\n")
            
            subtitle_index += 1
    
    print(f"SRT subtitle file saved to: {output_path}")

def main():
    """
    Main function to demonstrate word-level timestamp extraction.
    """
    # Configuration
    audio_file = "recorded.wav"  # Replace with your audio file path
    model_size = "tiny"  # Options: "tiny", "base", "small", "medium", "large"
    
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Transcribe audio with word timestamps
        words = transcribe_with_word_timestamps(audio_file, model_size)
        
        if not words:
            print("No words detected in the audio file.")
            return
        
        # Print results in different formats
        print_words_with_timestamps(words)
        print_real_time_format(words)
        
        # Save results
        save_timestamps_to_json(words, "word_timestamps.json")
        create_srt_subtitle(words, "subtitles.srt", words_per_subtitle=5)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total words: {len(words)}")
        print(f"Audio duration: {words[-1]['end']:.3f} seconds")
        print(f"Average word duration: {np.mean([w['end'] - w['start'] for w in words]):.3f} seconds")
        
    except FileNotFoundError:
        print(f"Error: Audio file '{audio_file}' not found.")
        print("Please update the 'audio_file' variable with the correct path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Advanced usage example
def analyze_speech_patterns(words: List[Dict[str, Any]]) -> None:
    """
    Analyze speech patterns from word timestamps.
    
    Args:
        words (List[Dict]): List of words with timestamp data
    """
    print("\n" + "="*60)
    print("SPEECH PATTERN ANALYSIS")
    print("="*60)
    
    # Calculate speaking rate (words per minute)
    total_duration = words[-1]["end"] - words[0]["start"]
    words_per_minute = (len(words) / total_duration) * 60
    
    # Find pauses between words
    pauses = []
    for i in range(1, len(words)):
        pause_duration = words[i]["start"] - words[i-1]["end"]
        if pause_duration > 0.1:  # Consider pauses longer than 100ms
            pauses.append({
                "after_word": words[i-1]["word"],
                "before_word": words[i]["word"],
                "duration": round(pause_duration, 3),
                "timestamp": words[i-1]["end"]
            })
    
    # Find longest and shortest words by duration
    word_durations = [(w["word"], w["end"] - w["start"]) for w in words]
    longest_word = max(word_durations, key=lambda x: x[1])
    shortest_word = min(word_durations, key=lambda x: x[1])
    
    print(f"Speaking rate: {words_per_minute:.1f} words per minute")
    print(f"Total pauses detected: {len(pauses)}")
    print(f"Longest word: '{longest_word[0]}' ({longest_word[1]:.3f}s)")
    print(f"Shortest word: '{shortest_word[0]}' ({shortest_word[1]:.3f}s)")
    
    if pauses:
        print(f"Average pause duration: {np.mean([p['duration'] for p in pauses]):.3f}s")
        
        # Show significant pauses
        significant_pauses = [p for p in pauses if p["duration"] > 0.5]
        if significant_pauses:
            print(f"\nSignificant pauses (>0.5s):")
            for pause in significant_pauses[:5]:  # Show first 5
                minutes = int(pause["timestamp"] // 60)
                seconds = pause["timestamp"] % 60
                print(f"  [{minutes:02d}:{seconds:06.3f}] {pause['duration']}s after '{pause['after_word']}'")

if __name__ == "__main__":
    # Example usage with analysis
    print("Whisper Word-Level Timestamp Extractor")
    print("====================================")
    
    # Uncomment and modify the following lines to use:
    main()
    
    # For advanced analysis, you can also use:
    # words = transcribe_with_word_timestamps("your_audio.wav", "base")
    # analyze_speech_patterns(words)
    
    print("\nTo use this script:")
    print("1. Install required packages: pip install openai-whisper torch")
    print("2. Update the 'audio_file' variable with your audio file path")
    print("3. Run the script")