#!/usr/bin/env python3
"""
Simple usage example for whisperpipe package with built-in signal handling
and audio device management.

This example shows how easy it is to use whisperpipe without manual signal handling.
"""

from whisperpipe import pipeStream

def main():
    """
    Demonstration of the simplified whisperpipe usage with built-in signal handling
    and optional audio device selection.
    """
    print("🎯 Simple WhisperPipe Usage Example")
    print("=" * 50)
    
    # Create transcriber with the requested API
    transcriber = pipeStream(
        model_name="base",
        language="en",
        finalization_delay=10.0,
        processing_interval=1.0
    )
    
    # Optional: List and select audio input devices
    print("\n📱 Audio Device Management:")
    devices = transcriber.input_devices()
    
    if devices:
        print(f"\nFound {len(devices)} input devices")
        
        # Optionally set a specific device (uncomment to use)
        # device_id = 0  # Change this to your preferred device ID
        # if transcriber.set_input_device(device_id):
        #     print(f"Successfully set input device to ID {device_id}")
        # else:
        #     print(f"Failed to set input device to ID {device_id}")
    
    # Set up callback for LLM integration (optional)
    def my_llm_callback(text):
        print(f"🤖 [LLM Processing]: {text}")
        # Your LLM integration here
        return "processed"
    
    transcriber.set_def_callback(my_llm_callback)
    
    print("\n🎤 Starting transcription...")
    print("- Signal handling is built-in (Ctrl+C to stop gracefully)")
    print("- Audio device management is available")
    print("- LLM callback is configured")
    print("\nSpeak into your microphone...")
    
    # Start streaming - signal handling is automatic
    if transcriber.start_streaming():
        print("✅ Transcription started successfully")
        print("   Press Ctrl+C to stop")
        
        # The transcriber will handle everything automatically
        # including signal handling, so we can just wait
        try:
            import time
            while True:
                time.sleep(1)
                # Transcriber is running in background threads
                # Signal handling will stop it gracefully
                
        except KeyboardInterrupt:
            # This should be handled by the built-in signal handler
            # but we include this as a fallback
            print("\nFallback: Stopping transcription...")
            transcriber.stop_streaming()
            transcriber.close()
    else:
        print("❌ Failed to start transcription")


if __name__ == "__main__":
    main()