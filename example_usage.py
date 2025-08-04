#!/usr/bin/env python3
"""
Example usage script demonstrating the new callback and pause/resume functionality
of the WhisperStreamingTranscriberWithSpecials class.

This example shows two main usage scenarios:
1. Real-time streaming with custom LLM integration
2. LLM input mode with pause/resume control
"""

import time
import signal
import sys

# Import the transcriber class
# Note: This will fail if Whisper dependencies aren't installed, which is expected
try:
    from main_stream import WhisperStreamingTranscriberWithSpecials
    TRANSCRIBER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Transcriber dependencies not available: {e}")
    print("This is expected in test environments. API examples will still be shown.\n")
    TRANSCRIBER_AVAILABLE = False
    WhisperStreamingTranscriberWithSpecials = None
except Exception as e:
    print(f"⚠️  Could not load transcriber: {e}")
    print("This is expected in test environments. API examples will still be shown.\n")
    TRANSCRIBER_AVAILABLE = False
    WhisperStreamingTranscriberWithSpecials = None


# Global variable to hold the transcriber instance for signal handling
current_transcriber = None

class LLMIntegrationExample:
    """Example class showing how to integrate with an LLM"""
    
    def __init__(self):
        self.conversation_history = []
        self.response_count = 0
    
    def process_speech_text(self, text):
        """
        Example LLM processing function
        This is where you would integrate with OpenAI, Claude, local LLM, etc.
        """
        print(f"\n🤖 [LLM PROCESSOR] Received: '{text}'")
        
        # Store in conversation history
        self.conversation_history.append({
            'timestamp': time.strftime('%H:%M:%S'),
            'user_input': text,
            'type': 'speech'
        })
        
        # Simulate LLM processing time
        print("🤖 [LLM PROCESSOR] Processing with LLM...")
        time.sleep(1)  # Simulate API call time
        
        # Example response generation (replace with actual LLM call)
        self.response_count += 1
        response = f"Response #{self.response_count}: I heard you say '{text}'. How can I help with that?"
        
        print(f"🤖 [LLM RESPONSE] {response}")
        
        # Store response in history
        self.conversation_history.append({
            'timestamp': time.strftime('%H:%M:%S'),
            'llm_response': response,
            'type': 'response'
        })
        
        return response
    
    def get_conversation_summary(self):
        """Get a summary of the conversation"""
        return {
            'total_exchanges': len([h for h in self.conversation_history if h['type'] == 'speech']),
            'history': self.conversation_history
        }


def example_real_time_mode():
    """
    Example 1: Real-time streaming mode with LLM integration
    Speech is continuously processed and sent to LLM immediately
    """
    print("=" * 60)
    print("EXAMPLE 1: REAL-TIME MODE WITH LLM INTEGRATION")
    print("=" * 60)
    print("In this mode, speech is continuously transcribed and immediately sent to LLM")
    print("Press 'Ctrl+C' to stop\n")
    
    # Create LLM integration instance
    llm_integration = LLMIntegrationExample()
    
    # Create transcriber
    try:
        transcriber = WhisperStreamingTranscriberWithSpecials(model_name="base.en")
        global current_transcriber
        current_transcriber = transcriber
        
        # Register the LLM callback
        # transcriber.set_llm_callback(llm_integration.process_speech_text)
        
        # Start streaming
        if transcriber.start_streaming():
            print("🎤 Listening... Speak into your microphone")
            print("   Speech will be automatically sent to LLM when sentences complete")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
                    # Could add periodic status checks here
                    if transcriber.is_running() and not transcriber.is_paused():
                        pass  # Everything is running normally
            
            except KeyboardInterrupt:
                print("\nStopping real-time mode...")
        
        # Stop and cleanup
        transcriber.stop_streaming()
        transcriber.close()
        
        # Show conversation summary
        summary = llm_integration.get_conversation_summary()
        print(f"\n📊 Session Summary: {summary['total_exchanges']} exchanges processed")
        
    except Exception as e:
        print(f"❌ Error in real-time mode example: {e}")
        print("Note: This example requires Whisper dependencies to run fully")


def example_llm_input_mode():
    """
    Example 2: LLM input mode with pause/resume control
    Transcriber pauses after each input, waits for LLM processing, then resumes
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: LLM INPUT MODE WITH PAUSE/RESUME")
    print("=" * 60)
    print("In this mode, transcriber pauses after each input for LLM processing")
    print("This simulates turn-based conversation with an LLM\n")
    
    class PausingLLMIntegration:
        def __init__(self, transcriber):
            self.transcriber = transcriber
            self.turn_count = 0
        
        def process_with_pause(self, text):
            """Process text and pause transcriber during LLM work"""
            self.turn_count += 1
            print(f"\n🎯 [TURN {self.turn_count}] User said: '{text}'")
            
            # Pause the transcriber while we process with LLM
            if self.transcriber.pause_streaming():
                print("⏸️  [PAUSED] Transcriber paused for LLM processing")
            
            # Simulate LLM processing
            print("🤖 [LLM] Processing your input...")
            time.sleep(2)  # Simulate longer LLM processing
            
            response = f"Turn {self.turn_count} response: That's interesting! Tell me more."
            print(f"🤖 [LLM] {response}")
            
            # Simulate speaking the response (pause before resuming)
            print("🔊 [SPEAKING] LLM is speaking response...")
            time.sleep(10)
            
            # Resume the transcriber for next input
            if self.transcriber.resume_streaming():
                print("▶️  [RESUMED] Ready for next input\n")
            
            return response
    
    # Create transcriber
    try:
        transcriber = WhisperStreamingTranscriberWithSpecials(model_name="base.en")
        global current_transcriber
        current_transcriber = transcriber
        
        # Create pausing LLM integration
        pausing_llm = PausingLLMIntegration(transcriber)
        
        # Register the callback
        transcriber.set_llm_callback(pausing_llm.process_with_pause)
        
        # Start streaming
        if transcriber.start_streaming():
            print("🎤 Listening in LLM input mode...")
            print("   Say something, then wait for LLM to respond before speaking again")
            
            # Run for demonstration (or until interrupted)
            try:
                start_time = time.time()
                while time.time() - start_time < 300:  # Run for 300 seconds in demo
                    time.sleep(10)

                    # Show status periodically
                    if int(time.time() - start_time) % 10 == 0:
                        status = "PAUSED" if transcriber.is_paused() else "LISTENING"
                        print(f"📊 [STATUS] {status} | Turns: {pausing_llm.turn_count}")
                
            except KeyboardInterrupt:
                print("\nStopping LLM input mode...")
        
        # Stop and cleanup
        transcriber.stop_streaming()
        transcriber.close()
        
        print(f"\n📊 LLM Input Mode Summary: {pausing_llm.turn_count} turns completed")
        
    except Exception as e:
        print(f"❌ Error in LLM input mode example: {e}")
        print("Note: This example requires Whisper dependencies to run fully")


def example_manual_control():
    """
    Example 3: Manual pause/resume control
    Shows how to manually control the transcriber state
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: MANUAL PAUSE/RESUME CONTROL")
    print("=" * 60)
    print("Demonstrates manual control of transcriber state\n")
    
    def simple_callback(text):
        print(f"📝 [CAPTURED] {text}")
    
    try:
        transcriber = WhisperStreamingTranscriberWithSpecials(model_name="base.en")
        global current_transcriber
        current_transcriber = transcriber
        transcriber.set_llm_callback(simple_callback)
        
        print("🎤 Starting transcriber...")
        if transcriber.start_streaming():
            print(f"Status - Running: {transcriber.is_running()}, Paused: {transcriber.is_paused()}")
            
            # Pause for 3 seconds
            print("\n⏸️  Pausing for 3 seconds...")
            transcriber.pause_streaming()
            print(f"Status - Running: {transcriber.is_running()}, Paused: {transcriber.is_paused()}")
            time.sleep(3)
            
            # Resume
            print("\n▶️  Resuming...")
            transcriber.resume_streaming()
            print(f"Status - Running: {transcriber.is_running()}, Paused: {transcriber.is_paused()}")
            time.sleep(2)
            
            # Stop completely
            print("\n⏹️  Stopping...")
            transcriber.stop_streaming()
            print(f"Status - Running: {transcriber.is_running()}, Paused: {transcriber.is_paused()}")
        
        transcriber.close()
        print("✅ Manual control example completed")
        
    except Exception as e:
        print(f"❌ Error in manual control example: {e}")
        print("Note: This example requires Whisper dependencies to run fully")


def main():
    """Main example runner"""
    print("🎯 Audio2Text Refactored - Usage Examples")
    print("🚀 Demonstrating callback and pause/resume functionality\n")
    
    # For demonstration purposes, we'll show the API usage
    # Note: These examples would require Whisper dependencies to run fully
    
    print("📋 Available examples:")
    print("1. Real-time mode with LLM integration")
    print("2. LLM input mode with pause/resume")
    print("3. Manual pause/resume control")
    print("\nNote: Full examples require Whisper dependencies")
    print("      This script shows the API structure and usage patterns\n")
    
    # Show API usage examples without actually running (since we may not have deps)
    try:
        # This would work with full dependencies:
        # example_real_time_mode()
        example_llm_input_mode()
        # example_manual_control()
        
        print("✅ API structure demonstrated successfully")
        print("\n🔧 To use in your project:")
        print("1. Install dependencies: pip install openai-whisper torch pyaudio numpy pynput")
        print("2. Import: from main_stream import WhisperStreamingTranscriberWithSpecials")
        print("3. Create instance: transcriber = WhisperStreamingTranscriberWithSpecials()")
        print("4. Register callback: transcriber.set_llm_callback(your_function)")
        print("5. Start streaming: transcriber.start_streaming()")
        print("6. Use pause/resume as needed: transcriber.pause_streaming(), transcriber.resume_streaming()")
        print("7. Stop when done: transcriber.stop_streaming()")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")


if __name__ == "__main__":
    # Handle interruption gracefully
    def signal_handler(sig, frame):
        print("\n\nExamples interrupted by user")
        
        # Properly clean up the transcriber if it exists
        global current_transcriber
        if current_transcriber is not None:
            try:
                print("Stopping transcriber and showing session summary...")
                current_transcriber.stop_streaming()
                
                # print("Transcribed Text:")
                all_text = current_transcriber.get_all_transcribed_text()
                # print(all_text)

                # print("Completed Sentences Report:")
                all_report = current_transcriber.get_completed_sentences()
                # print(all_report)

                current_transcriber.close()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    main()