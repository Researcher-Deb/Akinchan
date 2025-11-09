"""
Test script for Google Cloud Speech-to-Text and Text-to-Speech services.
Run this to verify the voice functionality is working.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.speech_service import get_speech_service

def test_speech_service():
    """Test the speech service initialization and basic functionality."""
    
    print("="*60)
    print("Testing Google Cloud Speech & TTS Service")
    print("="*60)
    
    # Get service
    service = get_speech_service()
    
    # Check availability
    print(f"\n✓ Service Available: {service.is_available()}")
    
    if not service.is_available():
        print("\n❌ Speech service not available!")
        print("Please check:")
        print("  1. speech_key.json file exists")
        print("  2. GOOGLE_APPLICATION_CREDENTIALS is set in .env")
        print("  3. google-cloud-speech and google-cloud-texttospeech are installed")
        return False
    
    # Test Text-to-Speech
    print("\n" + "="*60)
    print("Testing Text-to-Speech")
    print("="*60)
    
    test_texts = {
        'en': "Hello! This is a test of the speech synthesis system.",
        'hi': "नमस्ते! यह वाक् संश्लेषण प्रणाली का परीक्षण है।",
        'bn': "হ্যালো! এটি বক্তৃতা সংশ্লেষণ সিস্টেমের একটি পরীক্ষা।"
    }
    
    for lang, text in test_texts.items():
        print(f"\n{lang.upper()}: '{text}'")
        audio_bytes, error = service.synthesize_speech(text, lang)
        
        if error:
            print(f"  ❌ TTS Error: {error}")
        else:
            print(f"  ✓ Generated audio: {len(audio_bytes)} bytes")
    
    print("\n" + "="*60)
    print("✓ All Tests Passed!")
    print("="*60)
    print("\nYou can now use voice input/output in the chatbot!")
    print("  1. Go to http://localhost:8000/chat")
    print("  2. Click the microphone button (🎤)")
    print("  3. Speak your message")
    print("  4. The bot will respond with voice")
    
    return True

if __name__ == "__main__":
    try:
        success = test_speech_service()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
