"""
Test Gemini API Keys
Tests both primary and fallback API keys
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_key(api_key, key_name):
    """Test a single Gemini API key"""
    print(f"\n{'='*60}")
    print(f"Testing {key_name}")
    print(f"{'='*60}")
    print(f"Key: {api_key[:20]}...{api_key[-10:]}")
    
    try:
        import google.generativeai as genai
        
        # Configure with this key
        genai.configure(api_key=api_key)
        
        # Create model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Test with a simple prompt
        print("\nSending test request...")
        response = model.generate_content("Say hello in 5 words")
        
        print(f"✅ SUCCESS!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED!")
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Gemini API Key Validation")
    print("="*60)
    
    # Get keys from environment
    primary_key = os.getenv('GEMINI_API_KEY')
    fallback_key = os.getenv('GOOGLE_API_KEY')
    
    if not primary_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env")
        sys.exit(1)
    
    if not fallback_key:
        print("⚠️  WARNING: GOOGLE_API_KEY not found in .env (no fallback)")
    
    # Test primary key
    primary_works = test_gemini_key(primary_key, "Primary Key (GEMINI_API_KEY)")
    
    # Test fallback key
    fallback_works = False
    if fallback_key:
        fallback_works = test_gemini_key(fallback_key, "Fallback Key (GOOGLE_API_KEY)")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Primary Key (GEMINI_API_KEY):  {'✅ WORKING' if primary_works else '❌ NOT WORKING'}")
    print(f"Fallback Key (GOOGLE_API_KEY): {'✅ WORKING' if fallback_works else '❌ NOT WORKING'}")
    
    if primary_works:
        print("\n✅ PRIMARY KEY IS WORKING - No fallback needed")
    elif fallback_works:
        print("\n⚠️  PRIMARY KEY FAILED - FALLBACK KEY WILL BE USED")
    else:
        print("\n❌ BOTH KEYS FAILED - Please get a new API key from https://aistudio.google.com/apikey")
    
    print("="*60)
