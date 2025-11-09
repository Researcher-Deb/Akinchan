"""
Test script to verify Gemini API connectivity and diagnose application issues.
"""

import os
import sys
from dotenv import load_dotenv

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

print("=" * 70)
print("GEMINI API CONNECTION TEST")
print("=" * 70)

# Test 1: Check environment variables
print("\n1. ENVIRONMENT VARIABLES CHECK")
print("-" * 70)
api_key = os.getenv("GOOGLE_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
local_model = os.getenv("LOCAL_MODEL")

print(f"GOOGLE_API_KEY: {'✓ Set' if api_key else '✗ Not Set'}")
print(f"GEMINI_API_KEY: {'✓ Set' if gemini_key else '✗ Not Set'}")
print(f"LOCAL_MODEL: {local_model}")

if api_key:
    print(f"API Key (masked): {api_key[:20]}...{api_key[-4:]}")

# Test 2: Try importing google.generativeai
print("\n2. GOOGLE GENERATIVE AI LIBRARY CHECK")
print("-" * 70)
try:
    import google.generativeai as genai
    print("✓ google.generativeai library imported successfully")
    print(f"  Version: {genai.__version__ if hasattr(genai, '__version__') else 'Unknown'}")
except ImportError as e:
    print(f"✗ Failed to import google.generativeai: {e}")
    print("  Install with: pip install google-generativeai")
    exit(1)

# Test 3: Configure API and test connection
print("\n3. API CONFIGURATION & CONNECTION TEST")
print("-" * 70)

if not api_key:
    print("✗ No API key found. Cannot test connection.")
    exit(1)

try:
    # Configure with API key
    genai.configure(api_key=api_key)
    print("✓ API configured successfully")
    
    # Test 4: List available models
    print("\n4. AVAILABLE MODELS")
    print("-" * 70)
    try:
        models = genai.list_models()
        print("Available Gemini models:")
        for model in models:
            if 'gemini' in model.name.lower():
                print(f"  • {model.name}")
                print(f"    Display Name: {model.display_name}")
                print(f"    Supported Methods: {model.supported_generation_methods}")
                print()
    except Exception as e:
        print(f"⚠ Could not list models: {e}")
    
    # Test 5: Test Gemini 2.0 Flash Experimental
    print("\n5. TESTING GEMINI 2.0 FLASH EXPERIMENTAL")
    print("-" * 70)
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        print("✓ Model object created successfully")
        
        # Send a test prompt
        print("\nSending test prompt: 'What is 2+2?'")
        response = model.generate_content("What is 2+2? Answer in one sentence.")
        
        print(f"✓ Response received!")
        print(f"  Response text: {response.text}")
        print(f"  Response length: {len(response.text)} characters")
        
    except Exception as e:
        print(f"✗ Failed to use gemini-2.0-flash-exp: {e}")
        print("\nTrying alternative model: gemini-1.5-flash...")
        
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content("What is 2+2? Answer in one sentence.")
            print(f"✓ Alternative model works!")
            print(f"  Response: {response.text}")
        except Exception as e2:
            print(f"✗ Alternative model also failed: {e2}")
    
    # Test 6: Test with system instruction (like agents.py uses)
    print("\n6. TESTING WITH SYSTEM INSTRUCTION")
    print("-" * 70)
    
    try:
        model = genai.GenerativeModel(
            "gemini-2.0-flash-exp",
            system_instruction="You are a clinical trial research analyst."
        )
        print("✓ Model with system instruction created")
        
        response = model.generate_content(
            "Analyze the success rate of Phase II oncology trials in 2 sentences."
        )
        print(f"✓ Response received!")
        print(f"  Response: {response.text}")
        
    except Exception as e:
        print(f"✗ Failed with system instruction: {e}")
    
    # Test 7: Check API quota/limits
    print("\n7. API PROJECT INFORMATION")
    print("-" * 70)
    print(f"Project Name: projects/240351467368")
    print(f"Project Number: 240351467368")
    print(f"API Key Name: API key 1")
    
except Exception as e:
    print(f"✗ Configuration/Connection failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Test the application's agent implementation
print("\n8. TESTING APPLICATION AGENT IMPLEMENTATION")
print("-" * 70)

try:
    from app.agents import GOOGLE_ADK_AVAILABLE, ResearchAgent
    from app.models import AgentRequest
    
    print(f"GOOGLE_ADK_AVAILABLE: {GOOGLE_ADK_AVAILABLE}")
    
    if GOOGLE_ADK_AVAILABLE:
        print("✓ Attempting to create ResearchAgent...")
        agent = ResearchAgent()
        print(f"✓ ResearchAgent created. Model: {agent.model}")
        
        if agent.model:
            print("✓ Agent has valid model. Testing process()...")
            request = AgentRequest(
                agent_type="research",
                query="What is the average success rate of clinical trials?",
                context={}
            )
            response = agent.process(request)
            print(f"✓ Agent response received!")
            print(f"  Response: {response.response[:200]}...")
            print(f"  Confidence: {response.confidence}")
            print(f"  Sources: {response.sources}")
        else:
            print("✗ Agent model is None")
    else:
        print("✗ Google ADK not available in application")
        
except Exception as e:
    print(f"✗ Application agent test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Test AgentRouter
print("\n9. TESTING AGENT ROUTER")
print("-" * 70)

try:
    from app.agent_router import AgentRouter
    from app.models import AgentRequest
    
    status = AgentRouter.get_status()
    print(f"Router Status:")
    print(f"  Google ADK Available: {status['google_adk_available']}")
    print(f"  API Key Configured: {status['api_key_configured']}")
    print(f"  Active System: {status['active_system']}")
    print(f"  Fallback Available: {status['fallback_available']}")
    
    print("\n✓ Testing router with sample request...")
    request = AgentRequest(
        agent_type="research",
        query="What factors affect clinical trial success?",
        context={}
    )
    
    response = AgentRouter.process_request(request)
    print(f"✓ Router response received!")
    print(f"  Response: {response.response[:200]}...")
    print(f"  Confidence: {response.confidence}")
    print(f"  Sources: {response.sources}")
    
except Exception as e:
    print(f"✗ Router test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
