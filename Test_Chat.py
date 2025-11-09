"""
Test script for Chat API functionality
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_chat_message():
    """Test sending a chat message"""
    print("\n" + "="*60)
    print("Testing Chat API")
    print("="*60)
    
    # Test data
    test_message = "Show my simulations"
    test_user_id = "1"
    test_username = "user1"
    
    print(f"\n1. Testing POST /api/chat/message")
    print(f"   Message: {test_message}")
    print(f"   User: {test_username} (ID: {test_user_id})")
    
    try:
        # Send chat message
        response = requests.post(
            f"{BASE_URL}/api/chat/message",
            data={
                "message": test_message,
                "user_id": test_user_id,
                "username": test_username
            },
            timeout=30
        )
        
        print(f"\n   Response Status: {response.status_code}")
        print(f"   Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n   ✅ SUCCESS!")
            print(f"   Response Data:")
            print(f"   - Success: {data.get('success')}")
            print(f"   - Language: {data.get('language')}")
            print(f"   - Action Type: {data.get('action_type')}")
            print(f"   - Response (first 200 chars): {data.get('response', '')[:200]}...")
            return True
        else:
            print(f"\n   ❌ FAILED!")
            print(f"   Response Text: {response.text}")
            
            try:
                error_data = response.json()
                print(f"   Error Details:")
                print(f"   - Success: {error_data.get('success')}")
                print(f"   - Error: {error_data.get('error')}")
                print(f"   - Message: {error_data.get('message')}")
            except:
                pass
            
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n   ❌ TIMEOUT! Request took longer than 30 seconds")
        return False
    except Exception as e:
        print(f"\n   ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_user_simulations():
    """Test getting user simulations"""
    print(f"\n2. Testing GET /api/chat/user-simulations")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/chat/user-simulations",
            params={"user_id": "1"}
        )
        
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS!")
            print(f"   Simulations count: {data.get('count')}")
            
            if data.get('simulations'):
                for i, sim in enumerate(data['simulations'][:3], 1):
                    print(f"   Simulation {i}:")
                    print(f"     - ID: {sim.get('simulation_id')}")
                    print(f"     - Drug: {sim.get('drug_name', 'N/A')}")
                    print(f"     - Patients: {sim.get('sample_size', 'N/A')}")
            return True
        else:
            print(f"   ❌ FAILED!")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False


def test_chat_history():
    """Test getting chat history"""
    print(f"\n3. Testing GET /api/chat/history")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/chat/history",
            params={"user_id": "1", "limit": 5}
        )
        
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS!")
            print(f"   History count: {data.get('count')}")
            return True
        else:
            print(f"   ❌ FAILED!")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Chat API Test Suite")
    print("="*60)
    print(f"Testing against: {BASE_URL}")
    print("Make sure the server is running: python run.py")
    print("="*60)
    
    # Run tests
    results = []
    
    # Test 1: User simulations (should work)
    results.append(("User Simulations", test_user_simulations()))
    
    # Test 2: Chat message (main test)
    results.append(("Chat Message", test_chat_message()))
    
    # Test 3: Chat history
    results.append(("Chat History", test_chat_history()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    # Exit code
    sys.exit(0 if passed == total else 1)
