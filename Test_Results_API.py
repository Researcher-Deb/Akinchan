"""
Test script to check simulation results API
"""
import requests

BASE_URL = "http://localhost:8000"

def test_simulation_api():
    """Test the simulation details API"""
    
    print("="*60)
    print("Testing Simulation Results API")
    print("="*60)
    
    simulation_id = "study_of_immunox_101_7842"
    
    print(f"\n1. Testing GET /api/simulation/{simulation_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/api/simulation/{simulation_id}")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS!")
            print(f"   Response:")
            import json
            print(json.dumps(data, indent=2))
        else:
            print(f"   ❌ FAILED")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_simulation_api()
